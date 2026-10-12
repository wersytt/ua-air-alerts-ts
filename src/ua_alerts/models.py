"""Forecasters for the daily alert-hours series.

Every model implements the same tiny interface:

    fit(history)                 # called only when the model is due for (re)fitting
    predict(history, horizon)    # -> array of the next `horizon` daily values

`history` is a pandas Series with a daily DatetimeIndex that ends on the forecast origin.
Models never see anything after the origin, which is what makes the backtest honest.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

SEASON = 7


class Forecaster:
    name = "base"
    refit_every = 1  # in origins (days); 1 = refit at every origin

    def fit(self, history: pd.Series) -> None:  # noqa: D401
        pass

    def predict(self, history: pd.Series, horizon: int) -> np.ndarray:
        raise NotImplementedError


# --------------------------------------------------------------------------- baselines
class Naive(Forecaster):
    name = "Naive"

    def predict(self, history, horizon):
        return np.repeat(history.iloc[-1], horizon)


class SeasonalNaive7(Forecaster):
    name = "SeasonalNaive7"

    def predict(self, history, horizon):
        last_week = history.to_numpy()[-SEASON:]
        return np.resize(last_week, horizon)


class MovingAverage(Forecaster):
    def __init__(self, window: int):
        self.window = window
        self.name = f"MovingAvg{window}"

    def predict(self, history, horizon):
        return np.repeat(history.iloc[-self.window:].mean(), horizon)


class SeasonalMean4w(Forecaster):
    """Average of the same weekday over the last 4 weeks."""
    name = "SeasonalMean4w"

    def predict(self, history, horizon):
        y = history.to_numpy()[-4 * SEASON:].reshape(4, SEASON)
        return np.resize(y.mean(axis=0), horizon)


class RegimeSeasonalMean(Forecaster):
    """Regime-aware same-weekday mean.

    Change points (mean shifts) are detected in the last ``window`` days of the history. The forecast is the
    weekday-profile mean over the *current regime only* (at least ``min_days`` days, so a regime that has just
    begun borrows a little from the one before). A shift is only confirmed ``min_size`` days after it
    happened, so the model lags a sudden change by about three weeks - it avoids averaging across an old regime.
    """
    name = "RegimeSeasonalMean"
    refit_every = 1

    def __init__(self, window: int = 548, min_days: int = 28, min_size: int = 21, penalty_scale: float = 3.0):
        self.window, self.min_days, self.min_size, self.penalty_scale = window, min_days, min_size, penalty_scale

    def predict(self, history, horizon):
        from .changepoints import detect
        y = history.iloc[-self.window:]
        cps = detect(y.to_numpy(float), self.min_size, self.penalty_scale)
        start = min(cps[-1] if cps else 0, len(y) - self.min_days)
        seg = y.iloc[max(start, 0):]
        profile = seg.groupby(seg.index.dayofweek).mean().reindex(range(7)).fillna(seg.mean())
        level = seg.mean()
        # weekday shape relative to the regime mean, shrunk when the regime is short
        w = min(1.0, len(seg) / 56)
        out = []
        for h in range(1, horizon + 1):
            d = (history.index[-1] + pd.Timedelta(days=h)).dayofweek
            out.append(level + w * (profile[d] - level))
        return np.clip(np.array(out), 0, None)


# ------------------------------------------------------------------ statistical models
class ETS(Forecaster):
    """Exponential smoothing with damped additive trend, optionally weekly seasonality."""

    def __init__(self, seasonal: bool, window: int = 365):
        self.seasonal, self.window = seasonal, window
        self.name = "ETS-HW7" if seasonal else "ETS-damped"

    def predict(self, history, horizon):
        y = history.iloc[-self.window:].to_numpy(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    y, trend="add", damped_trend=True,
                    seasonal="add" if self.seasonal else None,
                    seasonal_periods=SEASON if self.seasonal else None,
                    initialization_method="estimated",
                ).fit()
                fc = model.forecast(horizon)
            if not np.all(np.isfinite(fc)):
                raise ValueError("non-finite forecast")
        except Exception:  # fall back rather than crash a long backtest
            fc = MovingAverage(7).predict(history, horizon)
        return np.clip(fc, 0, None)


class SARIMA(Forecaster):
    """ARIMA(1,0,1) with constant; optional (1,0,1,7) seasonal part."""

    def __init__(self, seasonal: bool, window: int = 365):
        self.seasonal, self.window = seasonal, window
        self.name = "SARIMA" if seasonal else "ARIMA"

    def predict(self, history, horizon):
        y = history.iloc[-self.window:].to_numpy(float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(
                    y, order=(1, 0, 1),
                    seasonal_order=(1, 0, 1, SEASON) if self.seasonal else (0, 0, 0, 0),
                    trend="c",
                ).fit()
                fc = model.forecast(horizon)
            if not np.all(np.isfinite(fc)):
                raise ValueError("non-finite forecast")
        except Exception:
            fc = MovingAverage(7).predict(history, horizon)
        return np.clip(fc, 0, None)


# ---------------------------------------------------------------- machine learning model
class GBMDirect(Forecaster):
    """Direct multi-horizon gradient boosting on lag features.

    One model per horizon h. Features are built *at the origin t* using only y[<= t]:
      * the last 14 daily values, rolling means (7/14/28d) and 7d std
      * the value of the same weekday 1..4 weeks before the target day (always <= t for h <= 7)
      * the target day's weekday
    The model predicts the deviation of y[t+h] from the 7-day mean, which lets a tree model
    cope with level drift (trees can't extrapolate beyond levels seen in training).
    """
    refit_every = 7

    def __init__(self, max_horizon: int = 7, train_window: int = 730, seed: int = 0):
        self.max_horizon, self.train_window, self.seed = max_horizon, train_window, seed
        self.models: dict[int, HistGradientBoostingRegressor] = {}
        self.name = "GBM"

    @staticmethod
    def _features(y: np.ndarray, dates: pd.DatetimeIndex, t: np.ndarray, h: int) -> np.ndarray:
        csum = np.concatenate([[0.0], np.cumsum(y)])

        def roll_mean(w):
            return (csum[t + 1] - csum[t + 1 - w]) / w

        cols = [y[t - k] for k in range(14)]
        cols += [roll_mean(7), roll_mean(14), roll_mean(28)]
        cols.append(np.array([y[i - 6:i + 1].std() for i in t]))
        cols += [y[t + h - SEASON * m] for m in (1, 2, 3, 4)]
        cols.append((dates[t] + pd.Timedelta(days=h)).dayofweek.to_numpy())
        return np.column_stack(cols)

    def fit(self, history):
        y = history.to_numpy(float)
        T = len(y)
        self.models = {}
        for h in range(1, self.max_horizon + 1):
            t_all = np.arange(27, T - h)  # need 28 days of history; target index t+h <= T-1
            t = t_all[-self.train_window:]
            X = self._features(y, history.index, t, h)
            base = X[:, 14]  # rolling 7-day mean
            target = y[t + h] - base
            m = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.04, max_iter=250, max_depth=3,
                min_samples_leaf=25, l2_regularization=2.0, random_state=self.seed,
            )
            m.fit(X, target)
            self.models[h] = m

    def predict(self, history, horizon):
        y = history.to_numpy(float)
        t = np.array([len(y) - 1])
        out = []
        for h in range(1, horizon + 1):
            X = self._features(y, history.index, t, h)
            out.append(X[0, 14] + self.models[h].predict(X)[0])
        return np.clip(np.array(out), 0, None)


def default_models(include_slow: bool = True) -> list[Forecaster]:
    models: list[Forecaster] = [
        Naive(), SeasonalNaive7(), MovingAverage(7), MovingAverage(28), SeasonalMean4w(), RegimeSeasonalMean(),
        ETS(seasonal=False), ETS(seasonal=True), SARIMA(seasonal=False),
    ]
    if include_slow:
        models.append(SARIMA(seasonal=True))
    models.append(GBMDirect())
    return models

