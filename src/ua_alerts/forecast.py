"""Probabilistic outlook: will oblast X be under air raid alert during each of the next 24 hours?

What is predicted
-----------------
For every hour slot in the next 24 h (Kyiv local time, starting with the hour containing "now"):

    P(oblast is under alert for at least part of that hour)

plus one number for the whole window, P(at least one alert in the next 24 h).

How
---
This module holds the original per-oblast model and the shared feature code; the default model used by
``predict.py`` is the pooled one in :mod:`pooled` (all oblasts in one classifier, neighbour features, calibration).
The per-oblast model is two gradient-boosting classifiers trained on this oblast's own recent history:

* an hourly model, stacked over horizons (the horizon is a feature), covers all 24 slots;
* a window model answers "any alert in the whole 24 h window?" directly (summing hourly
  probabilities would overcount because alerts last several hours).

Features are computed at the *origin* (the last fully observed hour) and use nothing later:
current/recent alert state, hours since the last alert, current streak, alert onsets, rolling
alert fractions (3 h ... 4 weeks), nationwide activity (are many oblasts under alert right now,
i.e. a mass-attack wave), the target hour/weekday, and a 4-week same-hour climatology.

Shrinkage
---------
Hour-level alert timing is close to unpredictable, and the rows are heavily autocorrelated, so a
flexible model overfits easily. Before the final fit, the model is compared with plain same-hour
climatology on a validation window just before the origin and the forecast becomes
``w * model + (1 - w) * climatology`` with w chosen on that window. w = 0 means "no skill here".

Stale data
----------
The upstream file is refreshed about once a day, so "now" is usually several hours after the last
observed hour. Models are trained for exactly that gap (slots ``s_min..s_max`` hours after the
origin), so information fades toward climatology the staler the data is.

This is a statistical estimate of aggregate patterns, **not** a warning system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

from .data import LOCAL_TZ, AlertData
from .sources import ukrainian_to_region

SPAN = 24                 # hours in the forecast window
CLIM_DAYS = 28            # same-hour climatology length
MIN_ORIGIN = 24 * CLIM_DAYS + 24   # earliest usable origin index
MAX_STALENESS_H = 96      # refuse to forecast from data older than this

ORIGIN_COLS = [
    "occ0", "occ1", "occ2", "occ_m3", "occ_m6", "occ_m12", "occ_m24", "occ_m72", "occ_m168", "occ_m672",
    "onsets24", "onsets168", "hours_since_alert", "streak", "nat0", "nat_m6", "nat_m24", "origin_hour",
]


# ----------------------------------------------------------------------------- oblasts
def resolve_oblast(query: str, available: list[str]) -> str:
    """Map user input (English or Ukrainian, partial) to a dataset region name."""
    q = query.strip()
    low = q.lower()
    by_low = {a.lower(): a for a in available}
    if low in by_low:
        return by_low[low]
    ua = ukrainian_to_region(q)
    if ua in available:
        return ua
    norm = lambda s: "".join(ch for ch in s.lower().replace("oblast", "") if ch.isalpha())  # noqa: E731
    nq = norm(q)
    if nq in ("kyiv", "kiev", "kyivcity", "citykyiv"):
        return "Kyiv City"
    if nq in ("kyivska", "kyivoblast", "kievoblast", "kyivregion"):
        return "Kyivska oblast"
    cands = [a for a in available if norm(a) == nq] or [a for a in available if norm(a).startswith(nq)] \
        or [a for a in available if nq and nq in norm(a)]
    if len(cands) == 1:
        return cands[0]
    hint = f" Did you mean: {', '.join(cands)}?" if cands else ""
    raise ValueError(f"Could not match oblast {query!r}.{hint} Use --list to see valid names.")


# --------------------------------------------------------------------------- features
@dataclass
class Prepared:
    region: str
    idx: pd.DatetimeIndex        # UTC hour starts, last = origin hour
    occ: np.ndarray              # fraction of each hour under alert (this region)
    a: np.ndarray                # 1 if any alert during that hour
    nat: np.ndarray              # share of oblasts under alert (national activity)
    lh: np.ndarray               # local hour-of-day, extended 200 h past the origin
    dow: np.ndarray              # local weekday, extended likewise
    cs_occ: np.ndarray = field(repr=False, default=None)
    cs_a: np.ndarray = field(repr=False, default=None)
    cs_nat: np.ndarray = field(repr=False, default=None)
    cs_on: np.ndarray = field(repr=False, default=None)
    hsl: np.ndarray = field(repr=False, default=None)
    streak: np.ndarray = field(repr=False, default=None)
    clim: np.ndarray = field(repr=False, default=None)


def prepare(data: AlertData, region: str, as_of: pd.Timestamp | None = None) -> Prepared:
    # Origin = the last hour that is *complete* in the data (as_of rounded down, minus one hour).
    # ``as_of`` lets callers replay history: only data before that moment is used.
    origin = (as_of or data.as_of).floor("h") - pd.Timedelta(hours=1)
    hr = data.hourly_region.loc[:origin]
    if region not in hr.columns:
        raise ValueError(f"{region!r} not in data")
    occ = hr[region].to_numpy(float)
    a = (occ > 1e-9).astype(float)
    nat = (hr.sum(axis=1) / hr.shape[1]).to_numpy(float)
    T = len(a)
    ext = pd.date_range(hr.index[0], periods=T + 200, freq="h", tz="UTC").tz_convert(LOCAL_TZ)
    P = Prepared(region, hr.index, occ, a, nat, ext.hour.to_numpy(), ext.dayofweek.to_numpy())

    z = np.zeros(1)
    P.cs_occ, P.cs_a, P.cs_nat = (np.concatenate([z, np.cumsum(x)]) for x in (occ, a, nat))
    onset = np.zeros(T)
    onset[1:] = a[1:] * (1 - a[:-1])
    P.cs_on = np.concatenate([z, np.cumsum(onset)])
    ar = np.arange(T)
    P.hsl = np.minimum(ar - np.maximum.accumulate(np.where(a > 0, ar, -10**6)), 168).astype(float)
    last_zero = np.maximum.accumulate(np.where(a == 0, ar, -1))
    P.streak = np.minimum(np.where(a > 0, ar - last_zero, 0), 48).astype(float)
    M = np.zeros(T)
    for k in range(CLIM_DAYS):                      # mean of the same hour over the last 28 days
        M[24 * k:] += a[: T - 24 * k]
    M /= CLIM_DAYS
    M[: 24 * (CLIM_DAYS - 1)] = np.nan
    P.clim = M
    return P


def _rmean(cs: np.ndarray, t: np.ndarray, w: int) -> np.ndarray:
    return (cs[t + 1] - cs[t + 1 - w]) / w


def origin_features(P: Prepared, t: np.ndarray) -> np.ndarray:
    cols = [P.occ[t], P.occ[t - 1], P.occ[t - 2]]
    cols += [_rmean(P.cs_occ, t, w) for w in (3, 6, 12, 24, 72, 168, 672)]
    cols += [P.cs_on[t + 1] - P.cs_on[t + 1 - 24], P.cs_on[t + 1] - P.cs_on[t + 1 - 168]]
    cols += [P.hsl[t], P.streak[t], P.nat[t], _rmean(P.cs_nat, t, 6), _rmean(P.cs_nat, t, 24), P.lh[t]]
    return np.column_stack(cols)


def hourly_matrix(P: Prepared, t: np.ndarray, s_values) -> tuple[np.ndarray, np.ndarray]:
    """Rows = (origin t, offset s). Target = alert during hour t+s. The LAST column is climatology."""
    X, y, base = [], [], origin_features(P, t)
    for s in s_values:
        u = t + s
        clim = P.clim[u - 24 * int(np.ceil(s / 24))]  # same-hour rate, only data <= t
        X.append(np.column_stack([base, np.full(len(t), s), P.lh[u], P.dow[u], clim]))
        y.append(P.a[u] if u.max() < len(P.a) else np.full(len(t), np.nan))
    return np.vstack(X), np.concatenate(y)


def window_matrix(P: Prepared, t: np.ndarray, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows = origins. Target = any alert in hours t+gap+1 .. t+gap+24."""
    start = t + gap + 1
    X = np.column_stack([origin_features(P, t), P.lh[start], P.dow[start]])
    end = start + SPAN
    y = (P.cs_a[end] - P.cs_a[start] > 0).astype(float) if end.max() < len(P.cs_a) else np.full(len(t), np.nan)
    return X, y


class _Const:
    def __init__(self, p): self.p = float(p)
    def predict_proba(self, X): return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


DEFAULT_CONFIG = dict(learning_rate=0.05, max_iter=80, max_depth=3, min_samples_leaf=400, l2_regularization=5.0)


def _fit_clf(X, y, seed=0):
    if len(np.unique(y)) < 2:
        return _Const(y.mean())
    clf = HistGradientBoostingClassifier(**DEFAULT_CONFIG, random_state=seed)
    return clf.fit(X, y)


def _train_origins(T: int, s_max: int, train_days: int, stride: int, t_hi: int | None = None) -> np.ndarray:
    hi = (T - 1 - s_max) if t_hi is None else t_hi
    lo = max(MIN_ORIGIN, hi - train_days * 24)
    return np.arange(lo, hi + 1, stride)


@dataclass
class Fitted:
    hourly: object
    window: object
    n_train: int
    blend: float          # weight on the model; 1 - blend goes to same-hour climatology
    val_skill: float      # Brier skill of the *blended* hourly forecast on the validation window


def _clim_col(X: np.ndarray, fill: float) -> np.ndarray:
    return np.nan_to_num(X[:, -1], nan=fill)


def _fit_raw(P, s_min, train_days, stride, t_hi, seed):
    s_max = s_min + SPAN - 1
    t = _train_origins(len(P.a), s_max, train_days, stride, t_hi)
    Xh, yh = hourly_matrix(P, t, range(s_min, s_max + 1))
    Xw, yw = window_matrix(P, t, s_min - 1)
    return _fit_clf(Xh, yh, seed), _fit_clf(Xw, yw, seed), len(t)


def fit(P: Prepared, s_min: int, train_days: int = 365, stride: int = 2, t_hi: int | None = None,
        seed: int = 0, val_days: int = 28) -> Fitted:
    """Fit the models, choosing how much to trust them on a validation window *before* the origin.

    1. Train on origins that end ``val_days`` before the newest usable origin.
    2. On the last ``val_days`` of origins compare model vs same-hour climatology and pick the blend
       weight w in [0, 1] minimising Brier score.
    3. Retrain on everything up to the newest usable origin; use the weight from step 2.
    """
    s_max = s_min + SPAN - 1
    hi = (len(P.a) - 1 - s_max) if t_hi is None else t_hi
    v_lo = hi - val_days * 24 + 1
    fill = float(P.a[: hi + 1].mean())
    blend, val_skill = 1.0, float("nan")
    try:
        m_val, _, _ = _fit_raw(P, s_min, train_days, stride, v_lo - 1 - s_max, seed)
        tv = np.arange(v_lo, hi + 1, 2)
        Xv, yv = hourly_matrix(P, tv, range(s_min, s_max + 1))
        pm, pc = m_val.predict_proba(Xv)[:, 1], _clim_col(Xv, fill)
        grid = np.linspace(0, 1, 11)
        briers = [np.mean((w * pm + (1 - w) * pc - yv) ** 2) for w in grid]
        blend = float(grid[int(np.argmin(briers))])
        val_skill = float(1 - min(briers) / max(np.mean((pc - yv) ** 2), 1e-12))
    except Exception:                      # not enough history for a validation split: be conservative
        blend = 0.5
    hourly, window, n = _fit_raw(P, s_min, train_days, stride, t_hi, seed)
    return Fitted(hourly, window, n, blend, val_skill)


# --------------------------------------------------------------------------- forecast
@dataclass
class ForecastResult:
    oblast: str
    generated_at: pd.Timestamp          # UTC
    data_as_of: pd.Timestamp            # UTC
    data_age_hours: float
    slots: pd.DataFrame                 # start_local, end_local, probability
    p_any: float
    expected_alert_hours: float         # expected NUMBER OF HOUR SLOTS touched by an alert (not duration)
    train_origins: int
    train_days: int
    blend: float                        # weight on the model vs climatology (chosen on validation data)
    source: str
    recent_state: dict
    warnings: list[str]
    live: list[dict] | None = None
    model: str = "per-oblast"

    def to_dict(self) -> dict:
        return {
            "oblast": self.oblast,
            "generated_at_utc": self.generated_at.isoformat(),
            "data_as_of_utc": self.data_as_of.isoformat(),
            "data_age_hours": round(self.data_age_hours, 1),
            "source": self.source,
            "model": self.model,
            "model_weight_vs_climatology": self.blend,
            "p_at_least_one_alert_in_window": round(self.p_any, 4),
            "expected_hour_slots_touched_by_alert": round(self.expected_alert_hours, 2),
            "slots": [{"start_local": r.start_local.isoformat(), "end_local": r.end_local.isoformat(),
                       "probability": round(r.probability, 4)} for r in self.slots.itertuples()],
            "recent_state": self.recent_state,
            "warnings": self.warnings,
        }


def predict_next_24h(data: AlertData, oblast: str, now: pd.Timestamp | None = None,
                     train_days: int = 365, live: list[dict] | None = None,
                     model: str = "pooled") -> ForecastResult:
    """Forecast the 24 hourly slots starting with the hour that contains ``now`` (default: the clock).

    If ``now`` is earlier than the newest data, the call is a *replay*: only data known at ``now``
    is used (useful for checking what the tool would have said).
    """
    if now is None:
        now = pd.Timestamp.now(tz="UTC")
    elif now.tzinfo is None:                      # a naive timestamp is read as Kyiv local time
        now = now.tz_localize(LOCAL_TZ)
    now = now.tz_convert("UTC")
    region = resolve_oblast(oblast, list(data.hourly_region.columns))
    replay = now < data.as_of
    as_of = now if replay else data.as_of
    P = prepare(data, region, as_of=as_of)
    t = len(P.a) - 1
    origin_hour = P.idx[t]
    first_slot = now.floor("h")
    s_min = max(1, int((first_slot - origin_hour) / pd.Timedelta(hours=1)))
    age_h = (now - as_of).total_seconds() / 3600
    if s_min > MAX_STALENESS_H:
        raise RuntimeError(
            f"The {data.source} dataset is {age_h:.0f} h old (limit {MAX_STALENESS_H} h), so a forecast for the next 24 h "
            "would be mostly guesswork. Try --refresh; if the upstream has stopped updating, run `python validate.py`. "
            "(The 'official' dataset typically lags by weeks - use --source volunteer for forecasting.)")
    warns = list(data.warnings)
    if replay:
        warns.append(f"REPLAY: --now is in the past; only data before {now:%Y-%m-%d %H:%M} UTC was used.")
    if age_h > 30:
        warns.append(f"Data is {age_h:.0f} h old; the first hours of the window are mostly climatology.")

    if model == "pooled":
        from . import pooled as PL
        W = PL.build_world(data, as_of=as_of)
        p, pf = PL.forecast_slots(W, region, s_min, train_days)
        blend, n_train = float(pf.blend[W.regions.index(region)]), pf.n_train
        window = PL.window_model_for_forecast(P, s_min, train_days)
        Xw = np.column_stack([origin_features(P, np.array([t])), P.lh[t + s_min], P.dow[t + s_min]])
    elif model == "per-oblast":
        fitted = fit(P, s_min, train_days)
        hourly, window, n_train, blend = fitted.hourly, fitted.window, fitted.n_train, fitted.blend
        s_vals = np.arange(s_min, s_min + SPAN)
        base = origin_features(P, np.array([t]))
        Xh = np.vstack([np.column_stack([base, [[s, P.lh[t + s], P.dow[t + s],
                        P.clim[t + s - 24 * int(np.ceil(s / 24))]]]]) for s in s_vals])
        p = blend * hourly.predict_proba(Xh)[:, 1] + (1 - blend) * _clim_col(Xh, float(P.a.mean()))
        Xw = np.column_stack([base, P.lh[t + s_min], P.dow[t + s_min]])
    else:
        raise ValueError(f"unknown model {model!r}; use 'pooled' or 'per-oblast'")
    s_vals = np.arange(s_min, s_min + SPAN)
    p_any = float(window.predict_proba(Xw)[:, 1][0])
    p_any = max(p_any, float(p.max()))       # logical floor: at least as likely as the single likeliest hour

    starts = pd.DatetimeIndex([origin_hour + pd.Timedelta(hours=int(s)) for s in s_vals]).tz_convert(LOCAL_TZ)
    slots = pd.DataFrame({"start_local": starts, "end_local": starts + pd.Timedelta(hours=1), "probability": p})
    state = {
        "active_in_last_observed_hour": bool(P.a[t]),
        "hours_since_last_alert": int(P.hsl[t]) if P.hsl[t] < 168 else None,
        "alert_hours_last_24h": round(float(P.occ[t - 23:t + 1].sum()), 1),
        "alert_hours_last_7d": round(float(P.occ[t - 167:t + 1].sum()), 1),
        "oblasts_under_alert_nationally_last_hour": int(round(P.nat[t] * data.hourly_region.shape[1])),
        "oblasts_in_data": int(data.hourly_region.shape[1]),
        "typical_alert_slots_per_24h_last_28d": round(float(24 * P.a[t - 671:t + 1].mean()), 1),
        "typical_alert_duration_h_per_24h_last_28d": round(float(P.occ[t - 671:t + 1].sum() / 28), 1),
        "days_with_alert_last_28d": int((P.a[t - 671:t + 1].reshape(28, 24).sum(axis=1) > 0).sum()),
    }
    return ForecastResult(region, now, as_of, age_h, slots, p_any, float(p.sum()), n_train, train_days,
                          blend, data.source, state, warns, live, model)


# ---------------------------------------------------------------------------- backtest
def _scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    out = {"Brier": float(np.mean((p - y) ** 2)), "LogLoss": float(log_loss(y, p, labels=[0, 1]))}
    out["AUC"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return out


@dataclass
class BacktestResult:
    oblast: str
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_days: int
    n_predictions: int
    base_rate: float
    hourly: pd.DataFrame         # model vs baselines on the hourly task
    by_horizon: pd.DataFrame     # Brier per horizon bucket, model vs climatology
    reliability: pd.DataFrame    # calibration bins
    by_hour: pd.DataFrame        # mean predicted vs observed by local hour
    window: pd.DataFrame         # "any alert in the next 24 h" task
    blend: float = 1.0           # model weight chosen on the validation window
    val_skill: float = float("nan")


MODEL_ROW = "model (GBM + climatology blend)"


def backtest(data: AlertData, oblast: str, test_days: int = 28, train_days: int = 365, seed: int = 0) -> BacktestResult:
    """Train on everything before the test window, then play the last ``test_days`` days forward.

    Every hour in the test window is a simulated "app run": features from data up to that hour,
    forecast for the next 24 slots, scored against what actually happened (fresh-data case, no gap).
    The model is trained once (not refit during the window), as a real user running it would not
    have access to the future either.
    """
    region = resolve_oblast(oblast, list(data.hourly_region.columns))
    P = prepare(data, region)
    T = len(P.a)
    t_first = T - 1 - SPAN - test_days * 24 + 1       # first test origin
    t_test = np.arange(t_first, T - SPAN)             # last origin has all 24 targets observed
    t_hi = t_first - 1 - SPAN                         # training targets end before the first test origin
    fitted = fit(P, 1, train_days, stride=2, t_hi=t_hi, seed=seed)
    hourly, window = fitted.hourly, fitted.window

    X, y = hourly_matrix(P, t_test, range(1, SPAN + 1))
    s_col = len(ORIGIN_COLS)
    horizon = X[:, s_col].astype(int)
    p_raw = hourly.predict_proba(X)[:, 1]
    clim = np.nan_to_num(X[:, -1], nan=float(P.a[:t_first].mean()))
    p = fitted.blend * p_raw + (1 - fitted.blend) * clim
    recent = X[:, ORIGIN_COLS.index("occ_m168")]
    train_y = hourly_matrix(P, _train_origins(T, SPAN, train_days, 2, t_hi), range(1, SPAN + 1))[1]
    return assemble_backtest(P, region, t_test, t_hi, train_days, y, horizon, p_raw, p, clim, recent,
                             X[:, s_col + 1].astype(int), float(train_y.mean()), window,
                             fitted.blend, fitted.val_skill)


def assemble_backtest(P, region, t_test, t_hi, train_days, y, horizon, p_raw, p, clim, recent, lh_t, base,
                      window_model, blend, val_skill, raw_label="model (GBM alone)") -> "BacktestResult":
    """Score the hourly forecasts and build every backtest table (shared by the per-oblast and pooled models)."""
    T = len(P.a)
    rows = {MODEL_ROW: p, raw_label: p_raw, "climatology (28d same hour)": clim,
            "recent rate (last 7 days)": np.clip(recent, 0, 1), "base rate (training mean)": np.full(len(y), base)}
    hourly_tbl = pd.DataFrame({k: _scores(y, v) for k, v in rows.items()}).T
    cb = hourly_tbl.loc["climatology (28d same hour)", "Brier"]
    hourly_tbl["skill_vs_clim"] = 1 - hourly_tbl["Brier"] / cb if cb > 1e-6 else np.nan   # undefined if nothing happened

    bh = []
    for lo, hi in [(1, 3), (4, 12), (13, 24)]:
        m = (horizon >= lo) & (horizon <= hi)
        bm, bc = _scores(y[m], p[m])["Brier"], _scores(y[m], clim[m])["Brier"]
        bh.append({"horizon_h": f"{lo}-{hi}", "Brier_model": bm, "Brier_climatology": bc,
                   "skill_vs_clim": 1 - bm / bc if bc > 1e-6 else np.nan})
    by_h = pd.DataFrame(bh).set_index("horizon_h")

    bins = np.linspace(0, 1, 11)
    cat = pd.cut(p, bins, include_lowest=True)
    rel = pd.DataFrame({"p": p, "y": y, "bin": cat}).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    rel.index = [f"{max(iv.left, 0):.1f}-{iv.right:.1f}" for iv in rel.index]

    by_hour = pd.DataFrame({"hour": lh_t, "pred": p, "obs": y}).groupby("hour").mean()

    Xw, yw = window_matrix(P, t_test, 0)
    pw = window_model.predict_proba(Xw)[:, 1]
    wbase = float(window_matrix(P, _train_origins(T, SPAN, train_days, 2, t_hi), 0)[1].mean())
    win = pd.DataFrame({"model (GBM)": _scores(yw, pw),
                        "base rate (training mean)": _scores(yw, np.full(len(yw), wbase))}).T
    win["observed_rate"] = float(yw.mean())

    return BacktestResult(region, P.idx[t_test[0]], P.idx[T - 1], train_days, len(y), float(y.mean()),
                          hourly_tbl, by_h, rel, by_hour, win, blend, val_skill)
