"""The backtest must be causal: changing data *after* a forecast origin must not change
any forecast issued at or before that origin."""
import numpy as np
import pandas as pd

from ua_alerts import evaluate as ev
from ua_alerts import models


def _series(n=420, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    level = 55 + 10 * np.sin(np.arange(n) / 40)
    return pd.Series(np.clip(level + rng.normal(0, 12, n), 3, None), index=idx)


def _fast_models():
    return [models.Naive(), models.SeasonalNaive7(), models.MovingAverage(7), models.SeasonalMean4w(), models.RegimeSeasonalMean(),
            models.ETS(seasonal=False), models.SARIMA(seasonal=False), models.GBMDirect()]


def test_backtest_predictions_do_not_depend_on_future_values():
    y = _series()
    test_end = y.index[-1]
    test_start = test_end - pd.Timedelta(days=13)
    cutoff = test_start + pd.Timedelta(days=6)           # corrupt everything strictly after this day

    y_corrupt = y.copy()
    y_corrupt.loc[cutoff + pd.Timedelta(days=1):] *= 5.0

    a = ev.backtest(y, _fast_models, test_start, test_end, verbose=False)
    b = ev.backtest(y_corrupt, _fast_models, test_start, test_end, verbose=False)

    key = ["model", "origin", "target", "h"]
    merged = a.merge(b, on=key, suffixes=("_a", "_b"))
    causal = merged[merged["origin"] <= cutoff]
    assert len(causal) > 0
    np.testing.assert_allclose(causal["y_pred_a"], causal["y_pred_b"], rtol=1e-9, atol=1e-9)
    # sanity: the check has teeth -- forecasts issued after the cutoff *do* change
    later = merged[merged["origin"] > cutoff]
    assert not np.allclose(later["y_pred_a"], later["y_pred_b"])


def test_models_return_requested_horizon_and_nonnegative():
    y = _series()
    for m in _fast_models():
        m.fit(y)
        fc = m.predict(y, 7)
        assert fc.shape == (7,) and np.all(np.isfinite(fc)) and np.all(fc >= 0), m.name
