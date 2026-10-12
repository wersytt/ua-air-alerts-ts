import numpy as np
import pandas as pd

from ua_alerts import data as D
from ua_alerts import forecast as F
from test_forecast import _make_csv


def _write_official(path, rows):
    pd.DataFrame(rows, columns=["oblast", "raion", "hromada", "level", "started_at", "finished_at", "source"]).to_csv(path, index=False)


def test_combined_uses_official_before_seam_and_volunteer_after(tmp_path):
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    seam = t0 + pd.Timedelta(days=10)
    off = [("Kyiv City", "", "", "oblast", t0 + pd.Timedelta(days=d, hours=5), t0 + pd.Timedelta(days=d, hours=7), "official")
           for d in range(10)]
    _write_official(tmp_path / "o.csv", off)
    pd.DataFrame([
        ("Kyiv City", t0 + pd.Timedelta(days=2), t0 + pd.Timedelta(days=2, hours=3), False),            # before seam: ignored
        ("Kyiv City", seam - pd.Timedelta(hours=1), seam + pd.Timedelta(hours=2), False),               # straddles: clipped
        ("Kyiv City", seam + pd.Timedelta(days=1), seam + pd.Timedelta(days=1, hours=1), False),        # after: kept
    ], columns=["region", "started_at", "finished_at", "naive"]).to_csv(tmp_path / "v.csv", index=False)
    alerts, info = D.load_combined(tmp_path / "o.csv", tmp_path / "v.csv")
    off_end = t0 + pd.Timedelta(days=9, hours=7)
    assert info["seam"] == off_end
    assert info["volunteer_tail_rows"] == 2
    assert len(alerts) == 12                                        # 10 official + 2 volunteer tail rows
    assert not ((alerts["started_at"] > off_end - pd.Timedelta(days=7)) & (alerts["started_at"] < off_end)
                & (alerts["finished_at"] > off_end + pd.Timedelta(hours=1))).any()


def test_seam_diagnostics_flags_level_shift():
    idx = pd.date_range("2026-01-01", periods=24 * 60, freq="h", tz="UTC")
    h = pd.DataFrame({"A": 0.0, "B": 0.0}, index=idx)
    seam = idx[24 * 30]
    h.loc[:seam, "A"] = 0.8; h.loc[seam:, "A"] = 0.1       # big drop
    h.loc[:, "B"] = 0.3                                    # stable
    d = D.seam_diagnostics(h, seam, idx[-1] + pd.Timedelta(hours=1))
    assert bool(d.loc["A", "level_shift"]) and not bool(d.loc["B", "level_shift"])
    assert D.seam_diagnostics(h, seam, seam + pd.Timedelta(hours=5)) is None      # too little data after the seam


def test_changepoints_find_known_shifts_and_ignore_noise():
    from ua_alerts import changepoints as C
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(50, 6, 200), rng.normal(80, 6, 150), rng.normal(55, 6, 120)])
    cps = C.detect(x)
    assert len(cps) == 2 and abs(cps[0] - 200) <= 5 and abs(cps[1] - 350) <= 5
    assert C.detect(rng.normal(50, 6, 400)) == []
    assert C.detect(x[:20]) == []                                     # too short for a regime


def test_regime_model_is_causal_and_tracks_new_level():
    from ua_alerts import models
    idx = pd.date_range("2025-01-01", periods=300, freq="D")
    rng = np.random.default_rng(0)
    y = pd.Series(np.r_[rng.normal(40, 3, 200), rng.normal(90, 3, 100)], index=idx).clip(lower=0)
    m = models.RegimeSeasonalMean()
    fc = m.predict(y, 7)
    assert fc.shape == (7,) and np.all(fc > 70)                        # level is the new regime, not the blend
    y2 = y.copy(); y2.iloc[-1] = 1e6
    assert not np.allclose(m.predict(y2, 7), fc)                       # uses the latest value (sanity: not constant)


def test_interval_backtest_uses_only_known_errors_and_covers_roughly():
    from ua_alerts import evaluate as ev, models
    idx = pd.date_range("2025-01-01", periods=400, freq="D")
    rng = np.random.default_rng(1)
    y = pd.Series(60 + rng.normal(0, 8, 400), index=idx)
    make = lambda: [models.MovingAverage(7), models.SeasonalMean4w()]
    ts, te = idx[-40], idx[-1]
    rows, summ = ev.interval_backtest(y, make, "MovingAvg7", ts, te, horizon=3)
    assert (rows["lo80"] <= rows["hi80"]).all() and (rows["lo50"] >= rows["lo80"]).all()
    assert (rows["origin"] < rows["target"]).all()
    assert 0.55 < summ.loc[0.8, "coverage"] < 0.99
    # corrupting the last target days must not change an interval issued earlier
    y2 = y.copy(); y2.iloc[-3:] *= 10
    rows2, _ = ev.interval_backtest(y2, make, "MovingAvg7", ts, te, horizon=3)
    k = rows.merge(rows2, on=["target", "h"], suffixes=("_a", "_b"))
    early = k[k["target"] < te - pd.Timedelta(days=10)]
    np.testing.assert_allclose(early["lo80_a"], early["lo80_b"])


def test_by_oblast_daily_sums_to_national():
    from ua_alerts import by_oblast
    idx = pd.date_range("2026-01-01", periods=24 * 10, freq="h", tz="UTC")
    h = pd.DataFrame({"A": 1.0, "B": 0.5}, index=idx)
    d = by_oblast.oblast_daily(h, pd.Timestamp("2026-01-09"))
    assert list(d.columns) == ["A", "B"] and abs(d.iloc[1].sum() - 36.0) < 1e-9
