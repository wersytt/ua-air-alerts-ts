import numpy as np
import pandas as pd
import pytest

from ua_alerts import data as D
from ua_alerts import forecast as F


def _make_csv(path, days=120, seed=0, extra_after=None):
    """Synthetic volunteer-format file: 3 oblasts, evening-biased alerts, optional 'future' alerts."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for d in range(days):
        for region, rate in (("Kyiv City", 0.5), ("Kharkivska oblast", 3.0), ("Lvivska oblast", 0.3)):
            for _ in range(rng.poisson(rate)):
                hour = int(np.clip(rng.normal(20, 4), 0, 23))
                s = start + pd.Timedelta(days=d, hours=hour, minutes=int(rng.integers(0, 60)))
                rows.append((region, s, s + pd.Timedelta(minutes=int(rng.integers(10, 90))), False))
    if extra_after is not None:
        for k in range(300):
            s = extra_after + pd.Timedelta(hours=k)
            rows.append(("Kyiv City", s, s + pd.Timedelta(minutes=45), False))
    df = pd.DataFrame(rows, columns=["region", "started_at", "finished_at", "naive"]).sort_values("started_at")
    df.to_csv(path, index=False)
    return start + pd.Timedelta(days=days)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "v.csv"
    end = _make_csv(p, days=2 * 365 + 50)
    return D.build_dataset(p), end


def test_resolve_oblast_handles_english_ukrainian_and_partial():
    av = ["Kyiv City", "Kyivska oblast", "Kharkivska oblast", "Lvivska oblast", "Ivano-Frankivska oblast", "Odeska oblast"]
    assert F.resolve_oblast("kyiv", av) == "Kyiv City"
    assert F.resolve_oblast("Київ", av) == "Kyiv City"
    assert F.resolve_oblast("Київська область", av) == "Kyivska oblast"
    assert F.resolve_oblast("Kharkiv", av) == "Kharkivska oblast"
    assert F.resolve_oblast("Харківська", av) == "Kharkivska oblast"
    assert F.resolve_oblast("ivano", av) == "Ivano-Frankivska oblast"
    with pytest.raises(ValueError, match="--list"):
        F.resolve_oblast("Atlantis", av)


@pytest.mark.parametrize("model", ["pooled", "per-oblast"])
def test_forecast_shape_range_and_staleness_handling(ds, model):
    data, end = ds
    now = data.as_of + pd.Timedelta(hours=6)           # data is ~6 h stale
    r = F.predict_next_24h(data, "Kharkivska oblast", now=now, train_days=300, model=model)
    assert r.model == model
    assert len(r.slots) == 24 and r.slots["probability"].between(0, 1).all()
    assert r.slots["start_local"].is_monotonic_increasing
    assert 0 <= r.p_any <= 1 and r.p_any >= r.slots["probability"].max() - 1e-9
    assert abs(r.expected_alert_hours - r.slots["probability"].sum()) < 1e-9
    assert r.slots["start_local"].iloc[0] == now.floor("h").tz_convert("Europe/Kyiv")
    assert r.to_dict()["oblast"] == "Kharkivska oblast"


def test_refuses_to_forecast_from_ancient_data(ds):
    data, _ = ds
    with pytest.raises(RuntimeError, match="old"):
        F.predict_next_24h(data, "Kyiv City", now=data.as_of + pd.Timedelta(days=10))


def test_model_learns_the_evening_pattern(ds):
    """On synthetic data alerts cluster around 20:00 local; the forecast should reflect that."""
    data, _ = ds
    r = F.predict_next_24h(data, "Kharkivska oblast", now=data.as_of + pd.Timedelta(hours=2), train_days=365)
    hours = r.slots["start_local"].dt.hour.to_numpy()
    p = r.slots["probability"].to_numpy()
    assert p[(hours >= 18) & (hours <= 23)].mean() > p[(hours >= 2) & (hours <= 8)].mean()


def test_features_never_use_data_after_the_origin(tmp_path):
    """Build the same history with and without lots of later alerts: every feature computed at an
    origin that both datasets cover must be identical (no peeking at the future)."""
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    cut = _make_csv(p1, days=450)
    _make_csv(p2, days=450, extra_after=cut - pd.Timedelta(days=20))
    A, B = D.build_dataset(p1), D.build_dataset(p2)
    PA, PB = F.prepare(A, "Kyiv City"), F.prepare(B, "Kyiv City")
    origin_cut = cut - pd.Timedelta(days=21)
    t = np.arange(800, int((origin_cut - PA.idx[0]) / pd.Timedelta(hours=1)), 7)
    np.testing.assert_allclose(F.origin_features(PA, t), F.origin_features(PB, t), equal_nan=True)
    XA, _ = F.hourly_matrix(PA, t, range(1, 25))
    XB, _ = F.hourly_matrix(PB, t, range(1, 25))
    np.testing.assert_allclose(XA, XB, equal_nan=True)
    t_late = np.array([len(PA.a) - 5])      # sanity: the test has teeth
    assert not np.allclose(F.origin_features(PA, t_late), F.origin_features(PB, t_late))


def test_backtest_runs_and_is_not_worse_than_base_rate(ds):
    data, _ = ds
    b = F.backtest(data, "Kharkivska oblast", test_days=14, train_days=365)
    assert b.n_predictions == 14 * 24 * 24
    assert b.hourly.loc[F.MODEL_ROW, "Brier"] <= b.hourly.loc["base rate (training mean)", "Brier"] * 1.05
    big = b.reliability[b.reliability["n"] > 500]
    assert (big["mean_predicted"] - big["observed_rate"]).abs().max() < 0.12


def test_replay_in_the_past_uses_only_data_known_then(ds):
    data, _ = ds
    now = data.as_of - pd.Timedelta(days=30)
    r = F.predict_next_24h(data, "Kharkivska oblast", now=now, train_days=300)
    assert r.slots["start_local"].iloc[0] == now.floor("h").tz_convert("Europe/Kyiv")
    assert r.data_as_of <= now and any("REPLAY" in w for w in r.warnings)
