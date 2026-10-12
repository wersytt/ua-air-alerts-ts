import numpy as np
import pandas as pd
import pytest

from ua_alerts import data as D
from ua_alerts import forecast as F
from ua_alerts import pooled as PL
from test_forecast import _make_csv


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    p = tmp_path_factory.mktemp("pooled") / "v.csv"
    _make_csv(p, days=2 * 365 + 50)
    return D.build_dataset(p)


def test_neighbours_are_symmetric_and_use_dataset_names():
    regions = ["Kyiv City", "Kyivska oblast", "Lvivska oblast", "Volynska oblast", "Odeska oblast"]
    adj = PL.neighbours(regions)
    assert adj["Kyiv City"] == ["Kyivska oblast"]
    assert "Volynska oblast" in adj["Lvivska oblast"] and "Lvivska oblast" in adj["Volynska oblast"]
    assert all(a in adj[b] for a, ns in adj.items() for b in ns)


def test_platt_calibrator_is_monotone_and_fixes_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < 0.5 + 0.3 * (p - 0.5)).astype(float)       # truth is much flatter than p claims
    cal = PL.Platt().fit(p, y)
    q = cal.predict(np.linspace(0.05, 0.95, 50))
    assert np.all(np.diff(q) > 0) and q[0] > 0.3 and q[-1] < 0.7


def test_pooled_features_never_use_data_after_the_origin(tmp_path):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    cut = _make_csv(p1, days=450)
    _make_csv(p2, days=450, extra_after=cut - pd.Timedelta(days=20))
    A, B = D.build_dataset(p1), D.build_dataset(p2)
    WA, WB = PL.build_world(A), PL.build_world(B)
    t = np.arange(800, int((cut - pd.Timedelta(days=21) - WA.P["Kyiv City"].idx[0]) / pd.Timedelta(hours=1)), 11)
    for r in WA.regions:
        XA, _ = PL.matrix(WA, r, t, range(1, 25))
        XB, _ = PL.matrix(WB, r, t, range(1, 25))
        np.testing.assert_allclose(XA, XB, equal_nan=True)


def test_pooled_backtest_covers_every_oblast_and_is_sane(ds):
    res = PL.backtest_all(ds, test_days=14, train_days=300)
    assert set(res) == set(ds.hourly_region.columns)
    b = res["Kharkivska oblast"]
    assert b.n_predictions == 14 * 24 * 24
    assert b.hourly.loc[F.MODEL_ROW, "Brier"] <= b.hourly.loc["base rate (training mean)", "Brier"] * 1.05
    assert 0.0 <= b.blend <= 1.0
    # replaying one day earlier must not change anything about a window that ends earlier (as_of is honoured)
    res2 = PL.backtest_all(ds, test_days=14, train_days=300, as_of=ds.as_of - pd.Timedelta(days=3))
    assert res2["Kharkivska oblast"].test_end < b.test_end
