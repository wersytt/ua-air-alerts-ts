import numpy as np
import pandas as pd

from ua_alerts.data import hourly_occupancy, merge_overlaps


def _alerts(rows):
    df = pd.DataFrame(rows, columns=["region", "started_at", "finished_at"])
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    df["naive"] = False
    return df


def test_merge_overlaps_collapses_nested_and_overlapping_but_not_across_regions():
    df = _alerts([
        ("A", "2024-01-01 10:00", "2024-01-01 12:00"),
        ("A", "2024-01-01 11:00", "2024-01-01 11:30"),   # nested
        ("A", "2024-01-01 11:50", "2024-01-01 13:00"),   # overlapping
        ("A", "2024-01-01 15:00", "2024-01-01 16:00"),   # separate
        ("B", "2024-01-01 10:30", "2024-01-01 11:00"),   # other region, must stay separate
    ])
    m = merge_overlaps(df).sort_values(["region", "started_at"]).reset_index(drop=True)
    assert len(m) == 3
    assert m.loc[0, "finished_at"] == pd.Timestamp("2024-01-01 13:00", tz="UTC")
    assert m.loc[2, "region"] == "B"


def test_hourly_occupancy_is_exact_and_conserves_total_time():
    df = _alerts([
        ("A", "2024-01-01 10:30", "2024-01-01 13:15"),   # 2h45m across 4 hourly bins
        ("B", "2024-01-01 11:00", "2024-01-01 11:20"),
    ])
    t0 = pd.Timestamp("2024-01-01 09:00", tz="UTC")
    occ = hourly_occupancy(df, t0, 8)
    assert np.isclose(occ.loc["2024-01-01 10:00+00:00", "A"], 0.5)
    assert np.isclose(occ.loc["2024-01-01 11:00+00:00", "A"], 1.0)
    assert np.isclose(occ.loc["2024-01-01 13:00+00:00", "A"], 0.25)
    assert np.isclose(occ.loc["2024-01-01 11:00+00:00", "B"], 20 / 60)
    assert np.isclose(occ.values.sum(), 2.75 + 20 / 60)
    assert occ.values.max() <= 1.0 + 1e-9
