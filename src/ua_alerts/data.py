"""Download, clean and aggregate the public Ukrainian air raid alert dataset.

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset (MIT license)
We use the *volunteer* file (alerts collected from the eTryvoga channel) because, unlike
the "official" file, it is
  * updated up to the current day, and
  * recorded at a single, consistent granularity (oblast level) for its whole history,
    so there is no artificial structural break when official alerts moved to raion level.

All timestamps in the raw file are UTC. Day boundaries / hour-of-day are computed in
Europe/Kyiv local time because that is how people experience the daily pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources

LOCAL_TZ = "Europe/Kyiv"

# Luhansk oblast has had a "permanent" siren since 2022-04-04 that is not present in the
# data, so its series would be meaningless. Crimea is not in the file at all.
EXCLUDED_REGIONS = frozenset({"Luhanska oblast"})

# The official file contains "administrative" alerts for occupied territories that stay on for
# months. Alerts longer than this are dropped when loading the official source.
OFFICIAL_MAX_ALERT_HOURS = 72


@dataclass
class AlertData:
    alerts: pd.DataFrame          # cleaned, overlap-merged alert episodes (UTC)
    hourly_region: pd.DataFrame   # UTC hourly index x region, fraction of hour under alert
    hourly: pd.Series             # UTC hourly, number of oblasts under alert (hour-averaged)
    daily: pd.DataFrame           # local-day index; columns: alert_hours, starts
    as_of: pd.Timestamp           # latest timestamp present in the data (UTC)
    last_complete_day: pd.Timestamp  # last fully observed local day (naive date)
    cleaning: dict                # bookkeeping about what was dropped / imputed
    source: str = "volunteer"     # which dataset this was built from
    warnings: list = field(default_factory=list)  # data-quality warnings


def download(path: str | Path, refresh: bool = False, max_age_hours: float = 12, source: str = "volunteer"):
    """Backwards-compatible wrapper around :func:`ua_alerts.sources.fetch`."""
    res = sources.fetch(source, Path(path).parent, refresh=refresh, max_age_hours=max_age_hours)
    for n in res.notes:
        print("  note:", n)
    return res.path


def merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping / nested alert intervals within each region.

    Two overlapping intervals in the same oblast describe one continuous period under
    alert, so we collapse them. This guarantees hourly occupancy never exceeds 1.
    """
    a = df.sort_values(["region", "started_at"]).reset_index(drop=True)
    running_end = a.groupby("region")["finished_at"].cummax()
    prev_end = running_end.groupby(a["region"]).shift()
    new_episode = prev_end.isna() | (a["started_at"] > prev_end)
    a["episode"] = new_episode.cumsum()
    merged = a.groupby("episode").agg(
        region=("region", "first"),
        started_at=("started_at", "min"),
        finished_at=("finished_at", "max"),
        naive=("naive", "any"),
    )
    return merged.reset_index(drop=True)


def load_alerts(path: str | Path, kind: str = "volunteer") -> tuple[pd.DataFrame, dict]:
    """Read a source CSV into the common schema ``region, started_at, finished_at, naive``."""
    raw = pd.read_csv(path)
    info = {"rows_raw": len(raw), "kind": kind}
    if kind == "official":
        raw = raw.rename(columns={"oblast": "region"})
        raw["naive"] = False
    elif kind != "volunteer":
        raise ValueError(f"unknown source kind {kind!r}")
    raw = raw[["region", "started_at", "finished_at", "naive"]].copy()
    raw["started_at"] = pd.to_datetime(raw["started_at"], utc=True)
    raw["finished_at"] = pd.to_datetime(raw["finished_at"], utc=True)
    raw["naive"] = raw["naive"].astype(bool)

    df = raw[~raw["region"].isin(EXCLUDED_REGIONS)]
    info["rows_excluded_region"] = len(raw) - len(df)

    bad = df["finished_at"] <= df["started_at"]
    info["rows_nonpositive_duration"] = int(bad.sum())
    df = df[~bad]

    if kind == "official":
        long_ = (df["finished_at"] - df["started_at"]) > pd.Timedelta(hours=OFFICIAL_MAX_ALERT_HOURS)
        info["rows_overlong_dropped"] = int(long_.sum())
        df = df[~long_]

    info["rows_naive_30min_imputed"] = int(df["naive"].sum())
    merged = merge_overlaps(df)
    info["rows_after_merge"] = len(merged)
    return merged, info


def load_combined(official_path: str | Path, volunteer_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Official records up to the point where they end, then volunteer records after that.

    The cut-off ("seam") is the newest record in the official file. Volunteer alerts that finish
    before the seam are ignored; a volunteer alert that straddles the seam is clipped to start at it.

    WARNING: the two sources do not measure the same thing (the official file includes raion- and
    hromada-level alerts, the volunteer file is oblast-level sirens), so the combined series has a
    level shift at the seam for many oblasts. See :func:`seam_diagnostics`.
    """
    off, info_o = load_alerts(official_path, "official")
    vol, info_v = load_alerts(volunteer_path, "volunteer")
    seam = off["finished_at"].max()
    tail = vol[vol["finished_at"] > seam].copy()
    tail["started_at"] = tail["started_at"].clip(lower=seam)
    tail = tail[tail["finished_at"] > tail["started_at"]]
    alerts = merge_overlaps(pd.concat([off, tail], ignore_index=True))
    info = {
        "kind": "combined", "seam": seam, "rows_raw": info_o["rows_raw"] + info_v["rows_raw"],
        "official_rows": len(off), "volunteer_tail_rows": len(tail),
        "rows_naive_30min_imputed": int(tail["naive"].sum()), "rows_after_merge": len(alerts),
        "rows_excluded_region": info_o["rows_excluded_region"] + info_v["rows_excluded_region"],
        "rows_nonpositive_duration": info_o["rows_nonpositive_duration"] + info_v["rows_nonpositive_duration"],
        "rows_overlong_dropped": info_o.get("rows_overlong_dropped", 0),
    }
    return alerts, info


def seam_diagnostics(hourly_region: pd.DataFrame, seam: pd.Timestamp, as_of: pd.Timestamp,
                     days: int = 28, min_hours: int = 168) -> pd.DataFrame | None:
    """Average alert-hours per day just before vs just after the splice point, per oblast.

    Equal-length windows are compared (up to ``days`` days each). Returns None if less than
    ``min_hours`` of volunteer data follows the seam. A ratio far from 1 means the two sources
    disagree about the level for that oblast, i.e. the combined series has an artificial jump.
    """
    seam_h = seam.ceil("h")
    after = hourly_region.loc[seam_h:as_of.floor("h") - pd.Timedelta(hours=1)]
    n = min(days * 24, len(after))
    if n < min_hours:
        return None
    after = after.iloc[:n]
    before = hourly_region.loc[:seam_h - pd.Timedelta(hours=1)].iloc[-n:]
    out = pd.DataFrame({"before_h_per_day": before.sum() * 24 / n, "after_h_per_day": after.sum() * 24 / n})
    out["ratio"] = out["after_h_per_day"] / out["before_h_per_day"].where(out["before_h_per_day"] > 0)
    out["level_shift"] = (out["ratio"] > 2) | (out["ratio"] < 0.5)
    return out.sort_values("ratio")


def hourly_occupancy(alerts: pd.DataFrame, t0: pd.Timestamp, n_hours: int) -> pd.DataFrame:
    """Fraction of every UTC hour each region spent under alert (exact, minute-level)."""
    regions = sorted(alerts["region"].unique())
    out = np.zeros((n_hours, len(regions)))
    for j, region in enumerate(regions):
        sub = alerts[alerts["region"] == region]
        s = ((sub["started_at"] - t0).dt.total_seconds() / 3600).to_numpy()
        e = ((sub["finished_at"] - t0).dt.total_seconds() / 3600).to_numpy()
        partial = np.zeros(n_hours + 1)   # fractional first / last hours
        full = np.zeros(n_hours + 2)      # difference array for whole hours in between
        for si, ei in zip(s, e):
            i0, i1 = int(si), int(ei)
            if i0 == i1:
                partial[i0] += ei - si
            else:
                partial[i0] += (i0 + 1) - si
                partial[i1] += ei - i1
                full[i0 + 1] += 1
                full[i1] -= 1
        out[:, j] = partial[:n_hours] + np.cumsum(full)[:n_hours]
    idx = pd.date_range(t0, periods=n_hours, freq="h", tz="UTC")
    return pd.DataFrame(out, index=idx, columns=regions)


def build_dataset(csv_path: str | Path, kind: str = "volunteer", aux_path: str | Path | None = None) -> AlertData:
    """Build all series from one source. For ``kind="combined"`` pass the official CSV as
    ``csv_path`` and the volunteer CSV as ``aux_path``."""
    if kind == "combined":
        if aux_path is None:
            raise ValueError("kind='combined' needs aux_path (the volunteer CSV)")
        alerts, info = load_combined(csv_path, aux_path)
    else:
        alerts, info = load_alerts(csv_path, kind)

    # "as of" = newest thing we actually observed. Imputed (naive) ends are guesses, so
    # they are not used to decide how far the data reaches.
    observed_ends = alerts.loc[~alerts["naive"], "finished_at"].max()
    as_of = max(alerts["started_at"].max(), observed_ends)

    t0 = alerts["started_at"].min().floor("h")
    n_hours = int(np.ceil((alerts["finished_at"].max() - t0) / pd.Timedelta(hours=1))) + 2
    hourly_region = hourly_occupancy(alerts, t0, n_hours)
    hourly = hourly_region.sum(axis=1).rename("oblasts_under_alert")

    # --- daily series in local time -------------------------------------------------
    local = hourly.tz_convert(LOCAL_TZ)
    daily_hours = local.resample("D").sum()
    daily_hours.index = daily_hours.index.tz_localize(None)

    starts_local = alerts["started_at"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    starts = starts_local.value_counts().sort_index()

    first_full_day = (alerts["started_at"].min().tz_convert(LOCAL_TZ).tz_localize(None)
                      .normalize() + pd.Timedelta(days=1))
    # The day containing `as_of` is still in progress -> exclude it.
    last_complete_day = as_of.tz_convert(LOCAL_TZ).tz_localize(None).normalize() - pd.Timedelta(days=1)

    days = pd.date_range(first_full_day, last_complete_day, freq="D")
    daily = pd.DataFrame(
        {
            "alert_hours": daily_hours.reindex(days).astype(float),
            "starts": starts.reindex(days).fillna(0).astype(float),
        },
        index=days,
    )
    daily.index.name = "date"
    assert daily["alert_hours"].notna().all()
    # (official/combined alerts are unions of many small-area alerts, so long unions are normal there)
    warnings = sources.quality_report(alerts, as_of, check_long_alerts=(kind == "volunteer"))
    if kind == "combined":
        seam_tbl = seam_diagnostics(hourly_region, info["seam"], as_of)
        info["seam_table"] = seam_tbl
        if seam_tbl is None:
            warnings.append(f"Combined source: official data ends {info['seam']:%Y-%m-%d %H:%M} UTC; too little volunteer "
                            "data follows to compare levels across the seam.")
        else:
            n_bad = int(seam_tbl["level_shift"].sum())
            warnings.append(
                f"Combined source: official records end {info['seam']:%Y-%m-%d %H:%M} UTC and volunteer records take over. "
                f"{n_bad} of {len(seam_tbl)} oblasts change by more than 2x across that point because the two sources "
                "define alerts differently - treat results spanning the seam with caution (see USAGE.md).")
    return AlertData(alerts, hourly_region, hourly, daily, as_of, last_complete_day, info, kind, warnings)


def load_source(kind: str, cache_dir: str | Path, refresh: bool = False, max_age_hours: float = 12):
    """Fetch (with cache/mirror fallback) and build the dataset for ``kind``.

    Returns ``(AlertData, fetch_results)`` where ``fetch_results`` maps source name to
    :class:`sources.FetchResult` (so callers can print notes / report fallbacks).
    """
    need = ["official", "volunteer"] if kind == "combined" else [kind]
    res = {k: sources.fetch(k, cache_dir, refresh=refresh, max_age_hours=max_age_hours) for k in need}
    if kind == "combined":
        data = build_dataset(res["official"].path, "combined", aux_path=res["volunteer"].path)
    else:
        data = build_dataset(res[kind].path, kind)
    return data, res
