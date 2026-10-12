#!/usr/bin/env python3
"""Cross-check the alert datasets against each other and against published official figures.

    python validate.py                       # volunteer vs official CSV
    python validate.py --months 6            # shorter comparison window

Writes outputs/validation_report.md and outputs/validation_by_oblast.csv. Exit code is always 0:
disagreement is a finding, not a crash.

What is compared
----------------
* **volunteer** (eTryvoga, oblast-level) vs **official** CSV (any alert in an oblast, incl. raion /
  hromada-level alerts, with >72 h "administrative" alerts removed). They define "alert" differently,
  so the ratio is expected to differ from 1 for structural reasons - what matters is whether it is
  *stable* over time and whether the day-to-day pattern still correlates.
* **Kyiv City**: volunteer vs official CSV, month by month.
* **Reference points** quoted in the press from official sources (REFERENCE_POINTS below).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ua_alerts import data as D, sources, tables  # noqa: E402

warnings.filterwarnings("ignore")

# Figures published by official bodies / reported by media quoting them. Extend freely.
REFERENCE_POINTS = [
    dict(region="Kyiv City", month="2026-09", hours=242 + 40 / 60,
         what="Kyiv total alert time, Sept 2026 ('Digital Kyiv' official history, quoted by Censor.NET)",
         url="https://censor.net/en/news/4026106/duration-of-air-raid-alerts-in-kyiv-statistics-for-september-2026"),
]


def local_hourly(alerts: pd.DataFrame, t0: pd.Timestamp, n: int) -> pd.DataFrame:
    occ = D.hourly_occupancy(alerts, t0, n)
    occ.index = occ.index.tz_convert(D.LOCAL_TZ).tz_localize(None)
    return occ


def first_incomplete_month(as_of: pd.Timestamp) -> pd.Timestamp:
    """First day of the month containing ``as_of`` - that month is still in progress."""
    return as_of.tz_convert(D.LOCAL_TZ).tz_localize(None).to_period("M").to_timestamp()


def monthly(occ: pd.DataFrame, region: str, upto: pd.Timestamp) -> pd.Series:
    s = occ[region].resample("MS").sum()
    return s[s.index < upto]


def build(kind: str, refresh: bool):
    res = sources.fetch(kind, ROOT / "data", refresh=refresh, max_age_hours=12)
    alerts, info = D.load_alerts(res.path, kind)
    as_of = max(alerts["started_at"].max(), alerts.loc[~alerts["naive"], "finished_at"].max())
    return res, alerts, info, as_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", type=int, default=12, help="comparison window in complete months (default 12)")
    ap.add_argument("--update", "--refresh", dest="refresh", action="store_true",
                    help="re-fetch the datasets (ignoring the local cache); --refresh is an alias")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    w = out.append
    w("# Data validation report\n")
    w(f"*Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC.*\n")

    # ---------------------------------------------------------------- 1. source health
    w("## 1. Source health\n")
    health, built = [], {}
    for kind in ("volunteer", "official"):
        try:
            res, alerts, info, as_of = build(kind, args.refresh)
            built[kind] = (alerts, as_of)
            health.append(dict(source=kind, status="ok" if not res.from_cache else "cache",
                               origin=("cache" if res.from_cache else res.origin.split("/")[2]),
                               age_h=res.age_hours, rows=info["rows_after_merge"],
                               first=f"{alerts['started_at'].min():%Y-%m-%d}", latest=f"{as_of:%Y-%m-%d %H:%M}"))
            for n in res.notes:
                w(f"- `{kind}`: {n}")
        except Exception as exc:
            health.append(dict(source=kind, status="FAILED", origin=str(exc)[:60].replace("\n", " "),
                               age_h=np.nan, rows=0, first="-", latest="-"))
    w(tables.md(pd.DataFrame(health).set_index("source"), index_name="source", decimals={"age_h": 1}) + "\n")
    if "volunteer" not in built or "official" not in built:
        w("Cannot compare the two main datasets (one failed to load). See notes above.\n")
        (args.outdir / "validation_report.md").write_text("\n".join(out), encoding="utf-8")
        print("\n".join(out))
        return

    (va, v_asof), (oa, o_asof) = built["volunteer"], built["official"]
    t0 = min(va["started_at"].min(), oa["started_at"].min()).floor("h")
    n = int((max(va["finished_at"].max(), oa["finished_at"].max()) - t0) / pd.Timedelta(hours=1)) + 3
    V, O = local_hourly(va, t0, n), local_hourly(oa, t0, n)
    upto_v, upto_o = first_incomplete_month(v_asof), first_incomplete_month(o_asof)
    common_upto = min(upto_v, upto_o)
    start = common_upto - pd.DateOffset(months=args.months)

    # --------------------------------------------------- 2. volunteer vs official, per oblast
    w(f"## 2. Volunteer vs official CSV, per oblast ({start:%Y-%m} to {(common_upto - pd.Timedelta(days=1)):%Y-%m})\n")
    w("`ratio` = volunteer alert-hours / official alert-hours. The two sources define an alert differently "
      "(the official one includes raion/hromada-level alerts), so ratios far from 1 are *expected* for border "
      "oblasts; what signals trouble is a ratio that **changes over time** (`ratio_first_half` vs `ratio_second_half`) "
      "or a low daily correlation.\n")
    rows = []
    for region in V.columns.intersection(O.columns):
        v = V.loc[start:common_upto - pd.Timedelta(hours=1), region]
        o = O.loc[start:common_upto - pd.Timedelta(hours=1), region]
        vd, od = v.resample("D").sum(), o.resample("D").sum()
        half = len(vd) // 2
        r1 = vd.iloc[:half].sum() / max(od.iloc[:half].sum(), 1e-9)
        r2 = vd.iloc[half:].sum() / max(od.iloc[half:].sum(), 1e-9)
        corr = float(np.corrcoef(vd, od)[0, 1]) if vd.std() > 0 and od.std() > 0 else np.nan
        ratio = vd.sum() / max(od.sum(), 1e-9)
        drift = max(r1, r2) / max(min(r1, r2), 1e-9)
        verdict = "consistent" if (0.5 <= ratio <= 2 and corr >= 0.5 and drift < 1.5) else \
                  "DIVERGES" if (drift >= 3 or corr < 0.2 or ratio < 0.25 or ratio > 4) else "differs"
        rows.append(dict(oblast=region, volunteer_h=vd.sum(), official_h=od.sum(), ratio=ratio,
                         ratio_first_half=r1, ratio_second_half=r2, daily_corr=corr, verdict=verdict))
    cmp_ = pd.DataFrame(rows).set_index("oblast").sort_values("ratio")
    w(tables.md(cmp_, index_name="oblast", decimals={"volunteer_h": 0, "official_h": 0, "daily_corr": 2}) + "\n")
    counts = cmp_["verdict"].value_counts()
    w(f"Verdicts: {', '.join(f'{k} = {v}' for k, v in counts.items())} (of {len(cmp_)} oblasts). "
      "`DIVERGES` = ratio drifts >=3x between the two halves of the window, or daily correlation < 0.2, or the "
      "sources differ by more than 4x. `differs` = in between.\n")
    cmp_.round(3).to_csv(args.outdir / "validation_by_oblast.csv")

    # ------------------------------------------------------------------- 3. Kyiv City
    w("## 3. Kyiv City, month by month (alert-hours)\n")
    cols = {"volunteer": monthly(V, "Kyiv City", upto_v), "official_csv": monthly(O, "Kyiv City", upto_o)}
    kt = pd.DataFrame(cols)
    kt = kt[(kt.index >= start) & (kt.index < upto_v)]
    kt.index = kt.index.strftime("%Y-%m")
    w(tables.md(kt, index_name="month", decimals={"volunteer": 1, "official_csv": 1}) + "\n")
    w("`-` = that source does not fully cover the month.\n")

    # ---------------------------------------------------------------- 4. reference points
    w("## 4. Published official reference points\n")
    ref_rows = []
    for ref in REFERENCE_POINTS:
        m = pd.Timestamp(ref["month"] + "-01")
        vv = monthly(V, ref["region"], upto_v).get(m, np.nan)
        ref_rows.append(dict(reference=ref["what"], official_h=ref["hours"], volunteer_h=vv, volunteer_over_official=vv / ref["hours"]))
        w(f"- {ref['what']}: <{ref['url']}>")
    w("")
    w(tables.md(pd.DataFrame(ref_rows).set_index("reference"), index_name="reference",
                decimals={"official_h": 1, "volunteer_h": 1, "volunteer_over_official": 2}) + "\n")
    w("Note: since 6 Sept 2026 Kyiv uses tiered threat levels (yellow 'drone threat', red, 'all clear'), and the "
      "figure above counts all of them. The datasets here record siren-style alerts, which is a plausible "
      "explanation for the gap, but this tool cannot prove it.\n")

    # ------------------------------------------------------------------ 5. what to do
    w("## 5. How to interpret this\n")
    w("- If most oblasts are `consistent`, the volunteer file is a reasonable proxy for *siren-level* alerts.")
    w("- `DIVERGES` oblasts: the two sources are measuring different things or one changed its method. "
      "Prefer the source whose definition matches your question, and avoid mixing periods before/after a change.")
    w("- Re-run the analysis on the other source with `python run.py --source official` and compare conclusions: "
      "findings that hold on both are far more trustworthy. (The official file lags by weeks, so it cannot drive "
      "`predict.py`.)")
    text = "\n".join(out) + "\n"
    (args.outdir / "validation_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {args.outdir / 'validation_report.md'}")


if __name__ == "__main__":
    main()
