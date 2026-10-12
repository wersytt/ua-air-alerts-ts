#!/usr/bin/env python3
"""End-to-end pipeline: download -> clean -> EDA -> rolling backtest -> report.

    python run.py                  # latest data, 56-day test window + 2 earlier robustness windows
    python run.py --update         # re-fetch the dataset first (ignore the local cache), then run
    python run.py --source official   # analyse the official dataset instead
    python run.py --folds 1 --skip-sarima    # fast run (~40 s)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ua_alerts import data as D          # noqa: E402
from ua_alerts import by_oblast, changepoints, eda, evaluate as ev, models, report, sources, tables  # noqa: E402

warnings.filterwarnings("ignore")
HORIZON = 7  # days ahead; the GBM feature design assumes <= 7


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", "--refresh", dest="update", action="store_true",
                    help="re-fetch the dataset (ignoring the local cache) before running; --refresh is an alias")
    ap.add_argument("--source", choices=["volunteer", "official", "combined"], default="volunteer",
                    help="dataset to analyse: volunteer (default), official, or combined = official until it ends, then volunteer "
                         "(experimental: levels differ across the splice; see USAGE.md)")
    ap.add_argument("--test-days", type=int, default=56, help="length of each test window (days)")
    ap.add_argument("--folds", type=int, default=3, help="1 = only the latest window; >1 adds earlier windows")
    ap.add_argument("--by-oblast", action="store_true",
                    help="also backtest per-oblast daily models and a bottom-up national forecast (adds ~5 min)")
    ap.add_argument("--skip-sarima", action="store_true", help="skip the slowest model")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    figdir = args.outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    data, fetched = D.load_source(args.source, ROOT / "data", refresh=args.update)
    for name, res in fetched.items():
        for note in res.notes:
            print(f"note ({name}):", note)
        if args.update:
            if res.from_cache:
                print(f"warning: --update could not fetch a new {name} copy; continuing with the cached dataset "
                      f"({res.age_hours:.0f} h old).")
            else:
                print(f"Dataset '{name}' updated from {res.origin}")
    for msg in data.warnings:
        print("data warning:", msg)
    y = data.daily["alert_hours"]
    print(f"Data as of {data.as_of:%Y-%m-%d %H:%M} UTC | last complete day {data.last_complete_day:%Y-%m-%d} "
          f"| {len(y)} days")

    test_end = data.last_complete_day
    test_start = test_end - pd.Timedelta(days=args.test_days - 1)

    print("EDA ...")
    eda_res = eda.run_eda(data, test_start, figdir)

    folds = []
    for i in range(args.folds):
        te = test_end - pd.Timedelta(days=i * args.test_days)
        ts = te - pd.Timedelta(days=args.test_days - 1)
        print(f"Backtest fold {i}: {ts:%Y-%m-%d} .. {te:%Y-%m-%d}")
        preds = ev.backtest(y, lambda: models.default_models(include_slow=not args.skip_sarima),
                            ts, te, horizon=HORIZON)
        scale = ev.naive_scale(y, ts)
        per_h, summary = ev.score(preds, scale)
        folds.append(dict(test_start=ts, test_end=te, preds=preds, per_h=per_h, summary=summary,
                          scale=scale, compare=ev.compare_to_best_baseline(preds, summary)))

    main_fold = folds[0]
    figs = [ev.plot_forecasts(main_fold["preds"], y, h, main_fold["test_start"], main_fold["test_end"],
                              main_fold["summary"], figdir) for h in (1, HORIZON)]
    figs.append(ev.plot_mae_by_horizon(main_fold["per_h"], figdir))
    non_base = [m for m in main_fold["summary"].index if m not in ev.BASELINES]
    figs.append(ev.plot_error_dist(main_fold["preds"], non_base[0], figdir))

    make = lambda: models.default_models(include_slow=not args.skip_sarima)  # noqa: E731
    extra = {}

    print("Change points ...")
    seg = changepoints.segments(y.iloc[-1095:])
    extra["segments"] = seg
    extra["segments_fig"] = changepoints.plot_segments(y.iloc[-1095:], seg, figdir)

    best_name = main_fold["summary"].index[0]
    print(f"Prediction intervals for {best_name} ...")
    irows, isumm = ev.interval_backtest(y, make, best_name, main_fold["test_start"], main_fold["test_end"], horizon=HORIZON)
    extra.update(interval_model=best_name, interval_summary=isumm,
                 interval_fig=ev.plot_intervals(irows, best_name, main_fold["test_start"], main_fold["test_end"], figdir))
    irows.round(3).to_csv(args.outdir / "intervals_latest_window.csv", index=False)

    if args.by_oblast:
        print("Per-oblast models (slow) ...")
        daily_ob = by_oblast.oblast_daily(data.hourly_region, data.last_complete_day)
        per, nat_tbl, _ = by_oblast.run(daily_ob, y, main_fold["test_start"], main_fold["test_end"], HORIZON, verbose=print)
        extra.update(by_oblast=per, by_oblast_national=nat_tbl)
        per.round(3).to_csv(args.outdir / "by_oblast_latest_window.csv")

    main_fold["summary"].round(3).to_csv(args.outdir / "metrics_latest_window.csv")
    main_fold["preds"].to_csv(args.outdir / "predictions_latest_window.csv", index=False)
    y.rename("alert_hours").to_csv(args.outdir / "daily_series.csv")
    (args.outdir / "report.md").write_text(report.build_report(data, eda_res, folds, vars(args), extra), encoding="utf-8")

    print("\nLatest window metrics (oblast-hours/day):")
    print(tables.text(main_fold["summary"], index_name="model", signed=("bias_1to7",)))
    print(f"\nReport: {args.outdir / 'report.md'}")


if __name__ == "__main__":
    main()
