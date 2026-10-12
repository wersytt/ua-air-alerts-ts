#!/usr/bin/env python3
"""Probabilistic air raid alert outlook for one oblast, for the next 24 hours.

    python predict.py --list                          # valid oblast names
    python predict.py --oblast "Kyiv City"            # next 24 h from now
    python predict.py --oblast Харківська             # Ukrainian / partial names work too
    python predict.py --oblast Lviv --backtest        # how good is this model for this oblast?
    python predict.py --oblast Lviv --model per-oblast   # the original, faster, less accurate model
    python predict.py --oblast all --backtest         # ... for every oblast
    python predict.py --oblast Odeska --json          # machine-readable output
    python predict.py --oblast "Kyiv City" --now "2026-10-11 18:00"   # replay: pretend it is 18:00 Kyiv time

This is a statistical estimate from historical patterns - NOT a warning system.
Follow official alert channels for safety.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ua_alerts import data as D                      # noqa: E402
from ua_alerts import forecast as F, forecast_plots, pooled as PL, sources, tables  # noqa: E402

warnings.filterwarnings("ignore")


def _bar(p: float, width: int = 28) -> str:
    ch = "█" if (sys.stdout.encoding or "").lower().startswith("utf") else "#"
    return ch * int(round(p * width))


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _show_forecast(r: F.ForecastResult) -> None:
    loc = lambda t: t.tz_convert("Europe/Kyiv")  # noqa: E731
    s = r.slots
    print(f"\nAir raid alert outlook: {r.oblast}")
    print(f"  generated   {loc(r.generated_at):%Y-%m-%d %H:%M} Kyiv time")
    print(f"  data as of  {loc(r.data_as_of):%Y-%m-%d %H:%M} Kyiv time ({r.data_age_hours:.1f} h old) | source: {r.source}")
    print(f"  window      {s['start_local'].iloc[0]:%a %d %b %H:00} -> {s['end_local'].iloc[-1]:%a %d %b %H:00} "
          "(24 hourly slots, Kyiv time; the first slot is the hour you are in)\n")

    view = pd.DataFrame({
        "P(alert)": s["probability"].to_numpy(),
        "chart": [_bar(p) for p in s["probability"]],
    }, index=[f"{a:%a %d %H:%M}-{b:%H:%M}" for a, b in zip(s["start_local"], s["end_local"])])
    print(tables.text(view, index_name="slot (Kyiv)", decimals=0, pct=("P(alert)",)))

    st = r.recent_state
    top = s.sort_values("probability", ascending=False).head(3)
    print("\nSummary")
    print(f"  expected hour-slots touched by an alert    : {r.expected_alert_hours:.1f} of 24"
          f"   (recent norm: {st['typical_alert_slots_per_24h_last_28d']} of 24; the alerts themselves"
          f" lasted {st['typical_alert_duration_h_per_24h_last_28d']} h/day in total)")
    trivial = st["days_with_alert_last_28d"] >= 27
    print(f"  P(at least one alert in the window)        : {'>99%' if r.p_any > 0.99 else f'{r.p_any:.0%}'}"
          + (f"   (uninformative: {st['days_with_alert_last_28d']}/28 recent days had an alert)" if trivial else ""))
    print("  likeliest slots                            : "
          + ", ".join(f"{a:%H:%M} ({p:.0%})" for a, p in zip(top['start_local'], top['probability'])))
    print(f"  model                                      : {r.model}")
    print(f"  model weight vs climatology                : {r.blend:.1f}  (chosen on validation data; 0 = no skill shown)")
    print(f"\nRecent history (latest data point: {loc(r.data_as_of):%a %d %b %H:%M} Kyiv time)")
    if st["active_in_last_observed_hour"]:
        now_txt = "an alert was in progress in the latest hour of data"
    elif st["hours_since_last_alert"] is not None:
        now_txt = f"no alert in the latest hour of data; the last alert hour was {st['hours_since_last_alert']} h earlier"
    else:
        now_txt = "no alert in the latest hour of data or in the week before it"
    print(f"  this oblast         : {now_txt}")
    print(f"  time under alert    : {st['alert_hours_last_24h']} h in the 24 h before that, "
          f"{st['alert_hours_last_7d']} h in the previous 7 days")
    print(f"  nationwide          : {st['oblasts_under_alert_nationally_last_hour']} of {st['oblasts_in_data']} oblasts "
          "under alert in the latest hour of data (this oblast included)")
    for w in r.warnings:
        if "longer than 72 h" not in w:          # dataset-wide detail; shown by validate.py instead
            print(f"  ! {w}")
    print("\nHow to read this: hour-by-hour alert timing is hard to predict; the model is typically only a few percent better")
    print("than 'same hour over the last 4 weeks'. Run with --backtest to see its measured accuracy for this oblast.")
    print("Statistical estimate only - not a warning system. Follow official alerts.")


def _show_backtest(b: F.BacktestResult) -> str:
    out = []
    w = out.append
    w(f"Backtest: {b.oblast}")
    w(f"  test window {b.test_start:%Y-%m-%d %H:%M} -> {b.test_end:%Y-%m-%d %H:%M} UTC | {b.n_predictions:,} hourly predictions "
      f"| trained on the preceding {b.train_days} days | share of hours with an alert: {b.base_rate:.1%}")
    w(f"  model weight vs climatology chosen on a validation window before the test: {b.blend:.1f} "
      f"(1.0 = trust model fully, 0.0 = model had no skill)\n")
    w("Hourly task: P(alert in hour t+h), h = 1..24, every hour of the test window used as a forecast origin")
    w(tables.text(b.hourly, index_name="predictor", decimals=3, signed=("skill_vs_clim",)))
    w("\nBy horizon (Brier score, lower is better; skill > 0 means better than climatology)")
    w(tables.text(b.by_horizon, index_name="horizon (h)", decimals=3, signed=("skill_vs_clim",)))
    w("\nCalibration: when the model says X, how often did an alert really happen?")
    w(tables.text(b.reliability, index_name="predicted bin", decimals=3))
    w("\nWindow task: any alert at all during the next 24 h")
    w(tables.text(b.window, index_name="predictor", decimals=3))
    if b.window["observed_rate"].iloc[0] > 0.97:
        w("  (nearly every 24 h window contained an alert, so this task is close to trivial for this oblast)")
    return "\n".join(out)


def _backtest_one_or_all(data, args, regions):
    """{region: BacktestResult} for the requested regions with the chosen model."""
    if args.model == "pooled":
        print("Fitting the pooled model on all oblasts (about a minute) ...", file=sys.stderr)
        res = PL.backtest_all(data, test_days=args.test_days, train_days=args.train_days)
        return {r: res[r] for r in regions}
    out = {}
    for reg in regions:
        out[reg] = F.backtest(data, reg, test_days=args.test_days, train_days=args.train_days)
        print(f"  done {reg}", file=sys.stderr)
    return out


def _backtest_all(data, regions, args) -> None:
    rows = []
    for reg, b in _backtest_one_or_all(data, args, regions).items():
        h = b.hourly.loc[F.MODEL_ROW]
        rows.append({"oblast": reg, "hours_with_alert": b.base_rate, "Brier_model": h["Brier"],
                     "Brier_climatology": b.hourly.loc["climatology (28d same hour)", "Brier"],
                     "skill_vs_clim": h["skill_vs_clim"], "model_weight": b.blend, "AUC": h["AUC"]})
    tbl = pd.DataFrame(rows).set_index("oblast").sort_values("skill_vs_clim", ascending=False)
    text = (f"Backtest on the latest {args.test_days} days for all oblasts "
            f"(hourly task, {args.model} model, trained on the preceding {args.train_days} days)\n"
            + tables.text(tbl, index_name="oblast", decimals=3, pct=("hours_with_alert",), signed=("skill_vs_clim",)))
    print(text)
    if not args.no_plot:
        (args.outdir / "backtest_all_oblasts.txt").write_text(text + "\n", encoding="utf-8")
        tbl.round(4).to_csv(args.outdir / "backtest_all_oblasts.csv")
        print(f"\nSaved {args.outdir / 'backtest_all_oblasts.txt'} and .csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oblast", help='oblast name, e.g. "Kyiv City", Kharkiv, Харківська (see --list); "all" with --backtest')
    ap.add_argument("--list", action="store_true", help="list valid oblast names and exit")
    ap.add_argument("--backtest", action="store_true", help="evaluate the model on the latest --test-days instead of forecasting")
    ap.add_argument("--test-days", type=int, default=28, help="backtest window length (default 28)")
    ap.add_argument("--train-days", type=int, default=None,
                    help="training history used for the model (default 365)")
    ap.add_argument("--model", choices=["pooled", "per-oblast"], default="pooled",
                    help="pooled (default): one model for all oblasts, with neighbour features, ~1 min per run; "
                         "per-oblast: the original single-oblast model, ~5 s")
    ap.add_argument("--now", help="override the current time (UTC if it has an offset, else Kyiv local), e.g. '2026-10-11 18:00'")
    ap.add_argument("--source", choices=["volunteer", "official", "combined"], default="volunteer",
                    help="dataset: volunteer (default), official, or combined = official until it ends, then volunteer (experimental)")
    ap.add_argument("--update", "--refresh", dest="refresh", action="store_true",
                    help="re-fetch the dataset (ignoring the local cache); --refresh is an alias")
    ap.add_argument("--json", action="store_true", help="print the forecast as JSON only")
    ap.add_argument("--no-plot", action="store_true", help="do not write PNG/CSV files")
    ap.add_argument("--outdir", type=Path, default=ROOT / "outputs", help="where to write files")
    args = ap.parse_args()
    if args.train_days is None:
        args.train_days = 365

    data, fetched = D.load_source(args.source, ROOT / "data", refresh=args.refresh, max_age_hours=3)
    if not args.json:
        for name, res in fetched.items():
            for n in res.notes:
                print(f"note ({name}):", n, file=sys.stderr)
    regions = list(data.hourly_region.columns)

    if args.list:
        print("Valid oblast names (English or Ukrainian, partial matches OK):")
        print("\n".join(f"  {r}" for r in regions))
        return
    if not args.oblast:
        ap.error("--oblast is required (use --list to see valid names)")

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.backtest and args.oblast.strip().lower() == "all":
        _backtest_all(data, regions, args)
        return

    try:
        region = F.resolve_oblast(args.oblast, regions)
    except ValueError as exc:
        ap.error(str(exc))

    if args.backtest:
        bt = _backtest_one_or_all(data, args, [region])[region]
        text = _show_backtest(bt)
        print(text)
        if not args.no_plot:
            png = forecast_plots.plot_backtest(bt, args.outdir / f"backtest_{_slug(region)}.png")
            (args.outdir / f"backtest_{_slug(region)}.txt").write_text(text + "\n", encoding="utf-8")
            print(f"\nSaved {png} and the table as .txt")
        return

    now = pd.Timestamp(args.now) if args.now else None
    try:
        if args.model == "pooled" and not args.json:
            print("Fitting the pooled model on all oblasts (about a minute) ...", file=sys.stderr)
        r = F.predict_next_24h(data, region, now=now, train_days=args.train_days, model=args.model)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        return
    _show_forecast(r)
    if not args.no_plot:
        stamp = r.generated_at.tz_convert("Europe/Kyiv").strftime("%Y%m%d_%H%M")
        base = args.outdir / f"forecast_{_slug(region)}_{stamp}"
        r.slots.assign(start_local=r.slots.start_local.dt.strftime("%Y-%m-%d %H:%M"),
                       end_local=r.slots.end_local.dt.strftime("%Y-%m-%d %H:%M")
                       ).round(4).to_csv(base.with_suffix(".csv"), index=False)
        forecast_plots.plot_forecast(r, base.with_suffix(".png"))
        print(f"\nSaved {base.with_suffix('.csv')} and {base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
