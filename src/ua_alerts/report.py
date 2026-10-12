"""Assemble a self-contained Markdown report from the pipeline results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import tables
from .evaluate import BASELINES


def build_report(data, eda, folds, cfg, extra=None) -> str:
    """folds: list of dicts with keys test_start, test_end, summary, per_h, compare."""
    extra = extra or {}
    main = folds[0]
    s, cmp_ = main["summary"], main["compare"]
    y = data.daily["alert_hours"]
    ts, te = main["test_start"], main["test_end"]

    best = s.index[0]
    best_base = s.loc[s.index.isin(BASELINES)].index[0]
    sig_better = cmp_[(cmp_["significant"]) & (cmp_["dMAE"] < 0)].index.tolist()
    test_mean = y.loc[ts:te].mean()
    prev_mean = y.loc[: ts - pd.Timedelta(days=1)].iloc[-365:].mean()

    out = []
    w = out.append
    w("# Air raid alerts in Ukraine: time series analysis\n")
    w(f"*Generated automatically. Data as of **{data.as_of:%Y-%m-%d %H:%M} UTC**; "
      f"last complete day used: **{data.last_complete_day:%Y-%m-%d}** (the in-progress day is excluded).*\n")

    w("## 1. Data\n")
    c = data.cleaning
    kind_txt = {
        "volunteer": "volunteer-collected alerts (eTryvoga channel), oblast level",
        "official": "official alert records (oblast/raion/hromada level, >72 h administrative alerts removed)",
        "combined": ("official records up to " + (f"{c['seam']:%Y-%m-%d %H:%M} UTC" if "seam" in c else "their end")
                     + ", then volunteer records after that point (experimental splice)"),
    }[data.source]
    w(f"Source (`{data.source}`): {kind_txt}, from the open "
      "[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset); "
      "timestamps UTC, converted to Europe/Kyiv for daily/hourly aggregation.\n")
    w(f"- Raw rows: {c['rows_raw']:,}; excluded Luhansk (permanent siren, not in data): {c['rows_excluded_region']}; "
      f"non-positive durations dropped: {c['rows_nonpositive_duration']}; "
      f"alerts with an *imputed* 30-minute end: {c['rows_naive_30min_imputed']:,} "
      f"({c['rows_naive_30min_imputed'] / c['rows_raw']:.1%}); episodes after merging overlaps: {c['rows_after_merge']:,}.")
    if data.source == "combined" and c.get("seam_table") is not None:
        st = c["seam_table"].copy()
        st["level_shift"] = st["level_shift"].map({True: "yes", False: "no"})
        w("- **Splice diagnostics.** Average alert-hours per day in equal windows just before and just after the splice "
          f"point ({c['seam']:%Y-%m-%d}); a ratio far from 1 is an artificial jump caused by the two sources defining "
          "alerts differently, not a change in the real situation:\n")
        w(tables.md(st, index_name="oblast", decimals=2) + "\n")
    w(f"- Series: {len(y):,} complete days, {y.index[0]:%Y-%m-%d} to {y.index[-1]:%Y-%m-%d}.")
    w("- **Target** `alert_hours`: summed over 24 oblasts, the number of hours spent under air raid alert "
      "that local day (max 576). It is duration-based on purpose: the *count* of alerts is distorted "
      "by changes in how alerts are issued/recorded (see yearly table).\n")
    w(tables.md(eda["yearly"].astype({"days": int}), index_name="year") + "\n")

    w("## 2. What the exploration shows\n")
    st = eda["stl"]
    w(f"- STL (period 7): trend strength **{st['trend_strength']:.2f}**, weekly-seasonal strength "
      f"**{st['seasonal_strength']:.2f}** (0 = none, 1 = fully deterministic). Weekday means over the last 26 weeks: "
      + ", ".join(f"{k} {v}" for k, v in eda["weekday_profile"].items()) + ".")
    h = eda["stationarity_levels_365d"]
    hd = eda["stationarity_diff_365d"]
    w(f"- Stationarity, last 365 days (levels): ADF p={h['adf_p']:.3g}, KPSS p={h['kpss_p']:.3f} "
      f"(KPSS p<0.05 rejects a constant level, i.e. the level drifts); first differences: "
      f"ADF p={hd['adf_p']:.3g}, KPSS p>={hd['kpss_p']:.2f}.")
    w(f"- Intraday (last 180 days, Kyiv time): most oblasts are under alert around **{eda['hour_peak']:02d}:00**, "
      f"fewest around **{eda['hour_trough']:02d}:00**.")
    w("- Most alert time per day over the last 90 days: "
      + ", ".join(f"{k.replace(' oblast', '')} ({v:.1f} h)" for k, v in eda["regions_top"].items()) + ".\n")
    for f in eda["figures"]:
        w(f"![{f}](figures/{f})")
    w("")

    w("## 3. Forecasting setup\n")
    w(f"- **Test window (latest data):** {ts:%Y-%m-%d} to {te:%Y-%m-%d} ({(te - ts).days + 1} days).")
    w("- **Rolling-origin backtest:** for every origin day, each model is given *only* data up to that day and "
      "forecasts the next 7 days; each target day is scored at horizons 1-7. Nothing after an origin is ever visible "
      "(enforced by `tests/test_no_leakage.py`). No external or non-public features are used.")
    w("- Models: naive / seasonal-naive / moving averages / same-weekday mean (baselines), damped-trend ETS, "
      "Holt-Winters (weekly), a regime-aware weekday mean (see change points), ARIMA(1,0,1), SARIMA(1,0,1)(1,0,1,7), and a direct multi-horizon gradient-boosting model "
      "on lag features. Statistical models are refit daily on the last 365 days; GBM is refit weekly on the last 730 days.")
    w(f"- Metrics: MAE and RMSE in oblast-hours/day; MASE scaled by the in-sample one-step naive MAE "
      f"({main['scale']:.2f}). Baseline models are those named in the table below as such.\n")

    w(f"## 4. Results on the latest window ({ts:%Y-%m-%d} to {te:%Y-%m-%d})\n")
    tbl = s.copy()
    tbl.insert(0, "type", ["baseline" if m in BASELINES else "model" for m in tbl.index])
    w(tables.md(tbl, index_name="model", signed=("bias_1to7",)) + "\n")
    w(f"Lowest overall MAE: **{best}**; best baseline: **{best_base}**. "
      f"Mean level in the test window is {test_mean:.1f} oblast-hours/day vs {prev_mean:.1f} over the preceding 365 days"
      f"{' - an elevated level, which penalises mean-reverting models (note their negative bias)' if test_mean > prev_mean * 1.05 else ''}.\n")
    w("Paired difference in MAE vs the best baseline (negative = better), 95% moving-block bootstrap CI "
      "(7-day blocks):\n")
    shown = cmp_.copy()
    shown["verdict"] = np.where(~shown["significant"], "no clear difference",
                                np.where(shown["dMAE"] < 0, "better", "worse"))
    w(tables.md(shown[["reference", "dMAE", "ci_low", "ci_high", "verdict"]], index_name="model",
                signed=("dMAE", "ci_low", "ci_high")) + "\n")
    if sig_better:
        w(f"Models significantly better than the best baseline: {', '.join(sig_better)}.\n")
    else:
        w("**No model is significantly better than the best baseline on this window.** "
          "With only ~8 weeks of test days, small differences should not be over-interpreted.\n")
    for f in ("06_forecast_h1.png", "06_forecast_h7.png", "07_mae_by_horizon.png", "08_error_diagnostics.png"):
        w(f"![{f}](figures/{f})")
    w("")

    if len(folds) > 1:
        w("## 5. Robustness: the same evaluation on earlier windows\n")
        names = ["W1 (latest)"] + [f"W{i + 1}" for i in range(1, len(folds))]
        legend = "; ".join(f"**{n}** = {f['test_start']:%Y-%m-%d} to {f['test_end']:%Y-%m-%d}"
                           for n, f in zip(names, folds))
        w(f"Overall MAE (horizons 1-7) per model per window, with the model's rank (1 = best) in that window. "
          "If a model's rank is stable across windows, the ordering is more trustworthy than the latest window alone.\n")
        w(f"Windows: {legend}.\n")
        table = pd.DataFrame(index=s.index)
        rank_cols = []
        for n, f in zip(names, folds):
            mae = f["summary"]["MAE_1to7"].reindex(s.index)
            table[f"{n} MAE"] = mae
            table[f"{n} rank"] = mae.rank().astype(int)
            rank_cols.append(f"{n} rank")
        table["mean rank"] = table[rank_cols].mean(axis=1)
        table = table.sort_values("mean rank")
        w(tables.md(table, index_name="model", decimals={"mean rank": 1}) + "\n")
        top = table.index[0]
        w(f"Best mean rank across windows: **{top}** ({table.loc[top, 'mean rank']:.1f}). "
          f"Winner of the latest window alone: **{best}**.\n")

    n_sec = 6 if len(folds) > 1 else 5
    if "segments" in extra:
        seg = extra["segments"].copy()
        seg["start"] = seg["start"].dt.strftime("%Y-%m-%d"); seg["end"] = seg["end"].dt.strftime("%Y-%m-%d")
        seg["days"] = seg["days"].astype(int)
        w(f"## {n_sec}. Regimes (change points)\n")
        n_sec += 1
        w("Mean-shift change points in the last three years of the daily series (exact optimal partitioning, minimum "
          "regime length 21 days, a deliberately strict penalty because daily alert-hours are autocorrelated). Each row is "
          "a regime; `change_vs_previous` is the relative jump of its mean. Detected breaks are descriptive - they often "
          "coincide with changes in alerting practice or the recording source, not only the threat itself. The "
          "`RegimeSeasonalMean` model forecasts from the current regime only.\n")
        w(tables.md(seg.set_index("start"), index_name="start", decimals={"mean": 1, "change_vs_previous": 2},
                    pct=("change_vs_previous",)) + "\n")
        w(f"![{extra['segments_fig']}](figures/{extra['segments_fig']})\n")
    if "interval_summary" in extra:
        isum = extra["interval_summary"]
        w(f"## {n_sec}. Prediction intervals\n")
        n_sec += 1
        w(f"Empirical intervals for **{extra['interval_model']}** (the lowest-MAE model on this window): the point "
          "forecast minus quantiles of that model's own errors at the same horizon, using only errors already observed "
          "at the forecast origin (trailing 42 days). `coverage` should be close to `nominal`; "
          "`winkler_score` combines width and misses (lower is better).\n")
        w(tables.md(isum, index_name="nominal", decimals={"coverage": 2, "mean_width": 1, "winkler_score": 1, "n": 0},
                    include_index=True) + "\n")
        w("With a window of about 56 days, coverage figures carry several points of sampling noise. Coverage below nominal "
          "is expected when the level shifts, because the past errors the intervals are built from lag behind the shift.\n")
        w(f"![{extra['interval_fig']}](figures/{extra['interval_fig']})\n")
    if "by_oblast" in extra:
        per = extra["by_oblast"].copy()
        w(f"## {n_sec}. Per-oblast models and bottom-up forecast\n")
        n_sec += 1
        w("Each oblast's own daily series (0-24 alert-hours/day) was backtested with the fast models "
          "(baselines, Regime, ETS-HW7) on the same window. `gain_vs_baseline` is the MAE reduction of the best model "
          "over the best baseline (0 means a baseline won).\n")
        w(tables.md(per, index_name="oblast", decimals={"mean_h_per_day": 2, "MAE_best_baseline": 2, "MAE_best": 2,
                                                       "gain_vs_baseline": 2, "MAE_best_non_baseline": 2},
                    pct=("gain_vs_baseline",)) + "\n")
        w("Bottom-up national forecast = sum of per-oblast forecasts from the same model, scored against the national "
          "total, compared with forecasting the national total directly (top-down). Negative = bottom-up is better:\n")
        w(tables.md(extra["by_oblast_national"], index_name="model",
                    decimals={"MAE_top_down": 2, "MAE_bottom_up": 2, "bottom_up_vs_top_down": 3},
                    signed=("bottom_up_vs_top_down",)) + "\n")
    w(f"## {n_sec}. Caveats\n")
    if data.source == "combined":
        w("- **Combined series:** the volunteer part is not comparable with the official part (see the splice diagnostics "
          "in section 1). Models are scored on a window that may include the splice, so errors there partly reflect the "
          "level shift.")
    w(f"- Source used: **{data.source}** dataset. It is unofficial/community-maintained; for the volunteer file ~5% of alerts "
      "have a guessed 30-minute end. Luhansk and Crimea are not represented.")
    w("- **Sources disagree.** Cross-checks (`python validate.py`) show the volunteer and official datasets differ strongly, "
      "in region-specific ways, since late 2025 (raion-level alerts, yellow/red threat levels). For Kyiv City, the "
      "'Digital Kyiv' history as quoted by Censor.NET reports 242 h 40 min of alerts in Sept 2026 vs ~71 h in the "
      "volunteer file. Treat the 2026 level in this report as 'alerts as recorded by this source', not as ground truth.")
    for msg in data.warnings:
        if "longer than 72 h" not in msg:
            w(f"- Data-quality warning: {msg}")
    w("- The series reflects attack activity *and* changes in alerting/recording practice; a level shift is not "
      "necessarily a change in threat.")
    w("- One short test window and eleven candidate models: the best-looking model is partly selected by luck. Use the "
      "confidence intervals and the robustness table, not the single ranking.")
    w("- This is a descriptive/statistical exercise on aggregate alert load. It does not predict specific attacks and "
      "must not be used for personal safety decisions; rely on official alert channels for that.")
    return "\n".join(out) + "\n"
