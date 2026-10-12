> **Note:** this project was written by Claude (Sonnet 5.5), an AI assistant made by Anthropic, in a conversation with
> a human who set the goals and asked for the features. Review the code and results before relying on them.

*Українська версія: [README-UA.md](README-UA.md)*

# Time series analysis of air raid alerts in Ukraine

A small, reproducible project: exploratory time series analysis plus an honest
forecasting backtest on the **latest publicly available** alert data.

## What's here

| Command                                     | Purpose                                                                       |
| :------------------------------------------ | :---------------------------------------------------------------------------- |
| `python run.py`                             | Exploratory analysis + daily-load forecasting backtest -> `outputs/report.md` |
| `python predict.py --oblast "Kyiv City"`    | Hour-by-hour alert probabilities for the next 24 h in one oblast (~1 min)     |
| `python predict.py --oblast all --backtest` | Measured accuracy of that model, per oblast                                   |
| `python validate.py`                        | Cross-checks the data sources against each other                              |

**Full usage guide and explanation of every output: [USAGE.md](USAGE.md).**

## Quick start

```bash
pip install -r requirements.txt
python run.py                 # fetches the data (cached), runs EDA + backtest, writes outputs/report.md
python run.py --update        # same, but re-fetches the dataset first
python run.py --folds 1 --skip-sarima   # fast variant (~40 s; the full run takes ~2 min)
pytest -q                     # tests, including no-leakage checks for both backtests
```

Everything lands in `outputs/`: `report.md` (auto-generated, open this first), `figures/`,
`metrics_latest_window.csv`, `predictions_latest_window.csv`, `daily_series.csv`.
The dataset is cached in `data/` and re-downloaded automatically when older than 12 h
(`--update` forces a re-fetch). If GitHub is unreachable the loader tries a CDN mirror, then a stale cache;
see USAGE.md, section 3, for fallbacks and how to use a custom mirror.

## Data

[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(MIT), **volunteer file** (alerts from the eTryvoga channel). It is the default instead of the "official"
file because it is current to the day and has one consistent granularity (oblast) for its
entire history. The official file lags by weeks and switched from oblast to raion/hromada-level
alerts, which would create an artificial structural break.

`--source official` and `--source combined` (official until it ends, then volunteer) are also available. The two
sources differ strongly in level, so `combined` jumps at the splice and is experimental; details in USAGE.md.

Cleaning: Luhansk oblast dropped (permanent siren not in data), non-positive durations
dropped, overlapping intervals per oblast merged, ~5% of alerts with a guessed 30-min end kept
and reported. Day boundaries use Europe/Kyiv time. The in-progress current day is excluded.

**Target:** `alert_hours` = oblast-hours under alert per day (sum over 24 oblasts). Counts of
alerts are *not* used as the target because they are distorted by changes in how alerts are
issued (starts/day rose ~65% in 2026 while alert-hours rose ~11%).

## Method

* EDA: rolling means, STL decomposition, ACF, ADF/KPSS, hour x weekday heatmap, regional split.
* Models: Naive, SeasonalNaive7, MovingAvg7/28, SeasonalMean4w, RegimeSeasonalMean (weekday mean over the current regime); damped ETS, Holt-Winters(7);
  ARIMA(1,0,1), SARIMA(1,0,1)(1,0,1,7); direct multi-horizon gradient boosting on lag features.
* Evaluation: rolling-origin backtest over the latest 56 complete days, horizons 1-7 days;
  every model sees only data up to its origin (`tests/test_no_leakage.py` proves it by
  corrupting future values). MAE / RMSE / bias / MASE, paired block-bootstrap CIs vs the best
  baseline, and the same evaluation on two earlier windows as a robustness check.
* Also reported: change points (regimes), empirical 50%/80% prediction intervals with measured coverage, and with
  `run.py --by-oblast` per-oblast models plus a bottom-up national forecast.
* No external or non-public inputs (no attack data, weather, or "insider" signals).

## Layout

```
run.py / predict.py / validate.py   command-line entry points
src/ua_alerts/sources.py   mirrors, cache validation, quality checks
src/ua_alerts/data.py      clean, merge, hourly/daily aggregation (volunteer or official source)
src/ua_alerts/forecast.py  24-hour probabilistic outlook (per-oblast model) + backtest tables
src/ua_alerts/pooled.py    default 24-hour model: one classifier for all oblasts, neighbour features, calibration
src/ua_alerts/tables.py    aligned table renderer used by every report
src/ua_alerts/eda.py       plots and statistical tests
src/ua_alerts/models.py    forecasters (common fit/predict interface)
src/ua_alerts/evaluate.py  backtest, metrics, bootstrap, plots
src/ua_alerts/report.py    Markdown report builder
tests/                     data, source-fallback, forecast, leakage and docs tests
tools/format_md_tables.py  aligns Markdown tables in the docs (`--check` mode is used by the tests)
```

## Limitations

Unofficial source; one short test window; level shifts can reflect alerting practice rather
than threat. This analyzes aggregate alert load and cannot predict specific attacks. Do not
use it for personal safety decisions; use official alert channels.

The 24-hour outlook (pooled model) is only a few percent better than "same hour over the last four weeks" in Brier
score (+2% to +5% on average over four recent 28-day windows, against +1% to +3% for the earlier per-oblast model) and
is no better for a couple of oblasts (see the all-oblast backtest). The data sources also disagree strongly since late 2025 (`validate.py`).

Implemented from the original ideas list: regime detection (change points) and a regime-aware model, empirical
prediction intervals for the daily forecasts, per-oblast daily models with a bottom-up national forecast
(`run.py --by-oblast`), and hourly-resolution forecasting (`predict.py`). Also implemented for the 24-hour model: pooling
across oblasts with neighbour features, time-weighted training, per-oblast blend weights from rolling validation
windows, and calibration. Still open: alert-onset (spread) features, quantile forecasts for the hourly model, level
adjustment at the official/volunteer splice, and a model that uses regime information in the hourly task.

## To-do

- **Kyiv City open-data JSON** (data.gov.ua `airalerts.json`) as a third cross-check in `validate.py`. A tolerant
  parser exists but is commented out in `src/ua_alerts/sources.py`: the file's real schema could not be checked.
- **alerts.in.ua live status** in `predict.py` (a "LIVE: alert active now" line). The client is commented out in
  `src/ua_alerts/sources.py`: it needs a personal API token and has never been run against the real service.
- Open modelling ideas are listed under "Ideas to extend" above.

## License

[MIT](LICENSE). The alert data belongs to its sources; see the dataset repository for its terms.
