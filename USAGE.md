# Usage guide and output reference

This guide covers every command, every file the project writes, and how to read each number.
For the project overview see [README.md](README.md).

> **Safety note.** Everything here is a statistical description of past alert patterns. It cannot
> tell you whether an attack is coming and must not be used for personal safety decisions. Use
> official alert channels for that.

## Contents

1. [Setup](#1-setup)
2. [The four commands at a glance](#2-the-four-commands-at-a-glance)
3. [Data sources, caching and fallbacks](#3-data-sources-caching-and-fallbacks)
4. [`predict.py`: the next-24-hours outlook](#4-predictpy-the-next-24-hours-outlook)
5. [`run.py`: the full analysis and its report](#5-runpy-the-full-analysis-and-its-report)
6. [`validate.py`: do the data sources agree?](#6-validatepy-do-the-data-sources-agree)
7. [Reading the metrics](#7-reading-the-metrics)
8. [Output file reference](#8-output-file-reference)
9. [Troubleshooting](#9-troubleshooting)
10. [Using it from Python](#10-using-it-from-python)

---

## 1. Setup

```bash
pip install -r requirements.txt     # Python 3.10+ (developed on 3.12)
pytest -q                           # optional: run the test suite (about 25 s)
```

No API key is needed for anything. The first run downloads about 9 MB (volunteer file) and caches
it in `data/`.

## 2. The four commands at a glance

| Command                                      | What it does                                                                    | Typical runtime                           |
| :------------------------------------------- | :------------------------------------------------------------------------------ | :---------------------------------------- |
| `python predict.py --oblast "Kyiv City"`     | Hour-by-hour alert probabilities for the next 24 h                              | ~1 min (`--model per-oblast`: ~5 s)       |
| `python predict.py --oblast Lviv --backtest` | How accurate was that model for this oblast over the last 28 days?              | ~1 min (`--oblast all`: the same ~1 min)  |
| `python run.py`                              | Exploration + 7-day daily-load forecasting backtest, writes `outputs/report.md` | ~2 min (`--folds 1 --skip-sarima`: ~40 s) |
| `python validate.py`                         | Compares the data sources against each other                                    | ~5 s                                      |

All three scripts accept `--update` (re-fetch the dataset, ignoring the local cache; `--refresh` is an alias).
`run.py` and `predict.py` also accept `--source volunteer` (default), `official` or `combined` (see [Sources](#sources)).
`predict.py` additionally has `--model` (`pooled` by default, or `per-oblast`). Use `--help` on any of them for
the full flag list.

## 3. Data sources, caching and fallbacks

### Sources

| Name                  | File                    | Coverage                                                                         | Updated                                                                              | Good for                     |
| :-------------------- | :---------------------- | :------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :--------------------------- |
| `volunteer` (default) | `volunteer_data_en.csv` | oblast level, from 25 Feb 2022                                                   | daily                                                                                | forecasting, recent analysis |
| `official`            | `official_data_en.csv`  | oblast, then raion/hromada level (mostly raion since Dec 2025), from 15 Mar 2022 | documented as daily, but its newest record was 2026-09-07 when checked on 2026-10-12 | cross-checking, history      |

| `combined` (experimental) | official file up to its last record, then the volunteer file after that | official history, then volunteer | as fresh as the volunteer file | **not recommended for forecasting** (see below) |

The first two come from the open [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
(MIT). The official file includes "administrative" alerts for occupied areas that stay active for
months, so alerts longer than 72 h are dropped when it is loaded.

**About `combined`.** It uses the official file up to its last record (the "seam") and the volunteer file after it,
so the series stays current. The two files measure alerts differently (the official one includes raion/hromada-level
alerts and is far higher for some oblasts), so the level can jump at the seam. When checked on 2026-10-12, 15 of 24
oblasts changed by more than 2x at the seam (for example Donetska 19.1 -> 4.0 alert-hours/day, Volynska 0.05 -> 3.93),
and the daily-load forecast error of `run.py --source combined` was about four times the volunteer-only error. The
loader prints a seam warning, and `report.md` gets a "splice diagnostics" table in section 1. Treat results on this
source as exploratory; use `volunteer` for forecasting.

### What happens when the upstream disappears

`ua_alerts/sources.py` tries, in order:

1. a **fresh local cache** (younger than 12 h; `predict.py` uses 3 h; `--update` skips this step);
2. a **custom URL**, if the environment variable `ALERTS_VOLUNTEER_URL` / `ALERTS_OFFICIAL_URL` is set (use this to point at a fork or a self-hosted copy);
3. GitHub raw, then the jsDelivr CDN mirror;
4. the **stale cache**, with a note saying how old it is.

A download only replaces the cache if it has the expected columns and a plausible number of rows,
so an HTML error page or a truncated file cannot overwrite good data. If there is no cache and every
URL fails, the error message says where to place the file by hand.

If the upstream repository is ever removed or made private, host a copy of the CSVs (a fork works) and set the variable:

```bash
export ALERTS_VOLUNTEER_URL="https://example.com/my_copy/volunteer_data_en.csv"
```

### Quality warnings

Every load runs the checks below and prints a warning for each one that fails. Warnings appear on the console
and in `report.md`.

| Check               | Warning is raised when                                                                                           |
| :------------------ | :--------------------------------------------------------------------------------------------------------------- |
| Staleness           | the newest record is more than 48 hours older than the current time                                              |
| Collection gaps     | at least one of the last 90 days has zero alerts nationwide (probably a data gap)                                |
| Volume change       | the number of alerts in the last 30 days is more than 2x or less than 0.5x the number in the 30 days before that |
| Clock               | the newest record lies in the future relative to the system clock                                                |
| Missing "all clear" | an alert lasts longer than 72 hours                                                                              |

### Other sources and their status

- **data.gov.ua Kyiv City JSON** and the **alerts.in.ua live API** are *not* active: their code is kept as
  comments in `src/ua_alerts/sources.py` until it can be tested against the real services (see the To-do list in the README).
- **Keyless community mirrors** of the official UkraineAlarm API exist (for example siren.pp.ua). None is
  integrated: they are run by individuals, publish no guarantees, and were not verified.

## 4. `predict.py`: the next-24-hours outlook

### Commands

```bash
python predict.py --list                                  # valid oblast names
python predict.py --oblast "Kyiv City"                    # forecast from the current time
python predict.py --oblast Харківська                      # Ukrainian and partial names work
python predict.py --oblast Odeska --json                  # JSON only, for scripts
python predict.py --oblast "Kyiv City" --now "2026-10-11 18:00"   # replay (see below)
python predict.py --oblast Lviv --backtest                # measured accuracy for one oblast
python predict.py --oblast all --backtest                 # ... for all oblasts
python predict.py --oblast Lviv --model per-oblast        # the original, faster (~5 s), less accurate model
```

Name matching accepts English, Ukrainian and prefixes: `kyiv` means Kyiv City, `Київська область`
means Kyiv Oblast. An unknown name prints suggestions.

The units are administrative oblasts, and the names follow the dataset's oblast spelling, so a city that is
not a unit of its own is found under its oblast: use `Odeska` (or `Одеська`) for Odesa, not `Odesa`. Kyiv City is the
only city that is a separate unit.

### What it predicts

For each of 24 one-hour slots (Kyiv local time; the first slot is the hour that contains the current time):
**P(the oblast is under an air raid alert during at least part of that hour)**.

**Default model (`--model pooled`).** One gradient-boosting classifier is trained on all 24 oblasts at once (the
oblast is an input), so each oblast benefits from 24 times more training rows. Its features are the oblast's own
recent history (alert state, time since the last alert, rolling alert shares from 3 h to 4 weeks), nationwide
activity, the recent alert state of the **neighbouring oblasts** (approximate land borders; Kyiv City borders Kyiv
Oblast), the hour of day, the weekday and the 28-day same-hour climatology. Older training rows count less (half-life
120 days). The result is then:

1. **blended** with plain same-hour climatology, with a weight per oblast chosen on three consecutive 14-day
   validation windows before the forecast origin and shrunk halfway toward the all-oblast optimum;
2. **calibrated** with a smooth logistic (Platt) correction fitted on the same validation forecasts, so that "30%"
   means about 30% (see [7. Reading the metrics](#7-reading-the-metrics)).

Fitting takes about a minute because the validation windows need their own models. `--model per-oblast` selects the
original model (one classifier per oblast, no neighbours, one validation window, no calibration): about 5 s, but less
accurate.

The "any alert in the next 24 h" figure always comes from a separate per-oblast model.

Because the upstream file updates about once a day, "now" is usually hours after the last
observed hour. The model is trained for exactly that gap, so its information fades toward climatology
the staler the data is. Beyond 96 h of staleness it refuses to forecast.

### Reading the console output

```
Air raid alert outlook: Kyiv City
  generated   2026-10-12 18:25 Kyiv time
  data as of  2026-10-12 07:59 Kyiv time (10.4 h old) | source: volunteer
  window      Mon 12 Oct 18:00 -> Tue 13 Oct 18:00 (24 hourly slots, ...)

slot (Kyiv)         P(alert)  chart
Mon 12 18:00-19:00       29%  ████████
...
Summary
  expected hour-slots touched by an alert    : 6.7 of 24   (recent norm: 6.8 of 24; ...)
  P(at least one alert in the window)        : 99%   (uninformative: 28/28 recent days had an alert)
  likeliest slots                            : 20:00 (34%), 22:00 (32%), 12:00 (31%)
  model                                      : pooled
  model weight vs climatology                : 0.7  (chosen on validation data; 0 = no skill shown)

Recent history (latest data point: Mon 12 Oct 07:59 Kyiv time)
  this oblast         : no alert in the latest hour of data; the last alert hour was 8 h earlier
  time under alert    : 0.6 h in the 24 h before that, 19.8 h in the previous 7 days
  nationwide          : 1 of 24 oblasts under alert in the latest hour of data (this oblast included)
```

| Line                          | Meaning                                                                                                                                                                                                                                                                                                  |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generated`                   | Time of the run (Kyiv time).                                                                                                                                                                                                                                                                             |
| `data as of`                  | Newest record in the dataset and how old it is. A large age means the first slots are mostly climatology.                                                                                                                                                                                                |
| `P(alert)`                    | Per-slot probability of an alert touching that hour. Not an alert *duration*.                                                                                                                                                                                                                            |
| `expected hour-slots touched` | Sum of the 24 probabilities: the expected number of hour slots containing an alert, compared with the `recent norm` (same count, last 28 days). Not a duration: a single 20-minute alert touches one slot.                                                                                               |
| `P(at least one alert)`       | Separate window model. For oblasts with an alert almost every day this is near 100% and uninformative; the tool says so.                                                                                                                                                                                 |
| `likeliest slots`             | Top three slots. Differences among them are usually small.                                                                                                                                                                                                                                               |
| `model`                       | `pooled` (default) or `per-oblast`.                                                                                                                                                                                                                                                                      |
| `model weight vs climatology` | 1.0 = fully trust the model, 0.0 = the model showed no skill in validation so climatology is used.                                                                                                                                                                                                       |
| `Recent history`              | What the dataset shows up to its newest record (the `latest data point`, not the present moment): whether this oblast was under alert in that last hour, hours since its last alert, alert time in the 24 h and 7 days before that point, and how many oblasts were under alert nationwide in that hour. |

### JSON output

`--json` prints one object: `oblast`, `generated_at_utc`, `data_as_of_utc`, `data_age_hours`, `source`,
`model`, `model_weight_vs_climatology`, `p_at_least_one_alert_in_window`, `expected_hour_slots_touched_by_alert`,
`slots` (list of `start_local`, `end_local`, `probability`), `recent_state` and `warnings`.

### Replay mode (`--now`)

`--now "2026-10-11 18:00"` sets the current time to that moment (Kyiv time unless an offset such as `+00:00`
is included) and uses **only data known by then**. A `REPLAY` warning is printed. Use it to see what the
tool would have said at that time, then compare with what the data shows actually happened.

### Files written (unless `--no-plot`)

`outputs/forecast_<oblast>_<YYYYMMDD_HHMM>.csv` (slots and probabilities) and a matching `.png` bar chart.

### `--backtest`: how accurate is it for this oblast?

The model is trained only on data from before the test window (the last 28 days, set with `--test-days`),
then every hour of the window is treated as a simulated run: forecast the next 24 slots from that moment, compare with what happened.
With the pooled model all oblasts are fitted in one go, so `--backtest` takes about a minute for one oblast or for all.
Example for Kyiv City (16,128 predictions); `model (GBM + climatology blend)` is the final forecast (blended, and
calibrated for the pooled model) and `model (GBM alone)` is the raw classifier:

```
predictor                        Brier  LogLoss    AUC  skill_vs_clim
model (GBM + climatology blend)  0.204    0.599  0.526         +0.032
model (GBM alone)                0.204    0.600  0.526         +0.031
climatology (28d same hour)      0.211    0.616  0.496         +0.000
recent rate (last 7 days)        0.242    0.762  0.452         -0.148
base rate (training mean)        0.229    0.688  0.500         -0.086
```

In practice the model is only a few percent better than "what happened at this hour over the last four weeks", and
for a few oblasts it is no better. Sections printed by `--backtest`:

- **Hourly task:** the table above. `climatology` is the benchmark to beat.
- **By horizon:** the same comparison for forecasts 1-3 h, 4-12 h and 13-24 h ahead.
- **Calibration:** among all hours where the model said "about 30%", how often did an alert really happen?
- **Window task:** "any alert in the next 24 h".

It also writes `outputs/backtest_<oblast>.png` (calibration and by-hour-of-day plots) and `.txt`.

**All oblasts:** `--oblast all --backtest` writes `outputs/backtest_all_oblasts.txt/.csv`. On the 28 days
ending 2026-10-12 the pooled model beat climatology in 21 of 23 oblasts that had alerts (Zakarpattia had none),
ranging from about +38% (Ternopil, probably a one-off) to slightly negative values (Chernihiv -1.0%, Kharkiv -0.2%),
with Kyiv City at +3.2%. Check the table before relying on the outlook for a particular oblast.

**Pooled vs per-oblast, and what mattered.** Average skill versus climatology (the mean Brier over all 24 oblasts) on four
consecutive 28-day windows:

| Window ending | climatology Brier | per-oblast model | pooled model | oblasts beating climatology (per-oblast / pooled) |
| :------------ | ----------------: | ---------------: | -----------: | ------------------------------------------------: |
| 2026-10-12    |             0.187 |           +0.013 |       +0.030 |                                           15 / 21 |
| 2026-09-14    |             0.179 |           +0.015 |       +0.048 |                                           17 / 22 |
| 2026-08-17    |             0.158 |           +0.028 |       +0.045 |                                           19 / 23 |
| 2026-07-20    |             0.151 |           +0.007 |       +0.020 |                                           14 / 18 |

The pooled model was better on all four windows. Most of the gain comes from pooling (24 times more training rows,
plus the neighbour features); the blend weights and the calibration changed the average Brier by only a few thousandths
(sometimes up, sometimes down). Calibration is kept because it makes the stated probabilities honest rather than because
it adds much accuracy. Time weighting of training rows added about +0.2 points on the one window where it was isolated.
An isotonic (staircase) calibrator was tried and rejected: no more accurate, and it collapses the forecast into a few
distinct values.

**One caveat on the weights:** they come from 42 days of validation, so they are noisy; a weight of 1.0 means the
validation windows showed no reason to mix in climatology, not that the model is certain.

## 5. `run.py`: the full analysis and its report

```bash
python run.py                          # latest data, 56-day test window + 2 earlier windows
python run.py --update                 # re-fetch the dataset first, then run
python run.py --folds 1 --skip-sarima  # fast variant
python run.py --by-oblast              # also per-oblast daily models + bottom-up forecast (adds ~5 min)
python run.py --test-days 28           # shorter test windows
python run.py --source official        # same analysis on the official dataset
```

This builds the **daily series** (oblast-hours under alert per day, summed over 24 oblasts, so the
maximum is 576), explores it, and backtests seven-day-ahead forecasts of it with eleven models (including a regime-aware one). Everything
is summarized in `outputs/report.md`.

### Sections of `report.md`

| Section              | What it tells you                                                                                       |
| :------------------- | :------------------------------------------------------------------------------------------------------ |
| 1. Data              | Source, rows dropped or imputed, the target definition, and a per-year table.                           |
| 2. Exploration       | Trend and weekly-seasonality strength, stationarity tests, busiest hour and oblasts, plus five figures. |
| 3. Forecasting setup | Test window, how the backtest avoids leakage, which models, which metrics.                              |
| 4. Results           | Metrics table, the paired comparison against the best baseline, forecast plots.                         |
| 5. Robustness        | The same ranking on earlier windows, plus mean rank.                                                    |
| 6. Regimes           | Detected change points (mean shifts) in the last three years, the regime means and a figure.            |
| 7. Intervals         | Empirical 50% / 80% prediction intervals for the best model, with their measured coverage.              |
| 8. Per-oblast        | Only with `--by-oblast`: per-oblast models and the bottom-up national forecast.                         |
| Last. Caveats        | Source limits, the cross-source disagreement, selection-by-luck warning.                                |

**Per-year table columns.** `alert_hours_per_day` is the mean of the target. `starts_per_day` is the mean number of
alert starts. `hours_per_alert` is their ratio. Starts rose sharply in 2026 while hours barely changed:
alerts became shorter and more numerous, which is why the target is duration, not count.

**Section 2 numbers.** STL "strength" runs from 0 (none) to 1 (fully deterministic). Weekly-seasonal
strength is about 0.06, so day-of-week barely matters. ADF p-value small means "no unit root";
KPSS p below 0.05 means the level drifts. Both can be true: the series is mean-reverting but its mean moves.

**Section 4 table.** One row per model; see [7. Reading the metrics](#7-reading-the-metrics) for columns.
The comparison table gives `dMAE` (model MAE minus best baseline MAE; negative is better), a 95%
block-bootstrap interval, and a verdict: `better`, `worse`, or `no clear difference`. On the latest window
(2026-08-17 to 2026-10-11) no model beat the 28-day moving average clearly, and ARIMA ranked first on
the other two windows. The report prints both, and that is the point of section 5.

### Figures (`outputs/figures/`)

| File                                       | Shows                                                                           |
| :----------------------------------------- | :------------------------------------------------------------------------------ |
| `01_daily_series.png`                      | Daily load with 28-day mean; below it, alert starts versus average alert length |
| `02_stl.png`                               | Trend, weekly seasonality and residual                                          |
| `03_hour_weekday_heatmap.png`              | Average number of oblasts under alert by weekday and hour (last 180 days)       |
| `04_acf.png`                               | Autocorrelation of the levels and of first differences                          |
| `05_regions.png`                           | Alert hours per day by oblast over the last 90 days                             |
| `06_forecast_h1.png`, `06_forecast_h7.png` | Actual vs best baseline and best two models, 1-day and 7-day ahead              |
| `07_mae_by_horizon.png`                    | Error growing with horizon, all models                                          |
| `08_error_diagnostics.png`                 | Best model's one-day errors, predicted vs actual                                |
| `09_changepoints.png`                      | Daily series with the detected regimes and their means                          |
| `10_intervals.png`                         | 1-day-ahead forecast with 50% and 80% prediction intervals                      |

## 6. `validate.py`: do the data sources agree?

```bash
python validate.py                         # everything it can reach
python validate.py --months 6              # shorter window
```

It writes `outputs/validation_report.md` and `outputs/validation_by_oblast.csv` and always exits
with code 0: a disagreement is a result.

| Section                | Contents                                                                                                                                                             |
| :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Source health       | Status, origin, cache age, rows, first and latest record per source. A failed source shows as `FAILED` or `unavailable` with the reason.                             |
| 2. Per oblast          | `volunteer_h` and `official_h` (alert-hours over the window), `ratio` (volunteer / official), the ratio in each half of the window, daily correlation and a verdict. |
| 3. Kyiv month by month | Volunteer and official CSV for Kyiv City. `-` means the source does not cover that month fully.                                                                      |
| 4. Reference points    | Published official figures with links (edit `REFERENCE_POINTS` in `validate.py` to add more)                                                                         |

**Verdicts.** `consistent`: ratio 0.5 to 2, daily correlation at least 0.5, ratio stable between halves.
`DIVERGES`: ratio drifting 3x or more between halves, daily correlation below 0.2, or a ratio below 0.25 or above 4.
`differs`: in between.

**Example result** (run on 2026-10-12, comparing the 12 months from September 2025 to August 2026 across 24 oblasts):
10 `DIVERGES`, 8 `consistent`, 6 `differs`. The
official file shows several times more alert time for the border oblasts (Dnipropetrovsk, Donetsk, Kharkiv,
Sumy, Zaporizhzhia) because it records smaller-area alerts. The reverse holds in the west (Volyn, Lviv), where
the volunteer file shows far more than the official one. For Kyiv City in September 2026 the City's own
"Digital Kyiv" history, as quoted by Censor.NET, reports 242 h 40 min against about 71 h in the volunteer file.
Kyiv introduced tiered threat levels (yellow/red) on 6 September 2026, which may be part of the explanation, but
this tool cannot prove that. Conclusion: use absolute 2026 levels with care, and prefer conclusions that hold on both sources.

## 7. Reading the metrics

### Daily-load forecasting (`run.py`)

| Metric             | Meaning                                                                                       | Lower/higher is better |
| :----------------- | :-------------------------------------------------------------------------------------------- | :--------------------- |
| `MAE_h1`, `MAE_h7` | Mean absolute error one and seven days ahead, in oblast-hours per day                         | lower                  |
| `MAE_1to7`         | Mean absolute error pooled over horizons 1-7                                                  | lower                  |
| `RMSE_1to7`        | Like MAE but punishes big misses more                                                         | lower                  |
| `bias_1to7`        | Mean(prediction - actual). Negative = under-predicts                                          | closer to 0            |
| `MASE_1to7`        | MAE divided by the one-step naive MAE on the preceding year. Below 1 beats "repeat yesterday" | lower                  |

Baselines are Naive (repeat yesterday), SeasonalNaive7 (repeat last week), MovingAvg7/28, SeasonalMean4w
(same weekday averaged over four weeks). `RegimeSeasonalMean` is a model, not a baseline: it detects mean-shift
change points in the history available at each origin and averages the same weekday over the current regime only
(a shift is confirmed about three weeks after it happens, so it lags sudden changes).

**Prediction intervals** (section 7 of the report):

| Metric          | Meaning                                                                                                                     |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| `nominal`       | The probability the interval is meant to contain (0.5 or 0.8).                                                              |
| `coverage`      | Share of test days whose actual value fell inside the interval. Should be near `nominal`; below it means overconfident.     |
| `mean_width`    | Average interval width in oblast-hours per day. Narrower is better at the same coverage.                                    |
| `winkler_score` | Width plus a penalty for each miss (scaled by 2 / (1 - nominal)); a single number that rewards narrow, well-covering bands. |

The intervals are the point forecast minus quantiles of the model's own earlier errors at the same horizon, using only
errors already observed at the origin.

**Per-oblast table** (`--by-oblast`): `gain_vs_baseline` is how much smaller the best model's MAE is than the best
baseline's (0% = a baseline won). The bottom-up table compares summing per-oblast forecasts with forecasting the
national total directly; negative `bottom_up_vs_top_down` favours bottom-up.

### Hourly probabilities (`predict.py --backtest`)

| Metric            | Meaning                                                                                                                                             |
| :---------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Brier`           | Mean squared error of the probabilities. Lower is better. Always compare against climatology on the same data.                                      |
| `LogLoss`         | Like Brier but punishes confident mistakes harder. Lower is better.                                                                                 |
| `AUC`             | Does the model rank alert hours above quiet hours? 0.5 = no better than chance. Ignores calibration.                                                |
| `skill_vs_clim`   | 1 - Brier / Brier(climatology). Positive means better than climatology. Shown as `-` when nothing happened in the window (for example Zakarpattia). |
| Calibration table | Per probability bin: `n`, `mean_predicted`, `observed_rate`. These should match.                                                                    |

## 8. Output file reference

| File                                                       | Written by                           | Content                                                                          |
| :--------------------------------------------------------- | :----------------------------------- | :------------------------------------------------------------------------------- |
| `outputs/report.md`                                        | `run.py`                             | The full analysis                                                                |
| `outputs/figures/*.png`                                    | `run.py`                             | Figures listed in section 5                                                      |
| `outputs/metrics_latest_window.csv`                        | `run.py`                             | Daily-forecast metrics, latest window                                            |
| `outputs/predictions_latest_window.csv`                    | `run.py`                             | Every daily prediction: model, origin, target, horizon, actual, predicted, error |
| `outputs/daily_series.csv`                                 | `run.py`                             | The daily target series                                                          |
| `outputs/intervals_latest_window.csv`                      | `run.py`                             | Prediction intervals per target day and horizon                                  |
| `outputs/by_oblast_latest_window.csv`                      | `run.py --by-oblast`                 | Per-oblast model comparison                                                      |
| `outputs/forecast_<oblast>_<time>.csv/.png`                | `predict.py`                         | The 24-hour outlook                                                              |
| `outputs/backtest_<oblast>.png/.txt`                       | `predict.py --backtest`              | Single-oblast accuracy                                                           |
| `outputs/backtest_all_oblasts.txt/.csv`                    | `predict.py --oblast all --backtest` | Accuracy for every oblast                                                        |
| `outputs/validation_report.md`, `validation_by_oblast.csv` | `validate.py`                        | Cross-source comparison                                                          |
| `data/*.csv`                                               | all                                  | Cached downloads (git-ignored)                                                   |

All timestamps in files are Kyiv local time unless the column name says UTC. Tables in `.md` and
`.txt` files are aligned: numbers right-aligned with a fixed number of decimals per column, text
left-aligned, missing values shown as `-`.

## 9. Troubleshooting

| Symptom                                                                | Cause and fix                                                                                                  |
| :--------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `The official dataset is ... h old (limit 96 h)`                       | The official file lags by weeks. Forecast with `--source volunteer`.                                           |
| `Data is N h old` warning                                              | The upstream stopped updating or the local cache is stale. Try `--update`, then `validate.py`.                 |
| `Could not obtain the 'volunteer' dataset and there is no local cache` | No network and no cache. Download the CSV manually to `data/` or set `ALERTS_VOLUNTEER_URL`.                   |
| `rejected <url>: missing columns`                                      | A mirror returned something that is not the dataset (an error page, a changed schema). The cache is untouched. |
| `Could not match oblast`                                               | Use `--list`. Crimea is not in the data and Luhansk is excluded (permanent siren).                             |
| `skill_vs_clim` shows `-`                                              | Nothing happened in the test window, so the benchmark's error is zero and skill is undefined.                  |
| Results differ between runs on different days                          | The dataset is updated daily, so the "latest window" moves.                                                    |

## 10. Using it from Python

```python
import sys; sys.path.insert(0, "src")
from ua_alerts import data, sources, forecast

res = sources.fetch("volunteer", "data")                 # cached, with mirror fallback
d = data.build_dataset(res.path)                         # d.daily, d.hourly_region, d.warnings ...
r = forecast.predict_next_24h(d, "Kharkiv")              # ForecastResult
print(r.slots.head())                                    # start_local, end_local, probability
print(r.p_any, r.expected_alert_hours, r.blend)

b = forecast.backtest(d, "Lviv", test_days=28)           # BacktestResult
print(b.hourly)                                          # the table shown by --backtest
```

Module map: `sources.py` (download, validation, quality checks, parsers), `data.py` (cleaning,
interval merging, hourly and daily aggregation), `forecast.py` (24-hour model and backtest),
`tables.py` (aligned tables), `eda.py`, `models.py`, `evaluate.py`, `report.py` (daily-load analysis).
