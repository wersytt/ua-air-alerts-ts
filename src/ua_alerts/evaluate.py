"""Rolling-origin backtesting and scoring.

For every day `o` (the forecast origin) inside / just before the test window we hand each
model only ``y[:o]`` and ask for the next 7 days. Predictions are kept for target days that
fall inside the test window, so every model is scored on exactly the same days.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import Forecaster

BASELINES = {"Naive", "SeasonalNaive7", "MovingAvg7", "MovingAvg28", "SeasonalMean4w"}


def backtest(
    y: pd.Series,
    make_models: Callable[[], list[Forecaster]],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon: int = 7,
    verbose: bool = True,
) -> pd.DataFrame:
    models = make_models()
    origins = pd.date_range(test_start - pd.Timedelta(days=horizon), test_end - pd.Timedelta(days=1))
    rows = []
    for k, origin in enumerate(origins):
        history = y.loc[:origin]  # <- the only data any model is allowed to see
        for m in models:
            if k % m.refit_every == 0:
                m.fit(history)
            fc = m.predict(history, horizon)
            for h in range(1, horizon + 1):
                target = origin + pd.Timedelta(days=h)
                if test_start <= target <= test_end:
                    rows.append((m.name, origin, target, h, y.loc[target], float(fc[h - 1])))
        if verbose and (k + 1) % 10 == 0:
            print(f"    origin {k + 1}/{len(origins)}")
    out = pd.DataFrame(rows, columns=["model", "origin", "target", "h", "y_true", "y_pred"])
    out["err"] = out["y_pred"] - out["y_true"]
    return out


def naive_scale(y: pd.Series, test_start: pd.Timestamp, window: int = 365) -> float:
    """In-sample MAE of the one-step naive forecast (denominator of MASE)."""
    train = y.loc[: test_start - pd.Timedelta(days=1)].iloc[-window:]
    return float(train.diff().abs().mean())


def score(preds: pd.DataFrame, scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-horizon metrics, one-row-per-model summary)."""
    def agg(g):
        return pd.Series({
            "MAE": g["err"].abs().mean(),
            "RMSE": np.sqrt((g["err"] ** 2).mean()),
            "bias": g["err"].mean(),
        })

    per_h = preds.groupby(["model", "h"]).apply(agg, include_groups=False)
    per_h["MASE"] = per_h["MAE"] / scale

    overall = preds.groupby("model").apply(agg, include_groups=False)
    overall["MASE"] = overall["MAE"] / scale
    summary = pd.DataFrame({
        "MAE_h1": per_h["MAE"].xs(1, level="h"),
        "MAE_h7": per_h["MAE"].xs(7, level="h"),
        "MAE_1to7": overall["MAE"],
        "RMSE_1to7": overall["RMSE"],
        "bias_1to7": overall["bias"],
        "MASE_1to7": overall["MASE"],
    })
    return per_h, summary.sort_values("MAE_1to7")


def block_bootstrap_ci(d: np.ndarray, block: int = 7, n: int = 2000, seed: int = 0):
    """95% moving-block bootstrap CI for mean(d); blocks keep day-to-day dependence."""
    rng = np.random.default_rng(seed)
    n_obs = len(d)
    block = min(block, n_obs)
    n_blocks = int(np.ceil(n_obs / block))
    starts = rng.integers(0, n_obs - block + 1, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :n_obs]
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare_to_best_baseline(preds: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Paired MAE difference of each non-baseline model vs. the best baseline (in hindsight,
    which is the *conservative* choice), pooled over horizons 1-7, with a block-bootstrap CI."""
    base_summary = summary.loc[summary.index.isin(BASELINES)]
    ref = base_summary["MAE_1to7"].idxmin()
    pivot = preds.assign(ae=preds["err"].abs()).pivot_table(
        index=["target", "h"], columns="model", values="ae")
    rows = []
    for m in summary.index:
        if m == ref:
            continue
        d = (pivot[m] - pivot[ref]).groupby("target").mean().to_numpy()  # avg over h per target day
        lo, hi = block_bootstrap_ci(d)
        rows.append({"model": m, "reference": ref, "dMAE": d.mean(), "ci_low": lo, "ci_high": hi,
                     "significant": (lo > 0) or (hi < 0)})
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- plotting
def plot_forecasts(preds, y, h, test_start, test_end, summary, outdir: Path) -> str:
    show = [m for m in summary.index if m not in BASELINES][:2]
    best_base = summary.loc[summary.index.isin(BASELINES)].index[0]
    show = [best_base] + show
    ctx = y.loc[test_start - pd.Timedelta(days=21): test_end]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ctx.index, ctx, color="black", lw=1.6, label="actual")
    colors = ["#7f8c8d", "#c0392b", "#2980b9"]
    for c, m in zip(colors, show):
        p = preds[(preds.model == m) & (preds.h == h)].set_index("target")["y_pred"]
        ax.plot(p.index, p, lw=1.4, color=c, ls="--" if m in BASELINES else "-", label=m)
    ax.axvline(test_start, color="#f39c12", lw=1)
    ax.set_ylabel("oblast-hours under alert / day")
    ax.set_title(f"Backtest on the latest {len(y.loc[test_start:test_end])} complete days, "
                 f"horizon h={h} day{'s' if h > 1 else ''} ahead")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    name = f"06_forecast_h{h}.png"
    fig.tight_layout()
    fig.savefig(outdir / name)
    plt.close(fig)
    return name


def plot_mae_by_horizon(per_h: pd.DataFrame, outdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, g in per_h["MAE"].groupby(level="model"):
        ax.plot(g.index.get_level_values("h"), g.values, marker="o", ms=3,
                ls="--" if m in BASELINES else "-", label=m)
    ax.set_xlabel("forecast horizon (days)")
    ax.set_ylabel("MAE (oblast-hours / day)")
    ax.set_title("Forecast error by horizon, latest test window")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    name = "07_mae_by_horizon.png"
    fig.tight_layout()
    fig.savefig(outdir / name)
    plt.close(fig)
    return name


def plot_error_dist(preds, best_model: str, outdir: Path) -> str:
    p = preds[(preds.model == best_model) & (preds.h == 1)].set_index("target")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].bar(p.index, p["err"], color=np.where(p["err"] > 0, "#c0392b", "#2980b9"))
    ax[0].set_title(f"{best_model}: 1-day-ahead error (pred - actual)")
    ax[0].set_ylabel("oblast-hours")
    ax[1].scatter(p["y_true"], p["y_pred"], s=14, color="#2c3e50")
    lim = [0, max(p["y_true"].max(), p["y_pred"].max()) * 1.05]
    ax[1].plot(lim, lim, color="#999", lw=1)
    ax[1].set_xlabel("actual")
    ax[1].set_ylabel("predicted")
    ax[1].set_title("Predicted vs actual")
    name = "08_error_diagnostics.png"
    fig.tight_layout()
    fig.savefig(outdir / name)
    plt.close(fig)
    return name


# ------------------------------------------------------------------ prediction intervals
def interval_backtest(
    y: pd.Series,
    make_models: Callable[[], list[Forecaster]],
    model_name: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    horizon: int = 7,
    warmup: int = 42,
    levels: tuple[float, ...] = (0.5, 0.8),
    min_residuals: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical prediction intervals for one model, scored on the test window.

    For a forecast made at origin ``o`` for horizon ``h``, the interval is the point forecast minus the
    empirical quantiles of that model's *past* errors at the same horizon - only errors whose target day is
    already known at ``o`` (target <= o) are used, over a trailing window of ``warmup`` days. The model is run
    ``warmup`` days before the test window to have errors to start from. Returns (rows, summary).
    """
    only = lambda: [m for m in make_models() if m.name == model_name]  # noqa: E731
    p = backtest(y, only, test_start - pd.Timedelta(days=warmup), test_end, horizon, verbose=False)
    p = p.sort_values(["h", "origin"]).reset_index(drop=True)
    rows = []
    for h, g in p.groupby("h"):
        g = g.sort_values("target")
        for r in g[(g["target"] >= test_start) & (g["target"] <= test_end)].itertuples():
            known = g[(g["target"] <= r.origin) & (g["target"] > r.origin - pd.Timedelta(days=warmup))]["err"].to_numpy()
            if len(known) < min_residuals:
                continue
            rec = dict(target=r.target, origin=r.origin, h=h, y_true=r.y_true, y_pred=r.y_pred)
            for lv in levels:
                lo_q, hi_q = np.quantile(known, [(1 - lv) / 2, 1 - (1 - lv) / 2])
                rec[f"lo{int(lv * 100)}"] = max(r.y_pred - hi_q, 0.0)
                rec[f"hi{int(lv * 100)}"] = r.y_pred - lo_q
            rows.append(rec)
    out = pd.DataFrame(rows)
    summ = []
    for lv in levels:
        k = int(lv * 100)
        inside = (out["y_true"] >= out[f"lo{k}"]) & (out["y_true"] <= out[f"hi{k}"])
        width = out[f"hi{k}"] - out[f"lo{k}"]
        a = 1 - lv                      # interval (Winkler) score: width + penalty for misses
        wink = width + (2 / a) * (out[f"lo{k}"] - out["y_true"]).clip(lower=0) + (2 / a) * (out["y_true"] - out[f"hi{k}"]).clip(lower=0)
        summ.append({"nominal": lv, "coverage": float(inside.mean()), "mean_width": float(width.mean()),
                     "winkler_score": float(wink.mean()), "n": int(len(out))})
    return out, pd.DataFrame(summ).set_index("nominal")


def plot_intervals(rows: pd.DataFrame, model_name: str, test_start, test_end, outdir: Path, h: int = 1) -> str:
    g = rows[rows["h"] == h].set_index("target")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(g.index, g["lo80"], g["hi80"], color="#2980b9", alpha=0.18, label="80% interval")
    ax.fill_between(g.index, g["lo50"], g["hi50"], color="#2980b9", alpha=0.35, label="50% interval")
    ax.plot(g.index, g["y_pred"], color="#2980b9", lw=1.4, label=f"{model_name} forecast")
    ax.plot(g.index, g["y_true"], color="black", lw=1.4, label="actual")
    ax.set_ylabel("oblast-hours under alert / day")
    ax.set_title(f"Empirical prediction intervals, horizon h={h} day ahead (latest test window)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    name = "10_intervals.png"
    fig.tight_layout()
    fig.savefig(outdir / name)
    plt.close(fig)
    return name
