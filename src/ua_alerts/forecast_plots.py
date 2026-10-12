"""Figures for the 24-hour outlook and its backtest."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .forecast import BacktestResult, ForecastResult


def plot_forecast(r: ForecastResult, path: Path) -> Path:
    s = r.slots
    labels = [f"{t:%H}" for t in s["start_local"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(s)), s["probability"], color="#c0392b", alpha=0.85)
    ax.set_xticks(range(len(s)), labels)
    ax.set_ylim(0, min(1.0, max(0.35, s["probability"].max() * 1.25)))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(f"hour starting at (Kyiv time); first slot = {s['start_local'].iloc[0]:%a %d %b %H:00}")
    ax.set_ylabel("P(alert during that hour)")
    ax.set_title(f"{r.oblast}: chance of an air raid alert in each of the next 24 hours "
                 f"(statistical estimate, data {r.data_age_hours:.0f} h old)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_backtest(b: BacktestResult, path: Path) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    rel = b.reliability
    ax[0].plot([0, 1], [0, 1], color="#999", lw=1, label="perfect calibration")
    ax[0].scatter(rel["mean_predicted"], rel["observed_rate"], s=np.clip(rel["n"] / rel["n"].max() * 220, 12, 220),
                  color="#2c3e50", label="bins (size = #predictions)")
    ax[0].set_xlabel("predicted probability")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title("Calibration (reliability) on the test window")
    ax[0].legend(frameon=False, fontsize=8)
    h = b.by_hour
    ax[1].plot(h.index, h["obs"], color="black", lw=1.6, label="observed share of hours with alert")
    ax[1].plot(h.index, h["pred"], color="#c0392b", lw=1.6, label="mean predicted probability")
    ax[1].set_xlabel("target hour of day (Kyiv time)")
    ax[1].set_title("By hour of day")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{b.oblast} - backtest {b.test_start:%Y-%m-%d} to {b.test_end:%Y-%m-%d} (UTC), "
                 f"{b.n_predictions:,} hourly predictions", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
