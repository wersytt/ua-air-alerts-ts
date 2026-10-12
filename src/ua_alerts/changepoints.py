"""Change-point detection for the daily alert-hours series (mean shifts).

Optimal partitioning (exact dynamic programming, the same objective PELT optimises) with a squared-error
cost, a minimum segment length and a BIC-style penalty. Only the *past* is needed to detect a change, but a
change can only be confirmed ``min_size`` days after it happened, which is why the regime-aware forecaster
never reacts to a shift in the last few days.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _noise_sigma(x: np.ndarray) -> float:
    """Robust day-to-day noise level: MAD of first differences / sqrt(2)."""
    d = np.diff(x)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) or float(np.std(x)) or 1.0


def detect(y, min_size: int = 21, penalty_scale: float = 3.0) -> list[int]:
    """Return indices ``i`` (0 < i < n) where a new regime starts, i.e. the mean changes between i-1 and i.

    ``penalty_scale`` multiplies sigma^2 * log(n). Daily alert-hours are autocorrelated, so the default is
    deliberately larger than the textbook BIC value: it favours fewer, larger changes.
    """
    x = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        return []
    beta = penalty_scale * _noise_sigma(x) ** 2 * np.log(n)
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def cost(s: np.ndarray, t: int) -> np.ndarray:
        m = t - s
        seg = c1[t] - c1[s]
        return (c2[t] - c2[s]) - seg * seg / m

    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, dtype=int)
    for t in range(min_size, n + 1):
        s = np.arange(0, t - min_size + 1)
        s = s[np.isfinite(F[s])]
        if len(s) == 0:
            continue
        tot = F[s] + cost(s, t) + beta
        j = int(np.argmin(tot))
        F[t], last[t] = tot[j], s[j]
    cps, t = [], n
    while t > 0 and np.isfinite(F[t]):
        s = last[t]
        if s > 0:
            cps.append(int(s))
        t = s
    return sorted(cps)


def segments(y: pd.Series, **kw) -> pd.DataFrame:
    """One row per regime: start, end, days, mean level (alert-hours/day)."""
    cps = detect(y.to_numpy(), **kw)
    edges = [0, *cps, len(y)]
    rows = [dict(start=y.index[a], end=y.index[b - 1], days=b - a, mean=float(y.iloc[a:b].mean()))
            for a, b in zip(edges[:-1], edges[1:])]
    out = pd.DataFrame(rows)
    out["change_vs_previous"] = out["mean"] / out["mean"].shift(1) - 1
    return out


def plot_segments(y: pd.Series, seg: pd.DataFrame, outdir: Path) -> str:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.index, y, lw=0.5, color="#8aa", label="daily")
    for k, r in enumerate(seg.itertuples()):
        ax.hlines(r.mean, r.start, r.end, color="#c0392b", lw=2.2, label="regime mean" if k == 0 else None)
        if k:
            ax.axvline(r.start, color="#f39c12", lw=0.8, ls="--")
    ax.set_ylabel("oblast-hours under alert / day")
    ax.set_title("Detected regimes (mean-shift change points)")
    ax.legend(frameon=False, loc="upper left")
    name = "09_changepoints.png"
    fig.tight_layout()
    fig.savefig(outdir / name)
    plt.close(fig)
    return name
