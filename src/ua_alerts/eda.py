"""Exploratory time series analysis: trend, seasonality, stationarity, regional split."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

from .data import LOCAL_TZ, AlertData

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, outdir: Path, name: str) -> str:
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return name


def plot_daily(daily: pd.DataFrame, test_start: pd.Timestamp, outdir: Path) -> str:
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    y = daily["alert_hours"]
    ax[0].plot(y.index, y, lw=0.5, color="#8aa", label="daily")
    ax[0].plot(y.index, y.rolling(28, center=False).mean(), lw=1.8, color="#c0392b",
               label="28-day rolling mean")
    ax[0].axvspan(test_start, y.index[-1], color="#f1c40f", alpha=0.25, label="latest test window")
    ax[0].set_ylabel("oblast-hours under alert / day")
    ax[0].set_title("Daily air raid alert load in Ukraine (24 oblasts, Kyiv local time)")
    ax[0].legend(loc="upper left", ncol=3, frameon=False)

    s = daily["starts"]
    ax[1].plot(s.index, s.rolling(28).mean(), color="#2c3e50", lw=1.6, label="alert starts / day")
    mean_dur = (y.rolling(28).sum() / s.rolling(28).sum())
    ax2 = ax[1].twinx()
    ax2.plot(mean_dur.index, mean_dur, color="#27ae60", lw=1.6, label="avg oblast-hours per alert")
    ax2.grid(False)
    ax[1].set_ylabel("starts / day (28d mean)")
    ax2.set_ylabel("avg alert length, h (28d)")
    ax[1].set_title("Alerts are getting more numerous but shorter: use hours, not counts, as the target")
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    ax[1].xaxis.set_major_locator(mdates.YearLocator())
    return _save(fig, outdir, "01_daily_series.png")


def plot_stl(daily: pd.DataFrame, outdir: Path) -> tuple[str, dict]:
    res = STL(daily["alert_hours"], period=7, seasonal=13, robust=True).fit()
    fig = res.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle("STL decomposition (weekly seasonality, robust)")
    var = res.resid.var()
    strength_season = max(0.0, 1 - var / (res.seasonal + res.resid).var())
    strength_trend = max(0.0, 1 - var / (res.trend + res.resid).var())
    return _save(fig, outdir, "02_stl.png"), {
        "seasonal_strength": float(strength_season),
        "trend_strength": float(strength_trend),
    }


def plot_hour_weekday(data: AlertData, days: int, outdir: Path) -> tuple[str, pd.DataFrame]:
    end = data.last_complete_day + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    h = data.hourly.tz_convert(LOCAL_TZ)
    h = h[(h.index.tz_localize(None) >= start) & (h.index.tz_localize(None) < end)]
    table = h.groupby([h.index.dayofweek, h.index.hour]).mean().unstack()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(table.values, aspect="auto", cmap="magma_r")
    ax.set_yticks(range(7), WEEKDAYS)
    ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
    ax.set_xlabel("hour of day (Kyiv local time)")
    ax.set_title(f"Average number of oblasts under alert, last {days} days")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="oblasts under alert")
    return _save(fig, outdir, "03_hour_weekday_heatmap.png"), table


def plot_acf_pacf(daily: pd.DataFrame, outdir: Path, window: int = 365) -> str:
    y = daily["alert_hours"].iloc[-window:]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    plot_acf(y, lags=35, ax=ax[0], title=f"ACF, last {window} days (levels)")
    plot_acf(y.diff().dropna(), lags=35, ax=ax[1], title="ACF, first differences")
    return _save(fig, outdir, "04_acf.png")


def plot_regions(data: AlertData, days: int, outdir: Path) -> tuple[str, pd.Series]:
    end = data.last_complete_day + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    idx = data.hourly_region.index.tz_convert(LOCAL_TZ).tz_localize(None)
    mask = (idx >= start) & (idx < end)
    per_day = data.hourly_region[mask].sum() / days
    per_day = per_day.sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(per_day.index.str.replace(" oblast", "", regex=False), per_day.values, color="#34495e")
    ax.set_xlabel("hours under alert per day")
    ax.set_title(f"Alert time by region, last {days} days")
    return _save(fig, outdir, "05_regions.png"), per_day.sort_values(ascending=False)


def stationarity(y: pd.Series) -> dict:
    adf_p = adfuller(y, autolag="AIC")[1]
    kpss_p = kpss(y, regression="c", nlags="auto")[1]
    return {"adf_p": float(adf_p), "kpss_p": float(kpss_p)}


def yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    g = daily.groupby(daily.index.year)
    out = pd.DataFrame({
        "days": g.size(),
        "alert_hours_per_day": g["alert_hours"].mean(),
        "starts_per_day": g["starts"].mean(),
    })
    out["hours_per_alert"] = out["alert_hours_per_day"] / out["starts_per_day"]
    return out.round(2)


def run_eda(data: AlertData, test_start: pd.Timestamp, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    daily = data.daily
    figs = [plot_daily(daily, test_start, outdir)]
    name, stl_stats = plot_stl(daily, outdir)
    figs.append(name)
    name, heat = plot_hour_weekday(data, 180, outdir)
    figs.append(name)
    figs.append(plot_acf_pacf(daily, outdir))
    name, regions = plot_regions(data, 90, outdir)
    figs.append(name)

    y365 = daily["alert_hours"].iloc[-365:]
    weekday_profile = (daily["alert_hours"].iloc[-182:].groupby(daily.index[-182:].dayofweek).mean())
    peak_hours = heat.mean(axis=0)
    return {
        "figures": figs,
        "stl": stl_stats,
        "stationarity_levels_365d": stationarity(y365),
        "stationarity_diff_365d": stationarity(y365.diff().dropna()),
        "yearly": yearly_summary(daily),
        "regions_top": regions.head(5),
        "weekday_profile": weekday_profile.rename(dict(enumerate(WEEKDAYS))).round(1),
        "hour_peak": int(peak_hours.idxmax()),
        "hour_trough": int(peak_hours.idxmin()),
    }
