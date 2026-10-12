"""Per-oblast daily models and a bottom-up national forecast.

Each oblast gets its own daily alert-hours series (0-24 h/day) and the cheap models are backtested on it with
exactly the same rolling-origin protocol as the national series. Summing the per-oblast forecasts gives a
*bottom-up* national forecast that can be compared with forecasting the national total directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import models as M
from .data import LOCAL_TZ


def oblast_daily(hourly_region: pd.DataFrame, last_complete_day: pd.Timestamp) -> pd.DataFrame:
    """Alert-hours per local day, one column per oblast (hourly_region is UTC-indexed occupancy in hours)."""
    h = hourly_region.copy()
    h.index = h.index.tz_convert(LOCAL_TZ).tz_localize(None)
    d = h.resample("D").sum()
    return d.loc[:last_complete_day]


def fast_models() -> list[M.Forecaster]:
    return [M.Naive(), M.SeasonalNaive7(), M.MovingAverage(7), M.MovingAverage(28), M.SeasonalMean4w(),
            M.RegimeSeasonalMean(), M.ETS(seasonal=True)]


def run(daily: pd.DataFrame, national: pd.Series, test_start, test_end, horizon: int = 7, verbose=print):
    """Backtest the fast models on every oblast. Returns (per_oblast_table, national_table, per_oblast_preds)."""
    all_preds = {}
    for i, region in enumerate(daily.columns, 1):
        if verbose:
            verbose(f"  by-oblast {i}/{daily.shape[1]}: {region}")
        all_preds[region] = ev.backtest(daily[region], fast_models, test_start, test_end, horizon, verbose=False)

    rows = []
    for region, p in all_preds.items():
        mae = p.assign(ae=p["err"].abs()).groupby("model")["ae"].mean()
        base = mae[mae.index.isin(ev.BASELINES)]
        non = mae[~mae.index.isin(ev.BASELINES)]
        best_base = base.idxmin()
        best = mae.idxmin()
        rows.append({"oblast": region, "mean_h_per_day": float(daily[region].loc[test_start:test_end].mean()),
                     "best_baseline": best_base, "MAE_best_baseline": base.min(),
                     "best_model": best, "MAE_best": mae.min(),
                     "gain_vs_baseline": 1 - mae.min() / base.min() if base.min() > 0 else np.nan,
                     "MAE_best_non_baseline": non.min() if len(non) else np.nan})
    per = pd.DataFrame(rows).set_index("oblast").sort_values("mean_h_per_day", ascending=False)

    # bottom-up national forecast = sum of per-oblast forecasts of the same model, scored against the national total
    cat = pd.concat(all_preds, names=["oblast"]).reset_index(level=0)
    bu = cat.groupby(["model", "origin", "target", "h"])[["y_pred"]].sum().reset_index()
    nat = national
    bu["y_true"] = nat.reindex(bu["target"]).to_numpy()
    bu["err"] = bu["y_pred"] - bu["y_true"]
    bu_mae = bu.assign(ae=bu["err"].abs()).groupby("model")["ae"].mean()
    td = ev.backtest(national, fast_models, test_start, test_end, horizon, verbose=False)
    td_mae = td.assign(ae=td["err"].abs()).groupby("model")["ae"].mean()
    nat_tbl = pd.DataFrame({"MAE_top_down": td_mae, "MAE_bottom_up": bu_mae})
    nat_tbl["bottom_up_vs_top_down"] = nat_tbl["MAE_bottom_up"] / nat_tbl["MAE_top_down"] - 1
    return per, nat_tbl.sort_values("MAE_top_down"), all_preds
