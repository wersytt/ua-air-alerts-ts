"""Experimental pooled model: one classifier for all oblasts at once.

Compared with the per-oblast model in :mod:`forecast`, this one

* trains on all oblasts stacked (24x more rows; the oblast id is a categorical feature),
* adds *neighbour* features (recent alert state of the bordering oblasts),
* optionally down-weights old training rows (exponential decay, half-life in days),
* chooses the model/climatology blend weight per oblast on several rolling validation windows, shrunk toward the
  all-oblast optimum, and
* optionally calibrates the blended probabilities with isotonic regression fitted on those validation windows.

Everything is computed from data available at the forecast origin, as in :mod:`forecast`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import forecast as F
from .data import AlertData

# Approximate land borders between the oblasts present in the data (Crimea and Luhansk are not in it).
# Kyiv City is enclosed by Kyivska oblast. Used only to build "state of the neighbours" features.
_ADJ = {
    "Volynska": ["Rivnenska", "Lvivska"],
    "Rivnenska": ["Volynska", "Lvivska", "Ternopilska", "Khmelnytska", "Zhytomyrska"],
    "Lvivska": ["Volynska", "Rivnenska", "Ternopilska", "Ivano-Frankivska", "Zakarpatska"],
    "Zakarpatska": ["Lvivska", "Ivano-Frankivska"],
    "Ivano-Frankivska": ["Zakarpatska", "Lvivska", "Ternopilska", "Chernivetska"],
    "Chernivetska": ["Ivano-Frankivska", "Ternopilska", "Khmelnytska", "Vinnytska"],
    "Ternopilska": ["Lvivska", "Rivnenska", "Khmelnytska", "Ivano-Frankivska", "Chernivetska"],
    "Khmelnytska": ["Rivnenska", "Ternopilska", "Chernivetska", "Vinnytska", "Zhytomyrska"],
    "Zhytomyrska": ["Rivnenska", "Khmelnytska", "Vinnytska", "Kyivska"],
    "Vinnytska": ["Zhytomyrska", "Khmelnytska", "Chernivetska", "Odeska", "Kirovohradska", "Cherkaska", "Kyivska"],
    "Kyivska": ["Zhytomyrska", "Vinnytska", "Cherkaska", "Poltavska", "Chernihivska", "Kyiv City"],
    "Chernihivska": ["Kyivska", "Poltavska", "Sumska"],
    "Sumska": ["Chernihivska", "Poltavska", "Kharkivska"],
    "Poltavska": ["Chernihivska", "Sumska", "Kharkivska", "Dnipropetrovska", "Kirovohradska", "Cherkaska", "Kyivska"],
    "Kharkivska": ["Sumska", "Poltavska", "Dnipropetrovska", "Donetska"],
    "Donetska": ["Kharkivska", "Dnipropetrovska", "Zaporizka"],
    "Dnipropetrovska": ["Poltavska", "Kharkivska", "Donetska", "Zaporizka", "Khersonska", "Mykolaivska", "Kirovohradska"],
    "Zaporizka": ["Donetska", "Dnipropetrovska", "Khersonska"],
    "Khersonska": ["Zaporizka", "Dnipropetrovska", "Mykolaivska"],
    "Mykolaivska": ["Khersonska", "Dnipropetrovska", "Kirovohradska", "Odeska"],
    "Kirovohradska": ["Cherkaska", "Poltavska", "Dnipropetrovska", "Mykolaivska", "Odeska", "Vinnytska"],
    "Odeska": ["Vinnytska", "Mykolaivska", "Kirovohradska"],
    "Cherkaska": ["Kyivska", "Vinnytska", "Kirovohradska", "Poltavska"],
    "Kyiv City": ["Kyivska"],
}
NBR_COLS = ["nbr0", "nbr_m6", "nbr_m24", "nbr_any"]
CONFIG = dict(learning_rate=0.06, max_iter=150, max_depth=5, min_samples_leaf=300, l2_regularization=10.0)


def neighbours(regions: list[str]) -> dict[str, list[str]]:
    key = lambda r: r.replace(" oblast", "")  # noqa: E731
    by_key = {key(r): r for r in regions}
    adj: dict[str, set[str]] = {r: set() for r in regions}
    for a, ns in _ADJ.items():
        for b in ns:
            if a in by_key and b in by_key:
                adj[by_key[a]].add(by_key[b]); adj[by_key[b]].add(by_key[a])
    return {r: sorted(v) for r, v in adj.items()}


@dataclass
class World:
    regions: list[str]
    P: dict                  # region -> Prepared
    nbr: dict                # region -> dict of arrays (cumsums) for neighbour features
    T: int


def build_world(data: AlertData, as_of: pd.Timestamp | None = None) -> World:
    regions = list(data.hourly_region.columns)
    P = {r: F.prepare(data, r, as_of) for r in regions}
    adj = neighbours(regions)
    occ = {r: P[r].occ for r in regions}
    nbr = {}
    z = np.zeros(1)
    for r in regions:
        ns = adj[r] or [r]
        m = np.mean([occ[n] for n in ns], axis=0)
        anyn = np.mean([P[n].a for n in ns], axis=0)
        nbr[r] = dict(m=m, cs_m=np.concatenate([z, np.cumsum(m)]), any=anyn)
    return World(regions, P, nbr, len(next(iter(P.values())).a))


def _nbr_features(W: World, r: str, t: np.ndarray) -> np.ndarray:
    d = W.nbr[r]
    return np.column_stack([d["m"][t], F._rmean(d["cs_m"], t, 6), F._rmean(d["cs_m"], t, 24), d["any"][t]])


def matrix(W: World, r: str, t: np.ndarray, s_values) -> tuple[np.ndarray, np.ndarray]:
    """Rows = (origin, offset). Columns: origin features, neighbour features, region id, s, local hour, weekday, clim."""
    P = W.P[r]
    rid = W.regions.index(r)
    base = np.column_stack([F.origin_features(P, t), _nbr_features(W, r, t), np.full(len(t), rid)])
    X, y = [], []
    for s in s_values:
        u = t + s
        clim = P.clim[u - 24 * int(np.ceil(s / 24))]
        X.append(np.column_stack([base, np.full(len(t), s), P.lh[u], P.dow[u], clim]))
        y.append(P.a[u] if u.max() < len(P.a) else np.full(len(t), np.nan))
    return np.vstack(X).astype(np.float32), np.concatenate(y)


def _cat_index() -> list[int]:
    return [len(F.ORIGIN_COLS) + len(NBR_COLS)]          # region id


def _stack(W: World, origins: dict, s_values):
    Xs, ys, rs = [], [], []
    for r, t in origins.items():
        X, y = matrix(W, r, t, s_values)
        Xs.append(X); ys.append(y); rs.append(np.full(len(y), W.regions.index(r)))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(rs)


def _fit(X, y, ages_days, half_life, seed=0):
    w = None if not half_life else 0.5 ** (ages_days / half_life)
    clf = HistGradientBoostingClassifier(**CONFIG, categorical_features=_cat_index(), random_state=seed)
    return clf.fit(X, y, sample_weight=w)


def _train_set(W: World, t_hi: int, train_days: int, stride: int, s_values):
    lo = max(F.MIN_ORIGIN, t_hi - train_days * 24)
    t = np.arange(lo, t_hi + 1, stride)
    X, y, _ = _stack(W, {r: t for r in W.regions}, s_values)
    ages = np.concatenate([np.tile((t_hi - t) / 24.0, len(s_values)) for _ in W.regions])   # matches _stack's row order
    return X, y, ages


def _fills(W: World, t_hi: int) -> np.ndarray:
    """Per-oblast fallback for a missing climatology value: that oblast's mean alert share up to ``t_hi``."""
    return np.array([W.P[r].a[: t_hi + 1].mean() for r in W.regions])


def _raw_predict(W, t_hi, t_pred, s_values, train_days, stride, half_life, seed=0):
    X, y, ages = _train_set(W, t_hi, train_days, stride, s_values)
    ok = ~np.isnan(y)
    model = _fit(X[ok], y[ok], ages[ok], half_life, seed)
    Xp, yp, rp = _stack(W, t_pred, s_values)
    clim = np.where(np.isnan(Xp[:, -1]), _fills(W, t_hi)[rp], Xp[:, -1])
    return model, model.predict_proba(Xp)[:, 1], yp, rp, clim, int(ok.sum())


class Platt:
    """Smooth monotone calibrator: logistic regression on logit(p). Keeps the ranking and the resolution of p."""

    def __init__(self):
        self.lr = LogisticRegression(C=1e6)

    @staticmethod
    def _z(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.lr.fit(self._z(p), y)
        return self

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


@dataclass
class PooledFit:
    model: object
    regions: list
    blend: np.ndarray            # per-oblast weight on the model (rest = climatology)
    blend_global: float
    iso: object | None           # isotonic calibrator fitted on validation predictions
    platt: object | None         # smooth logistic calibrator fitted on the same predictions
    calibration: str             # which calibrator predict() applies: 'platt', 'isotonic' or 'none'
    val_skill: np.ndarray        # per-oblast Brier skill of the final (blended, calibrated) validation forecasts
    n_train: int
    s_values: range
    t_hi: int
    half_life: float | None


def fit(W: World, t_hi: int, s_values=range(1, F.SPAN + 1), train_days: int = 365, stride: int = 6,
        half_life: float | None = 120, val_windows: int = 3, val_days: int = 14, calibrate: bool = True,
        seed: int = 0) -> PooledFit:
    """Fit the pooled model on origins <= ``t_hi``; choose blend weights and calibration on rolling validation windows.

    For each of ``val_windows`` consecutive windows of ``val_days`` days before ``t_hi``, a model is trained only on
    data before that window and its predictions are pooled. Per-oblast weights are then shrunk 50/50 toward the
    all-oblast optimum, and an isotonic calibrator is fitted on the pooled blended validation forecasts.
    """
    R = len(W.regions)
    win = val_days * 24
    VP, VY, VR, VC = [], [], [], []
    for k in range(val_windows):
        v_hi = t_hi - k * win
        v_lo = v_hi - win + 1
        tv = np.arange(v_lo, v_hi + 1, 2)
        _, pv, yv, rv, cv, _ = _raw_predict(W, v_lo - 1 - max(s_values), {r: tv for r in W.regions}, s_values,
                                            train_days, stride, half_life, seed)
        VP.append(pv); VY.append(yv); VR.append(rv); VC.append(cv)
    VP, VY, VR, VC = map(np.concatenate, (VP, VY, VR, VC))
    grid = np.linspace(0, 1, 11)

    def best_w(mask):
        return grid[int(np.argmin([np.mean((w * VP[mask] + (1 - w) * VC[mask] - VY[mask]) ** 2) for w in grid]))]

    wg = float(best_w(np.ones(len(VP), bool)))
    blend = 0.5 * np.array([best_w(VR == i) for i in range(R)]) + 0.5 * wg
    vb = blend[VR] * VP + (1 - blend[VR]) * VC
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(vb, VY)
    platt = Platt().fit(vb, VY)
    vf = platt.predict(vb) if calibrate else vb
    val_skill = np.array([1 - np.mean((vf[VR == i] - VY[VR == i]) ** 2) / max(np.mean((VC[VR == i] - VY[VR == i]) ** 2), 1e-12)
                          for i in range(R)])
    X, y, ages = _train_set(W, t_hi, train_days, stride, s_values)
    ok = ~np.isnan(y)
    model = _fit(X[ok], y[ok], ages[ok], half_life, seed)
    return PooledFit(model, list(W.regions), blend, wg, iso, platt, "platt" if calibrate else "none", val_skill,
                     int(ok.sum()), s_values, t_hi, half_life)


def predict(W: World, f: PooledFit, t_pred: dict, s_values=None, calibration: str | None = None):
    """Final probabilities for rows (region-major, then offset, then origin). Returns (p_final, p_raw, y, region_ids, clim)."""
    s_values = s_values or f.s_values
    Xp, yp, rp = _stack(W, t_pred, s_values)
    clim = np.where(np.isnan(Xp[:, -1]), _fills(W, f.t_hi)[rp], Xp[:, -1])
    raw = f.model.predict_proba(Xp)[:, 1]
    p = f.blend[rp] * raw + (1 - f.blend[rp]) * clim
    cal = calibration or f.calibration
    if cal == "platt":
        p = f.platt.predict(p)
    elif cal == "isotonic":
        p = f.iso.predict(p)
    return p, raw, yp, rp, clim


def _window_model(P, t_hi, train_days, seed=0):
    """Per-oblast classifier for 'any alert in the whole 24 h window' (unchanged from the per-oblast model)."""
    t = F._train_origins(len(P.a), F.SPAN, train_days, 2, t_hi)
    Xw, yw = F.window_matrix(P, t, 0)
    return F._fit_clf(Xw, yw, seed)


def backtest_all(data: AlertData, test_days: int = 28, train_days: int = 365, half_life: float | None = 120,
                 calibrate: bool = True, seed: int = 0, as_of: pd.Timestamp | None = None) -> dict:
    """Pooled backtest on the latest ``test_days`` days for every oblast at once. Returns {region: BacktestResult}.

    Same protocol as :func:`forecast.backtest`: trained only on data before the test window, every hour of the
    window is a forecast origin, forecasts for the next 24 hourly slots.
    """
    W = build_world(data, as_of)
    T = W.T
    t_first = T - 1 - F.SPAN - test_days * 24 + 1
    t_test = np.arange(t_first, T - F.SPAN)
    t_hi = t_first - 1 - F.SPAN
    f = fit(W, t_hi, range(1, F.SPAN + 1), train_days, stride=6, half_life=half_life,
            calibrate=calibrate, seed=seed)
    p, raw, y, rid, clim = predict(W, f, {r: t_test for r in W.regions})
    n = len(t_test)
    out = {}
    for i, r in enumerate(W.regions):
        m = rid == i
        P = W.P[r]
        horizon = np.tile(np.repeat(np.arange(1, F.SPAN + 1), n), 1)
        Xr, _ = matrix(W, r, t_test, range(1, F.SPAN + 1))
        recent = Xr[:, F.ORIGIN_COLS.index("occ_m168")]
        lh = Xr[:, len(F.ORIGIN_COLS) + len(NBR_COLS) + 2].astype(int)
        base = float(np.concatenate([F.hourly_matrix(P, F._train_origins(T, F.SPAN, train_days, 6, t_hi), range(1, F.SPAN + 1))[1]]).mean())
        out[r] = F.assemble_backtest(P, r, t_test, t_hi, train_days, y[m], horizon, raw[m], p[m], clim[m], recent, lh, base,
                                     _window_model(P, t_hi, train_days, seed), float(f.blend[i]), float(f.val_skill[i]))
    return out


def window_model_for_forecast(P, s_min: int, train_days: int, seed: int = 0):
    """'Any alert in the 24 h window' classifier for a forecast whose first slot is ``s_min`` hours after the origin."""
    s_max = s_min + F.SPAN - 1
    t = F._train_origins(len(P.a), s_max, train_days, 2, None)
    Xw, yw = F.window_matrix(P, t, s_min - 1)
    return F._fit_clf(Xw, yw, seed)


def forecast_slots(W: World, region: str, s_min: int, train_days: int = 365, half_life: float | None = 120,
                   seed: int = 0):
    """Probabilities for the 24 slots s_min..s_min+23 hours after the newest complete hour. Returns (p, PooledFit)."""
    s_values = range(s_min, s_min + F.SPAN)
    t_hi = W.T - 1 - max(s_values)
    f = fit(W, t_hi, s_values, train_days, stride=6, half_life=half_life, seed=seed)
    p, *_ = predict(W, f, {region: np.array([W.T - 1])}, s_values)
    return p, f
