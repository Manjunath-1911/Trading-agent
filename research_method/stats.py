"""Overfitting statistics.

deflated_sharpe  Bailey & Lopez de Prado (2014): P(true Sharpe > 0) after allowing for having picked the best of
                 `n_trials` variants, with non-normal returns (skew, kurtosis) accounted for.
net_over_dd      the selection score used for strategy cells: total R / maximum drawdown in R.
walk_forward     the honest backtest: each test period is traded with the cell that scored best on ALL PRIOR periods
                 only; the stitched result is out-of-sample by construction.
"""
from __future__ import annotations

import math

import numpy as np


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _phi_inv(p: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):                                    # bisection is plenty for a CDF
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _phi(mid) < p else (lo, mid)
    return (lo + hi) / 2


def deflated_sharpe(x, n_trials: int) -> float:
    """Probability that the per-observation Sharpe ratio of `x` is genuinely > 0 given it was the best of `n_trials`."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 10 or x.std(ddof=1) == 0:
        return float("nan")
    sd = x.std(ddof=1)
    sr = x.mean() / sd
    z = (x - x.mean()) / sd
    skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    var_sr = max((1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (n - 1), 1e-12)
    g = 0.5772156649                                            # Euler-Mascheroni
    sr0 = math.sqrt(var_sr) * ((1 - g) * _phi_inv(1 - 1 / n_trials) + g * _phi_inv(1 - 1 / (n_trials * math.e)))
    return _phi((sr - sr0) / math.sqrt(var_sr))


def net_over_dd(x) -> float:
    x = np.asarray(x, float)
    if not len(x):
        return float("nan")
    eq = np.r_[0.0, np.cumsum(x)]
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    if dd > 0:
        return float(x.sum() / dd)
    return float("inf") if x.sum() > 0 else 0.0              # never lost: the best possible score, not "undefined"


def walk_forward(table: dict, periods: list, first_test: int, score=net_over_dd) -> tuple[np.ndarray, list]:
    """table: cell -> {period: array of per-trade R}. For every test period from index `first_test`, pick the cell with
    the best `score` on all EARLIER periods and take its result in that period. -> (stitched R, picks)."""
    out, picks = [], []
    for k in range(first_test, len(periods)):
        past = periods[:k]
        best = max(table, key=lambda c: np.nan_to_num(score(np.concatenate([table[c][p] for p in past])), nan=-1e9))
        picks.append((periods[k], best))
        out.append(table[best][periods[k]])
    return np.concatenate(out) if out else np.zeros(0), picks
