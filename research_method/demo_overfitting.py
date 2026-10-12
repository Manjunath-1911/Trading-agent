"""Why a backtest's best variant always looks better than it is — a self-contained demo on synthetic data.

500 "strategy variants" that are pure noise, plus a family of 20 variants sharing one small real edge (+0.05 R/trade),
each trading ~150 times a year for 6 years. Pick the variant with the best full-history result (what an in-sample
optimisation does), then compare with a walk-forward that re-picks each year from prior years only.

Run: python -m research_method.demo_overfitting
"""
from __future__ import annotations

import numpy as np

from research_method.stats import deflated_sharpe, net_over_dd, walk_forward

YEARS = list(range(2020, 2026))


def make_table(seed: int = 7, n_noise: int = 500, n_real: int = 20, edge: float = 0.05, per_year: int = 150) -> dict:
    rng = np.random.default_rng(seed)
    table = {}
    for i in range(n_noise + n_real):
        mu = edge if i >= n_noise else 0.0
        # 2R winners / 1R losers with ~1/3 win rate: realistic fat-ish trade distribution, mean shifted by `mu`
        table[("real" if mu else "noise", i)] = {
            y: np.where(rng.random(per_year) < 1 / 3 + mu / 3, 2.0, -1.0) for y in YEARS}
    return table


def main() -> None:
    table = make_table()
    full = {c: np.concatenate([v[y] for y in YEARS]) for c, v in table.items()}
    best = max(full, key=lambda c: net_over_dd(full[c]))
    wf, picks = walk_forward(table, YEARS, first_test=2)
    in_sample_same_years = np.concatenate([table[best][y] for y in YEARS[2:]])
    real = [c for c in table if c[0] == "real"]
    print("variants: %d (500 noise, 20 with a real +0.05 R/trade edge)" % len(table))
    print("in-sample best:   %-14s R/trade %+.3f   (2022-25: %+.3f)" % (best, full[best].mean(), in_sample_same_years.mean()))
    print("walk-forward:     %-14s R/trade %+.3f   picks: %s" % ("re-picked yearly", wf.mean(),
                                                                 ", ".join("%d %s" % (y, c[0]) for y, c in picks)))
    print("true edge of the real family: +0.050 R/trade; mean of the real family's backtests: %+.3f"
          % np.mean([full[c].mean() for c in real]))
    print("deflated Sharpe of the in-sample best, N=%d trials: %.2f  (naive N=1: %.2f)"
          % (len(table), deflated_sharpe(full[best], len(table)), deflated_sharpe(full[best], 1)))


if __name__ == "__main__":
    main()
