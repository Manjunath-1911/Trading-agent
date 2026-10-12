import time
from datetime import datetime, timezone

import numpy as np

from research_method.demo_overfitting import make_table
from research_method.stats import deflated_sharpe, net_over_dd, walk_forward
from tradebot.mt5_clock import ServerClock, model_offset


def ts(*a):
    return datetime(*a, tzinfo=timezone.utc).timestamp()


def test_new_york_close_server_is_utc_plus_3_in_summer_and_2_in_winter():
    assert model_offset(ts(2025, 7, 1, 12)) == 3 * 3600
    assert model_offset(ts(2025, 12, 1, 12)) == 2 * 3600


class Tick:
    def __init__(self, t):
        self.time = t


class FakeMT5:
    def __init__(self, offset_h):
        self.off = offset_h * 3600

    def symbol_info_tick(self, s):
        return Tick(time.time() + self.off)


def test_offset_measured_from_a_fresh_tick_and_history_converted_per_bar():
    c = ServerClock(FakeMT5(round(model_offset(time.time()) / 3600)))
    assert c.offset() == model_offset(time.time()) and c.source.startswith("tick:")
    summer, winter = ts(2025, 7, 1, 12), ts(2025, 12, 1, 12)
    srv = np.array([summer + 3 * 3600, winter + 2 * 3600], dtype=np.int64)
    assert list(c.to_utc(srv)) == [int(summer), int(winter)]          # each bar with the offset in force THEN


def test_stale_tick_falls_back_to_the_env_offset():
    class Stale(FakeMT5):
        def symbol_info_tick(self, s):
            return Tick(time.time() - 3 * 86400)                          # Friday's tick on a Monday
    c = ServerClock(Stale(0), fallback_hours=2)
    assert c.offset() == 2 * 3600 and c.source == "env"


def test_walk_forward_only_ever_looks_back():
    table = {"a": {1: np.array([5.0]), 2: np.array([-9.0]), 3: np.array([1.0])},
             "b": {1: np.array([1.0, -0.5]), 2: np.array([3.0]), 3: np.array([2.0])}}
    r, picks = walk_forward(table, [1, 2, 3], first_test=1)
    assert picks == [(2, "a"), (3, "b")] and list(r) == [-9.0, 2.0]   # 'a' looked best on period 1 alone


def test_deflated_sharpe_falls_with_the_number_of_trials():
    x = np.random.default_rng(1).normal(0.1, 1.0, 400)
    assert deflated_sharpe(x, 1) > deflated_sharpe(x, 100) > deflated_sharpe(x, 10_000)
    assert net_over_dd([1, -2, 3]) == 1.0


def test_demo_best_backtest_overstates_its_edge():
    table = make_table()                                              # the README's numbers (seed 7)
    full = {c: np.concatenate(list(v.values())) for c, v in table.items()}
    best = max(full, key=lambda c: net_over_dd(full[c]))
    wf, _ = walk_forward(table, sorted(next(iter(table.values()))), first_test=2)
    assert best[0] == "noise"                                         # the in-sample winner has NO edge at all...
    assert full[best].mean() > 3 * wf.mean()                          # ...and looks 3x better than walk-forward keeps
    assert deflated_sharpe(full[best], len(table)) < 0.95             # the deflated Sharpe is not fooled
