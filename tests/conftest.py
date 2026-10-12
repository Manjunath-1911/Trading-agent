import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradebot.bars import Bars                      # noqa: E402
from tradebot.strategies.base import ny_ts          # noqa: E402

DAY = 20376                                          # 2025-10-14, a Tuesday (UTC day index)


def day_bars(symbol: str, path, day: int = DAY) -> Bars:
    """1m bars for one UTC day; `path(minutes_since_ny_0930)` gives the bar's (open, high, low, close)."""
    t0 = day * 86400
    ts = np.arange(t0, t0 + 86400, 60, dtype=np.int64)
    open_ = ny_ts(day, 9, 30)
    ohlc = np.array([path((t - open_) // 60) for t in ts], float)
    return Bars(symbol, ts, ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], np.ones(len(ts)))


def breakout_up(m):
    """Quiet 100-110 opening range, breakout at 10:20 New York, rally to 135 by 11:00, then flat."""
    if m < 0:
        return (105, 106, 104, 105)
    if m < 30:
        return (105, 110 if m == 7 else 107, 100 if m == 21 else 103, 105)
    if m < 50:
        return (106, 108, 104, 106)
    if m < 90:
        p = 106 + (m - 50) * 0.75
        return (p, p + 0.75, p - 0.25, p + 0.75)
    return (136, 136.5, 135.5, 136)


@pytest.fixture
def cfg(tmp_path):
    from tradebot.config import Config
    return Config(mode="PAPER", dry_run=True, capital=10_000, initial_balance=10_000, log_dir=tmp_path,
                  max_daily_loss_pct=3.75, max_overall_loss_pct=9.5, dd_cascade="4:0.5,6:0.3", collective_cap=0.03,
                  max_open=3)
