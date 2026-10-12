"""EXAMPLE strategy: the classic opening-range breakout (ORB) on a US index.

This is a textbook, public-domain rule included to exercise the engine end to end — NOT one of the strategies the
private book trades. Rules:
  * opening range = high / low of the first `range_min` minutes after the 09:30 New York open
  * at the end of the range, rest a buy-stop at the high and a sell-stop at the low (one cancels the other)
  * stop = the other side of the range (1R = range width, floored), target `rr` R, entries valid until `last_entry`
  * flat at 15:55 New York
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tradebot.bars import Bars
from tradebot.strategies.base import STOP, Plan, ny_ts


@dataclass
class OpeningRangeBreakout:
    name: str = "ORB"
    symbol: str = "US100"
    magic: int = 910001
    stake: float = 0.005
    rank: int = 1
    history_days: int = 3
    range_min: int = 30
    rr: float = 2.0
    min_risk: float = 5.0           # price units; a range narrower than this is padded to it
    last_entry: tuple = (12, 0)
    flat: tuple = (15, 55)

    def signal_time(self, day: int) -> int:
        return ny_ts(day, 9, 30) + self.range_min * 60

    def arm(self, bars: Bars, day: int) -> Optional[Plan]:
        t0 = ny_ts(day, 9, 30)
        t1 = t0 + self.range_min * 60
        w = bars.window(t0, t1)
        if w.stop - w.start < self.range_min // 2:          # half the range missing: holiday / feed gap
            return None
        hi, lo = float(np.max(bars.h[w])), float(np.min(bars.l[w]))
        risk = max(hi - lo, self.min_risk)
        return Plan(strategy=self.name, symbol=self.symbol, day=day, side=0, entry=STOP,
                    level_up=hi, level_dn=lo, risk=risk, rr=self.rr, be_r=0.0,
                    signal_ts=t1, expiry_ts=ny_ts(day, *self.last_entry), deadline_ts=ny_ts(day, *self.flat))
