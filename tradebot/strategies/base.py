"""The strategy contract.

A strategy is a pure function of closed bars: at its `signal_time` it returns a `Plan` (or None for "no trade today").
Everything after that — sizing, orders, fills, exits, the account's risk limits — is the engine's job. Keeping the
rules side-effect free is what makes the live signal provably identical to the backtested one: the same function
runs in both (see tests/test_runtime.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional, Protocol
from zoneinfo import ZoneInfo

from tradebot.bars import Bars

MARKET, STOP = "MARKET", "STOP"
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Plan:
    """One strategy's decision for one session day, fixed at its signal time."""
    strategy: str
    symbol: str
    day: int                    # UTC day index (epoch // 86400)
    side: int                   # +1 long, -1 short, 0 = both sides rest (one-cancels-other)
    entry: str                  # MARKET | STOP
    level_up: float             # buy-stop level (STOP entries)
    level_dn: float             # sell-stop level (STOP entries)
    risk: float                 # stop distance in price == 1R
    rr: float                   # target in R (0 = no target)
    be_r: float                 # move the stop to entry at +be_r R (0 = never)
    signal_ts: int              # instant the signal became knowable
    expiry_ts: int              # a resting entry is cancelled here
    deadline_ts: int            # hard flat
    abort_ts: int = 0           # one-shot check on the first closed bar at/after this instant (0 = none)
    abort_r: float = float("nan")   # ... closing the trade if it marks at or below this many R


class Strategy(Protocol):
    name: str
    symbol: str
    magic: int                  # one MT5 magic number per strategy: their orders and positions never mix
    stake: float                # risk per trade, fraction of account balance
    rank: int                   # book priority when the account is full (higher displaces lower)
    history_days: int

    def signal_time(self, day: int) -> int: ...
    def arm(self, bars: Bars, day: int) -> Optional[Plan]: ...


def ny_ts(day: int, hh: int, mm: int) -> int:
    """Epoch seconds of hh:mm New York wall-clock on UTC day `day` (DST handled by the tz database)."""
    d = datetime.utcfromtimestamp(int(day) * 86400).date()
    return int(datetime.combine(d, time(hh, mm), tzinfo=NY).timestamp())
