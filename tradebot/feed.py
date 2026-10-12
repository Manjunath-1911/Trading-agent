"""Price feeds. Both return (ts, o, h, l, c, v) numpy arrays in TRUE UTC epoch seconds (bar open time).

`include_forming=True` keeps the bar that has not closed yet — but only its OPEN is safe to read (a rule may anchor a
level on the open of the current minute). Every window a strategy measures must end before it.
"""
from __future__ import annotations

import os
import time as _time
from pathlib import Path

import numpy as np

_TF = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


class ReplayFeed:
    """Serves stored bars up to `now`, the forming bar collapsed to its open. Drives tests and paper replays: the
    runtime cannot see one second further than it could live."""
    name = "replay"

    def __init__(self, archive: dict):
        self.archive, self.now = archive, 0          # archive: symbol -> Bars

    def bars(self, symbol, days=7, tf="1m", include_forming=False):
        b = self.archive.get(symbol)
        if b is None:
            return None
        hi = int(np.searchsorted(b.ts, self.now, "right"))
        lo = max(0, int(np.searchsorted(b.ts, self.now - days * 86400, "left")))
        if hi <= lo:
            return None
        ts, o, h, l, c, v = (x[lo:hi].copy() for x in (b.ts, b.o, b.h, b.l, b.c, b.v))
        if ts[-1] + _TF[tf] > self.now:              # the last bar has not closed
            if not include_forming:
                return tuple(x[:-1] for x in (ts, o, h, l, c, v))
            h[-1] = l[-1] = c[-1] = o[-1]
        return ts, o, h, l, c, v


class MT5Feed:
    """Bars from the MetaTrader 5 terminal, converted bar by bar from server time to true UTC (mt5_clock)."""
    name = "mt5"

    def __init__(self, symbol_map: dict | None = None, log_dir: Path | None = None):
        import MetaTrader5 as mt5
        from tradebot.mt5_clock import CLOCK_SYMBOLS, ServerClock
        path = os.getenv("MT5_PATH", "").strip()     # pick WHICH terminal when two accounts run on one machine
        if not (mt5.initialize(path=path) if path else mt5.initialize()):
            raise RuntimeError("mt5.initialize() failed: %s" % (mt5.last_error(),))
        self.mt5 = mt5
        self.smap = dict(symbol_map or {})
        for kv in filter(None, os.getenv("MT5_SYMBOL_MAP", "").split(",")):
            k, v = kv.split("=")
            self.smap[k.strip()] = v.strip()
        self.tfmap = {"1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15, "1h": mt5.TIMEFRAME_H1}
        self.clock = ServerClock(mt5, tuple(dict.fromkeys(CLOCK_SYMBOLS + tuple(self.smap.values()))),
                                 cache=(log_dir / "mt5_utc_offset.json") if log_dir else None)

    def bars(self, symbol, days=7, tf="1m", include_forming=False):
        bsym = self.smap.get(symbol, symbol)
        self.mt5.symbol_select(bsym, True)
        n = int(days * 86400 / _TF[tf]) + 10
        rates = self.mt5.copy_rates_from_pos(bsym, self.tfmap[tf], 0, n)
        # symbol_select only REQUESTS history; a cold terminal answers from a stale local cache without any error.
        # Give the download a bounded moment: a stale window would rebuild an OLD session's levels.
        deadline = _time.time() + 3.0
        while (rates is not None and len(rates) and _time.time() < deadline
               and _time.time() - (int(rates[-1]["time"]) - self.clock.offset()) > 2 * _TF[tf]):
            _time.sleep(0.4)
            rates = self.mt5.copy_rates_from_pos(bsym, self.tfmap[tf], 0, n)
        if rates is None or not len(rates):
            return None
        if len(rates) > 1 and not include_forming:
            rates = rates[:-1]
        ts = self.clock.to_utc(rates["time"].astype(np.int64))
        return ts, rates["open"], rates["high"], rates["low"], rates["close"], rates["tick_volume"].astype(float)
