r"""The broker-server clock: measured, cached, applied per bar.

MT5 stamps everything (ticks, bars, deals, positions, order expiries) in the broker's SERVER time, encoded as if it were
UTC. Many prop servers run a "New York close" clock (midnight = 17:00 New York): UTC+3 in US summer, UTC+2 in winter,
switching on the US DST dates. Every rule here is stated in true UTC or New York time, so every timestamp crossing the
boundary is shifted by the offset in force AT THAT INSTANT. Three bugs this design fixes:
  1. measuring the offset once at start-up — a process running across a DST switch drifts an hour;
  2. reading it off the last tick of a FX symbol — on a weekend that tick is Friday's (offset ~ -40 h);
  3. applying today's offset to history — bars before a DST switch are an hour off.
So: a fresh tick from a 24/7 symbol (crypto) decides the current offset, re-measured every 10 minutes, cached on disk for
cold starts, env fallback last; history is converted with the New York-close model when it matches the measured offset.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

log = logging.getLogger("tradebot.clock")

NY = ZoneInfo("America/New_York")
NY_CLOSE_HOURS = 7                  # server midnight == 17:00 New York
CLOCK_SYMBOLS = ("BTCUSD", "EURUSD", "XAUUSD")
TTL_S = 600                         # re-measure this often
FRESH_S = 300                       # a tick older than this cannot anchor the offset
_HOUR = 3600


def model_offset(utc_ts: float) -> int:
    """The New York-close server's offset (seconds) in force at true-UTC `utc_ts`."""
    return NY_CLOSE_HOURS * _HOUR + int(datetime.fromtimestamp(float(utc_ts), tz=NY).utcoffset().total_seconds())


def model_offset_array(utc_ts) -> np.ndarray:
    import pandas as pd
    idx = pd.to_datetime(np.asarray(utc_ts, dtype=np.int64), unit="s", utc=True).tz_convert(NY)
    ny = (idx.tz_localize(None) - idx.tz_convert("UTC").tz_localize(None)).total_seconds()
    return (NY_CLOSE_HOURS * _HOUR + np.asarray(ny, dtype=np.int64)).astype(np.int64)


class ServerClock:
    def __init__(self, mt5, symbols=CLOCK_SYMBOLS, cache: Path | None = None, fallback_hours: int | None = None):
        self.mt5, self.symbols, self.cache = mt5, tuple(symbols), cache
        fb = os.getenv("MT5_UTC_OFFSET", "") if fallback_hours is None else str(fallback_hours)
        self.fallback = int(float(fb)) * _HOUR if str(fb).strip() else None
        self._off: int | None = None
        self._measured_at = 0.0
        self.source = "none"

    def _measure(self) -> int | None:
        now = time.time()
        for s in self.symbols:
            try:
                t = self.mt5.symbol_info_tick(s)
            except Exception:
                t = None
            if not t or not getattr(t, "time", 0):
                continue
            delta = float(t.time) - now
            off_h = int(round(delta / _HOUR))
            if -14 <= off_h <= 14 and abs(delta - off_h * _HOUR) <= FRESH_S:
                self.source = "tick:%s" % s
                return off_h * _HOUR
        return None

    def offset(self) -> int:
        """Server minus UTC, seconds. Fresh tick > disk cache > MT5_UTC_OFFSET > 0 (logged)."""
        now = time.time()
        if self._off is not None and now - self._measured_at < TTL_S:
            return self._off
        off = self._measure()
        if off is not None:
            if self._off is not None and off != self._off:
                log.warning("server offset changed %+dh -> %+dh (DST switch?)", self._off // _HOUR, off // _HOUR)
            self._off, self._measured_at = off, now
            self._save()
            return off
        if self._off is not None:                       # keep the last good one, retry in a minute
            self._measured_at = now - TTL_S + 60
            return self._off
        for value, src in ((self._load(), "cache"), (self.fallback, "env")):
            if value is not None:
                self._off, self._measured_at, self.source = value, now - TTL_S + 60, src
                return value
        log.error("server offset unknown — assuming 0; set MT5_UTC_OFFSET")
        self._off, self._measured_at, self.source = 0, now - TTL_S + 60, "assumed"
        return 0

    def server_now(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=self.offset())

    def model_applies(self) -> bool:
        return model_offset(time.time()) == self.offset()

    def to_utc(self, server_ts):
        """Server-stamped epoch seconds (scalar or array) -> true UTC epoch seconds, DST-correct per bar."""
        srv = np.asarray(server_ts, dtype=np.int64)
        if not self.model_applies():
            out = srv - self.offset()
        else:
            guess = srv - self.offset()
            out = srv - model_offset_array(np.atleast_1d(guess)).reshape(srv.shape)
        return int(out) if np.ndim(server_ts) == 0 else out

    def to_server(self, utc_ts: float) -> int:
        return int(utc_ts) + (model_offset(utc_ts) if self.model_applies() else self.offset())

    def _save(self) -> None:
        if self.cache:
            try:
                self.cache.parent.mkdir(parents=True, exist_ok=True)
                self.cache.write_text(json.dumps({"offset": self._off, "ts": time.time()}), encoding="utf-8")
            except Exception:
                log.debug("offset cache write failed", exc_info=True)

    def _load(self) -> int | None:
        if not self.cache or not self.cache.exists():
            return None
        try:
            d = json.loads(self.cache.read_text(encoding="utf-8"))
            off = int(d["offset"])
            if model_offset(time.time()) != off and model_offset(float(d.get("ts", 0))) == off:
                return model_offset(time.time())          # cached before a DST switch: trust the model
            return off
        except Exception:
            return None
