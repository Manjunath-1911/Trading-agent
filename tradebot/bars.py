"""OHLCV bars as parallel numpy arrays (epoch seconds, true UTC, bar OPEN time)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bars:
    symbol: str
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    def window(self, t0: int, t1: int) -> slice:
        """Index slice of the bars opening in [t0, t1)."""
        a, b = np.searchsorted(self.ts, [t0, t1], "left")
        return slice(int(a), int(b))

    @classmethod
    def from_arrays(cls, symbol, ts, o, h, l, c, v=None) -> "Bars":
        ts = np.asarray(ts, np.int64)
        keep = np.r_[True, np.diff(ts) > 0]                     # feeds occasionally repeat a bar
        v = np.zeros(len(ts)) if v is None else np.asarray(v, float)
        return cls(symbol, ts[keep], *(np.asarray(x, float)[keep] for x in (o, h, l, c, v)))
