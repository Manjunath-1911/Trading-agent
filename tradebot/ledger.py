"""One book across several processes: at most N positions open, ranked displacement, an account-wide risk cap.

Strategies run in separate processes (e.g. an index runtime and a crypto runtime on the same prop account), so the
book they must agree on lives in ONE JSON file guarded by an exclusive-create lock file. Rules:
  * a new position is admitted while fewer than `max_open` are open;
  * when the book is full, it DISPLACES the lowest-ranked open position if it outranks it (newest first among equal
    ranks), otherwise it is refused;
  * with `risk_cap` > 0 the account's total open risk is capped; a new position gets min(stake, room left);
  * a displaced position owned by another process becomes a close request that its owner honours next cycle.
Every position counts at stake x `risk_pad` (lot rounding can only make it bigger), so the cap holds in the worst case.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

LOCK_TIMEOUT_S = 5.0
LOCK_STALE_S = 30.0


class Ledger:
    def __init__(self, path: Path, rank: dict, max_open: int = 3, risk_cap: float = 0.0, risk_pad: float = 1.0,
                 last: tuple = ()):
        self.path, self.rank, self.max_open = Path(path), dict(rank), max_open
        self.risk_cap, self.risk_pad, self.last = risk_cap, risk_pad, set(last)   # `last`: seats ranked below all
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ── file + lock ──────────────────────────────────────────────────────────
    def _lock(self) -> int:
        lk, t0 = self.path.with_suffix(".lock"), time.time()
        while True:
            try:
                return os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    if time.time() - lk.stat().st_mtime > LOCK_STALE_S:     # a crashed holder never blocks forever
                        lk.unlink()
                        continue
                except FileNotFoundError:
                    continue
                if time.time() - t0 > LOCK_TIMEOUT_S:
                    raise TimeoutError("ledger lock held > %ss: %s" % (LOCK_TIMEOUT_S, lk))
                time.sleep(0.05)

    def _unlock(self, fd: int) -> None:
        os.close(fd)
        try:
            self.path.with_suffix(".lock").unlink()
        except FileNotFoundError:
            pass

    def _load(self) -> dict:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            d = {}
        d.setdefault("positions", [])
        d.setdefault("close_requests", [])
        return d

    def _save(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
        for _ in range(40):                     # Windows: a reader holding the file can block the rename briefly
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05)
        os.replace(tmp, self.path)

    def _txn(self, fn, write=True):
        fd = self._lock()
        try:
            d = self._load()
            out = fn(d)
            if write:
                self._save(d)
            return out
        finally:
            self._unlock(fd)

    def _rank(self, seat: str) -> int:
        if seat in self.last:
            return 0
        if seat not in self.rank:
            raise KeyError("ledger: seat %r has no rank" % seat)
        return self.rank[seat]

    # ── the rule ─────────────────────────────────────────────────────────────
    def admit(self, owner: str, seat: str, key: str, now: int, ticket: int = 0, risk_frac: float = 0.0,
              side: int = 0) -> dict:
        """-> {"status": "ok"|"displace"|"refuse", "victim", "open", "risk_frac" (granted), "why"}."""
        rank = self._rank(seat)

        def fn(d):
            pos = d["positions"]
            for p in pos:                                        # re-admitting an open key is idempotent
                if p["owner"] == owner and p["key"] == key:
                    p.update(ticket=ticket or p["ticket"], risk_frac=risk_frac or p["risk_frac"])
                    return dict(status="ok", victim=None, open=len(pos), risk_frac=p["risk_frac"])
            victim = None
            if len(pos) >= self.max_open:
                victim = min(pos, key=lambda p: (self.rank.get(p["seat"], 0) if p["seat"] not in self.last else 0,
                                                 -p["opened_ts"]))
                if rank <= (0 if victim["seat"] in self.last else self.rank.get(victim["seat"], 0)):
                    return dict(status="refuse", why="max_open", victim=None, open=len(pos), risk_frac=0.0)
            grant = float(risk_frac)
            if self.risk_cap > 0 and grant > 0:
                held = sum(p["risk_frac"] for p in pos if p is not victim)
                grant = min(grant, self.risk_cap / self.risk_pad - held)
                if grant <= 1e-9:
                    return dict(status="refuse", why="risk_cap", victim=None, open=len(pos), risk_frac=0.0)
            pos.append(dict(owner=owner, seat=seat, key=key, ticket=int(ticket), opened_ts=int(now),
                            risk_frac=grant, side=int(side)))
            if victim is None:
                return dict(status="ok", victim=None, open=len(pos), risk_frac=grant)
            pos.remove(victim)
            victim = dict(victim, requested_for=seat)
            d["close_requests"].append(victim)
            return dict(status="displace", victim=victim, open=len(pos), risk_frac=grant)
        return self._txn(fn)

    def restore(self, owner: str, key: str, victim: dict | None) -> None:
        """Undo admit() for a trade that was then NOT opened; a displacement for it must not close anything."""
        def fn(d):
            d["positions"] = [p for p in d["positions"] if not (p["owner"] == owner and p["key"] == key)]
            if victim:
                req = [c for c in d["close_requests"] if c["owner"] == victim["owner"] and c["key"] == victim["key"]]
                if req:
                    d["close_requests"] = [c for c in d["close_requests"] if c not in req]
                    d["positions"].append({k: v for k, v in req[0].items() if k != "requested_for"})
        self._txn(fn)

    def release(self, owner: str, key: str) -> None:
        def fn(d):
            keep = lambda p: not (p["owner"] == owner and p["key"] == key)
            d["positions"] = [p for p in d["positions"] if keep(p)]
            d["close_requests"] = [p for p in d["close_requests"] if keep(p)]
        self._txn(fn)

    def pending_closes(self, owner: str) -> list:
        return self._txn(lambda d: [dict(p) for p in d["close_requests"] if p["owner"] == owner], write=False)

    def sync(self, owner: str, held: dict) -> None:
        """Make the ledger equal what `owner` really holds ({key: (seat, ticket, opened_ts, risk_frac)}). Never displaces:
        a position that exists exists; the rule only decides at the moment a new one opens."""
        def fn(d):
            keep = [p for p in d["positions"] if p["owner"] != owner or p["key"] in held]
            known = {p["key"] for p in keep if p["owner"] == owner}
            known |= {p["key"] for p in d["close_requests"] if p["owner"] == owner}
            for key, (seat, ticket, opened_ts, risk_frac) in held.items():
                if key not in known:
                    keep.append(dict(owner=owner, seat=seat, key=key, ticket=int(ticket), opened_ts=int(opened_ts),
                                     risk_frac=float(risk_frac), side=0))
            d["positions"] = keep
            d["close_requests"] = [p for p in d["close_requests"] if p["owner"] != owner or p["key"] in held]
        self._txn(fn)

    def open_positions(self) -> list:
        return self._txn(lambda d: [dict(p) for p in d["positions"]], write=False)

    # ── the account breaker, published by its owner, obeyed by every process ─
    def set_halt(self, reason: str | None, flatten: bool = True) -> None:
        def fn(d):
            d["halt"] = {"reason": reason, "flatten": bool(flatten)} if reason else None
            d["breaker_ts"] = int(time.time())               # heartbeat: a silent breaker owner stops new entries
        self._txn(fn)

    def halt(self) -> dict | None:
        return self._txn(lambda d: d.get("halt"), write=False)

    def breaker_age(self) -> float | None:
        ts = self._txn(lambda d: d.get("breaker_ts"), write=False)
        return None if ts is None else time.time() - float(ts)
