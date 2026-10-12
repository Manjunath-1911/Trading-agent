"""Operational plumbing: Telegram alerts, the single-instance lock, Windows keep-awake, the trade log."""
from __future__ import annotations

import csv
import logging
import os
import sqlite3
import sys
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("tradebot.ops")


class Notifier:
    """Best-effort Telegram sender on a daemon thread: a slow or blocked endpoint never delays a trading cycle."""

    def __init__(self, token: str = "", chat_id: str = ""):
        self.token, self.chat_id = token.strip(), chat_id.strip()
        self.enabled = bool(self.token and self.chat_id)

    @classmethod
    def from_env(cls) -> "Notifier":
        return cls(os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", ""))

    def _send(self, text: str) -> None:
        try:
            data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}).encode()
            urllib.request.urlopen("https://api.telegram.org/bot%s/sendMessage" % self.token, data=data, timeout=8)
        except Exception as e:                                  # never let alerting break trading
            log.warning("notify failed: %s", e)

    def info(self, text: str) -> None:
        if self.enabled:
            threading.Thread(target=self._send, args=(text,), daemon=True).start()


class AlreadyRunning(RuntimeError):
    pass


_lock_handle = None


def acquire_singleton(lock_file: Path) -> None:
    """OS-level lock held for the process lifetime: two copies of one runtime would double every position. The kernel
    drops it on any exit (crash, kill, power loss), so a dead predecessor never blocks a restart."""
    global _lock_handle
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_file, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(1 << 30)                                    # lock a byte past the content: the PID stays readable
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise AlreadyRunning("another instance holds %s" % lock_file)
    fh.seek(0)
    fh.truncate()
    fh.write("pid=%d started=%s\n" % (os.getpid(), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    fh.flush()
    _lock_handle = fh


def keep_awake(reason: str = "trading agent") -> bool:
    """Windows Modern Standby froze a live agent for two hours once; a suspended agent does not manage positions.
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED [| ES_AWAYMODE_REQUIRED]) — what media players do."""
    if not sys.platform.startswith("win") or os.getenv("KEEP_AWAKE", "1") != "1":
        return False
    import ctypes
    for flags in (0x80000000 | 0x1 | 0x40, 0x80000000 | 0x1):
        if ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)):
            log.info("keep-awake held for %s", reason)
            return True
    log.error("keep-awake refused by the OS — Windows may suspend this process")
    return False


class TradeLog:
    """Append-only trade log: SQLite (queryable) + CSV (human-readable)."""
    COLS = ("ts", "strategy", "symbol", "side", "entry", "exit", "stop", "pnl", "r", "reason", "ticket")

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.csv = directory / "trades.csv"
        self.db = sqlite3.connect(str(directory / "trades.db"))
        self.db.execute("CREATE TABLE IF NOT EXISTS trades (%s)" % ", ".join("%s TEXT" % c for c in self.COLS))
        self.db.commit()
        if not self.csv.exists():
            with self.csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.COLS)

    def log(self, **row) -> None:
        row.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        vals = [str(row.get(c, "")) for c in self.COLS]
        self.db.execute("INSERT INTO trades VALUES (%s)" % ",".join("?" * len(vals)), vals)
        self.db.commit()
        with self.csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(vals)
