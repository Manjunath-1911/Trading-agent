"""The account circuit breaker: daily and overall equity loss limits, persisted across restarts.

Stops and targets protect one position. This protects the ACCOUNT — floating P&L included, across every strategy.
  * daily:   equity below the server-day anchor by `daily_pct` -> cancel entries, flatten, halt until the next day
  * overall: equity below the static starting balance by `overall_pct` -> flatten and halt for good (operator reset)
Design rules, each learned the hard way:
  * persistence is load-bearing: a restart must never manufacture a fresh daily allowance;
  * missing data fails CLOSED for new entries but never liquidates at an unknown price;
  * the day anchor is max(balance, equity) at the rollover — prop firms measure from the higher one;
  * the state is bound to the broker LOGIN: a new account (next challenge phase) gets a fresh budget automatically.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("tradebot.safety")


@dataclass
class SafetyState:
    day: str = ""
    day_start_balance: float | None = None
    initial_balance: float | None = None
    day_halted: bool = False
    halted: bool = False
    reason: str = ""
    last_equity: float | None = None
    login: int = 0


def load_state(path: Path, initial: float) -> SafetyState:
    st = SafetyState(initial_balance=initial if initial > 0 else None)
    if not path.exists():
        return st
    try:
        for k, v in json.loads(path.read_text(encoding="utf-8")).items():
            if hasattr(st, k):
                setattr(st, k, v)
    except Exception as exc:                                    # a corrupt breaker must never re-arm the account
        st.halted, st.reason = True, "safety_state_corrupt"
        log.error("safety state unreadable (%s) — failing closed", exc)
    return st


def save_state(st: SafetyState, path: Path) -> None:
    """Atomic write: a crash cannot leave a half-written loss budget."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(st), sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        log.exception("safety state write failed — new entries fail closed")
        st.halted, st.reason = True, "safety_state_write_failed"


def snapshot(brokers) -> dict | None:
    """One valid {balance, equity, login} from any broker (they share one account)."""
    for b in brokers:
        try:
            s = b.equity()
            if s and s["balance"] > 0 and s["equity"] > 0:
                return s
        except Exception:
            continue
    return None


def _adopt_account(st: SafetyState, snap: dict, initial: float) -> bool:
    login = int(snap.get("login") or 0)
    fresh = bool(login and st.login and login != st.login)
    if fresh:
        log.error("safety state belonged to account %s, now %s — fresh loss budget", st.login, login)
        st.day, st.day_start_balance, st.day_halted, st.halted, st.reason = "", None, False, False, ""
        st.initial_balance = initial if initial > 0 else None
    if login:
        st.login = login
    return fresh


def check(st: SafetyState, brokers, server_now: datetime, cfg, realized_today: float = 0.0,
          has_open_positions: bool = False) -> tuple[bool, str | None, bool]:
    """-> (new entries allowed, reason, flatten now). Call once per cycle, then save_state()."""
    initial = float(cfg.initial_balance or 0.0)
    snap = snapshot(brokers)
    if st.halted and not (snap and _adopt_account(st, snap, initial)):
        return False, st.reason or "overall_breaker", True
    if snap is None:
        return False, "equity_unavailable", False
    _adopt_account(st, snap, initial)
    today = server_now.strftime("%Y-%m-%d")
    if st.day != today:
        first_attach = not st.day
        st.day, st.day_halted, st.reason = today, False, ""
        if first_attach and has_open_positions:
            # positions carried over a rollover we did not see: their floating P&L at midnight is unknown
            st.day_start_balance, st.reason = None, "day_anchor_unavailable"
            return False, st.reason, False
        st.day_start_balance = (snap["balance"] - realized_today if first_attach
                                else max(snap["equity"], snap["balance"]))
    if st.day_start_balance is None:
        return False, st.reason or "day_anchor_unavailable", False
    if not st.initial_balance:
        st.initial_balance = snap["balance"]
    st.last_equity = snap["equity"]
    daily = 100.0 * (st.day_start_balance - snap["equity"]) / st.day_start_balance
    overall = 100.0 * (st.initial_balance - snap["equity"]) / st.initial_balance
    if cfg.max_overall_loss_pct > 0 and overall >= cfg.max_overall_loss_pct:
        st.halted, st.reason = True, "overall_breaker"
    elif cfg.max_daily_loss_pct > 0 and daily >= cfg.max_daily_loss_pct:
        st.day_halted, st.reason = True, "daily_breaker"
    if st.halted or st.day_halted:
        log.error("%s TRIPPED — equity %.2f, day start %.2f (%.2f%%), initial %.2f (%.2f%%)", st.reason.upper(),
                  snap["equity"], st.day_start_balance, -daily, st.initial_balance, -overall)
        return False, st.reason, True
    return True, None, False


def server_midnight_utc(server_now: datetime, offset_s: int) -> datetime:
    """True-UTC instant of the broker server's midnight (for 'realized P&L since the day opened')."""
    return server_now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(seconds=offset_s)
