"""The engine: a one-minute loop that arms strategies, sends orders, manages exits and enforces the account's rules.

Order of every cycle:
  1. account breaker (daily / overall equity loss) — a trip cancels entries and flattens everything
  2. the shared book (ledger): honour close requests from other processes, then sync what we hold
  3. per strategy, highest book rank first:
       not armed  -> at its signal time, on fresh bars: arm -> adopt an existing position (restart) / skip if it already
                     traded today (broker deal history) / place the entry (market, or resting stops)
       armed      -> retry a close the broker rejected; adopt a fill (run the book rule on it); manage the position
                     (breakeven, one-shot abort, hard flat); book a position the broker closed; expire resting orders
Everything is re-derivable from the broker plus the bars, which is what makes a restart mid-session safe. Stops and
targets always live AT the broker: a dead agent still takes its stop and its target.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from tradebot import news_gate as NG
from tradebot import safety as SF
from tradebot.bars import Bars
from tradebot.broker import CLOSED_OK
from tradebot.config import Config
from tradebot.ledger import Ledger
from tradebot.risk import cascade_factor, governor
from tradebot.strategies import MARKET, Plan

log = logging.getLogger("tradebot.runtime")


@dataclass
class SleeveState:
    name: str
    day: int = 0
    armed: bool = False
    plan: Optional[Plan] = None
    orders: dict = field(default_factory=dict)     # "up" / "dn" -> resting order ticket
    ticket: int = 0                                 # open position
    entry: float = math.nan
    stop: float = math.nan
    side: int = 0
    entry_ts: int = 0
    be_done: bool = False
    abort_checked: bool = False
    risk_frac: float = 0.0                          # equity fraction this position actually committed
    finished: bool = False                          # nothing more to do today
    close_due: str = ""                             # a close the broker rejected: retried every cycle

    def reset(self, day: int) -> None:
        self.__init__(self.name, day=day)


class Engine:
    def __init__(self, strategies, brokers: dict, feed, cfg: Config, ledger: Ledger | None = None, tlog=None,
                 notify=None, owner: str = "main"):
        self.strat = {s.name: s for s in strategies}
        self.order = sorted(self.strat, key=lambda n: -self.strat[n].rank)
        self.states = {n: SleeveState(n) for n in self.strat}
        self.brokers, self.feed, self.cfg, self.ledger, self.owner = brokers, feed, cfg, ledger, owner
        self.tlog, self.notify = tlog, notify
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        self.events = cfg.log_dir / "events.jsonl"
        self.safety_path = cfg.log_dir / "safety.json"
        self.safety = SF.load_state(self.safety_path, cfg.initial_balance)
        self.news: list = []

    # ── bookkeeping ──────────────────────────────────────────────────────────
    def event(self, **row) -> None:
        """One JSON line per decision, REFUSALS INCLUDED: 'did live do what the backtest did?' needs the trades that
        did not happen too."""
        row["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self.events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            log.debug("event log write failed", exc_info=True)

    def committed(self) -> float:
        return sum(st.risk_frac for st in self.states.values() if st.ticket)

    def _info(self, text: str) -> None:
        if self.notify:
            self.notify.info(text)

    # ── entries ──────────────────────────────────────────────────────────────
    def granularity_ok(self, broker, st, entry: float, stop: float, risk_usd: float) -> bool:
        """Refuse a trade whose SMALLEST tradeable size already risks far more than the stake: below some account size
        the contract, not the strategy, would be choosing the risk."""
        try:
            floor_risk = broker.risk_per_lot(self.strat[st.name].symbol, entry, stop) * broker.min_lots(
                self.strat[st.name].symbol)
        except Exception:
            return True
        if floor_risk > self.cfg.max_risk_overshoot * risk_usd > 0:
            self.event(kind="refused", strategy=st.name, reason="granularity", floor_risk=floor_risk, want=risk_usd)
            return False
        return True

    def place(self, st: SleeveState, bars: Bars, now: int) -> None:
        p, s, b = st.plan, self.strat[st.name], self.brokers[st.name]
        try:
            bal = float(b.account_balance())
        except Exception:
            bal = float("nan")
        bal = bal if bal == bal and bal > 0 else self.cfg.capital
        frac = governor(s.stake, self.committed(), self.cfg.collective_cap)
        if frac <= 0:
            self.event(kind="refused", strategy=st.name, reason="collective_cap", committed=self.committed())
            st.finished = True
            return
        frac *= cascade_factor(bal, self.cfg.initial_balance, self.cfg.dd_cascade)
        st.risk_frac, risk_usd = frac, frac * bal

        if p.entry == MARKET:
            if now - p.signal_ts > self.cfg.max_market_lag_s:          # late is a different trade: skip, never chase
                self.event(kind="refused", strategy=st.name, reason="late_market_entry", lag=now - p.signal_ts)
                st.finished = True
                return
            px = float(bars.c[-1])
            if not self.granularity_ok(b, st, px, px - p.side * p.risk, risk_usd):
                st.finished = True
                return
            res = b.market_order(p.symbol, p.side, risk_usd, px - p.side * p.risk)
            self.event(kind="market_order", strategy=st.name, side=p.side, risk_usd=risk_usd, result=res)
            if res.get("status") not in ("FILLED", "DRY"):
                st.finished = True
                return
            fill = float(res.get("fill_price") or px)
            st.ticket, st.entry, st.side, st.entry_ts = int(res.get("ticket") or 0), fill, p.side, now
            st.stop = fill - p.side * p.risk                           # re-anchor 1R on the ACTUAL fill
            if st.ticket:
                b.modify_sltp(p.symbol, st.ticket, st.stop, fill + p.side * p.rr * p.risk if p.rr else 0.0)
                # a market fill already holds its ticket, so the adopt() path never sees it as new: judge it now
                self.admit_fill(st, now)
            return

        legs = [("up", 1), ("dn", -1)] if p.side == 0 else [("up" if p.side > 0 else "dn", p.side)]
        for key, side in legs:
            level = p.level_up if side > 0 else p.level_dn
            stop = level - side * p.risk
            if not self.granularity_ok(b, st, level, stop, risk_usd):
                st.finished = True
                return
            res = b.stop_order(p.symbol, side, risk_usd, level, stop, tp=level + side * p.rr * p.risk if p.rr else 0.0,
                               expiry_ts=p.expiry_ts)
            self.event(kind="stop_order", strategy=st.name, leg=key, level=level, stop=stop, result=res)
            if res.get("status") == "IMMEDIATE":                     # price already through: the backtest filled AT
                self.event(kind="missed", strategy=st.name, leg=key)  # the level; chasing would select the losers
            elif res.get("status") in ("PLACED", "DRY"):
                st.orders[key] = int(res.get("ticket") or 0)
        if not st.orders:
            st.finished = True

    def cancel(self, st: SleeveState, why: str) -> None:
        """Cancel resting entries. MT5 carries the order ticket over to the position, so the filled leg is skipped."""
        for key, tkt in list(st.orders.items()):
            if tkt and tkt != st.ticket:
                self.brokers[st.name].cancel_order(tkt)
            self.event(kind="cancelled", strategy=st.name, leg=key, reason=why)
            st.orders.pop(key)

    # ── fills, the book rule, exits ──────────────────────────────────────────
    def adopt(self, st: SleeveState) -> bool:
        """Pick up this strategy's position from the broker (its own magic: never a sibling's)."""
        pos = self.brokers[st.name].list_positions(self.strat[st.name].symbol)
        if not pos:
            return False
        p = pos[0]
        if st.ticket != p["ticket"]:
            st.ticket, st.entry, st.side, st.stop = int(p["ticket"]), float(p["entry"]), int(p["side"]), float(p["sl"])
            st.entry_ts = int(p.get("open_ts") or time.time())
            if st.risk_frac <= 0:                                  # a restart: the fraction placed is gone
                st.risk_frac = self.strat[st.name].stake
            self.event(kind="filled", strategy=st.name, ticket=st.ticket, entry=st.entry, side=st.side)
            self._info("<b>%s</b> filled %s @ %.2f" % (st.name, "LONG" if st.side > 0 else "SHORT", st.entry))
        return True

    def admit_fill(self, st: SleeveState, now: int) -> bool:
        """Run the book rule on a fresh fill. False = refused and closed straight back."""
        if self.ledger is None:
            return True
        v = self.ledger.admit(self.owner, st.name, st.name, now, st.ticket, st.risk_frac, side=st.side)
        if v["status"] == "refuse":
            res = self.brokers[st.name].close_ticket(self.strat[st.name].symbol, st.ticket)
            self.event(kind="refused_fill", strategy=st.name, why=v.get("why"), open=v["open"], result=res)
            self.settle(st, v.get("why", "max_open"), res)
            return False
        if v["status"] == "displace" and v["victim"]["owner"] == self.owner:
            self.close_displaced(v["victim"]["key"], by=st.name)
        return True

    def close_displaced(self, name: str, by: str) -> None:
        st = self.states.get(name)
        if st is None or not st.ticket:
            if self.ledger:
                self.ledger.release(self.owner, name)
            return
        res = self.brokers[name].close_ticket(self.strat[name].symbol, st.ticket)
        self.event(kind="displaced", strategy=name, by=by, result=res)
        self.cancel(st, "displaced")
        self.settle(st, "displaced", res)

    def settle(self, st: SleeveState, reason: str, res: dict) -> bool:
        """Book the trade only once the broker CONFIRMS the close; a rejected close stays tracked and is retried."""
        if str((res or {}).get("status", "")).upper() in CLOSED_OK:
            self.book(st, reason)
            return True
        st.close_due = reason
        self.event(kind="close_failed", strategy=st.name, reason=reason, result=res)
        return False

    def book(self, st: SleeveState, reason: str) -> None:
        p, tkt = st.plan, st.ticket
        try:
            res = self.brokers[st.name].closed_result(tkt) if tkt else {"pnl": 0.0, "price": 0.0}
        except Exception:
            res = {"pnl": float("nan"), "price": 0.0}
        r = st.side * (res["price"] - st.entry) / p.risk if res.get("price") and p else float("nan")
        self.event(kind="exit", strategy=st.name, reason=reason, ticket=tkt, pnl=res.get("pnl"), r=r)
        if self.tlog:
            self.tlog.log(strategy=st.name, symbol=p.symbol if p else "", side=st.side, entry=st.entry,
                          exit=res.get("price"), stop=st.stop, pnl=res.get("pnl"), r=r, reason=reason, ticket=tkt)
        self._info("<b>%s</b> exit (%s) pnl %+.2f" % (st.name, reason, res.get("pnl") or 0.0))
        st.ticket, st.risk_frac, st.close_due, st.finished = 0, 0.0, "", True
        if self.ledger:
            self.ledger.release(self.owner, st.name)

    def manage(self, st: SleeveState, bars: Bars, now: int) -> None:
        """The exits the broker cannot hold, decided on CLOSED bars exactly as the backtest decides them."""
        p, b = st.plan, self.brokers[st.name]
        closed = bars.ts + 60 <= now                                    # never decide on the forming bar
        ts, cl = bars.ts[closed], bars.c[closed]
        since = ts >= st.entry_ts
        if p.be_r > 0 and not st.be_done and (since & (st.side * (cl - st.entry) / p.risk >= p.be_r)).any():
            b.modify_sltp(p.symbol, st.ticket, st.entry, st.entry + st.side * p.rr * p.risk if p.rr else 0.0)
            st.be_done, st.stop = True, st.entry
            self.event(kind="breakeven", strategy=st.name)
        if p.abort_ts and not st.abort_checked and np.isfinite(p.abort_r):
            k = np.flatnonzero(since & (ts >= p.abort_ts))
            if len(k):                                                   # evaluated ONCE, on the first bar at/after
                st.abort_checked = True
                mark = st.side * (float(cl[k[0]]) - st.entry) / p.risk
                if mark <= p.abort_r:
                    res = b.close_ticket(p.symbol, st.ticket)
                    self.event(kind="abort", strategy=st.name, mark_r=mark, result=res)
                    self.settle(st, "abort", res)
                    return
        if now >= p.deadline_ts:                                        # the hard flat: unconditional
            res = b.close_ticket(p.symbol, st.ticket)
            self.event(kind="deadline_flat", strategy=st.name, result=res)
            self.settle(st, "time", res)

    def flatten_all(self, why: str) -> None:
        """Cancel every entry and close every broker-visible position — enumerated from the BROKER, not memory: after a
        restart in-memory tickets may be empty while the account still carries risk."""
        for n, st in self.states.items():
            self.cancel(st, why)
            b = self.brokers[n]
            try:
                for o in b.list_pending_orders():
                    b.cancel_order(o["ticket"])
                for pos in b.list_positions():
                    res = b.close_ticket(pos["symbol"], pos["ticket"])
                    self.event(kind="safety_flatten", strategy=n, reason=why, result=res)
                    if str(res.get("status", "")).upper() in CLOSED_OK and pos["ticket"] == st.ticket:
                        st.ticket, st.risk_frac = 0, 0.0
                        if self.ledger:
                            self.ledger.release(self.owner, n)
            except Exception:
                log.exception("[%s] flatten failed", n)
            st.finished = st.finished or st.plan is not None
        self._info("<b>SAFETY %s</b> — flattened, new entries blocked" % why)

    def traded_today(self, st: SleeveState, now: int) -> bool:
        """Did this strategy already OPEN a position today? Read from the broker's deal history (its own magic), so it
        survives a restart — `finished` does not. Without it a restart re-arms a finished day."""
        b = self.brokers[st.name]
        if not hasattr(b, "deals_since"):
            return False
        try:
            deals = b.deals_since(datetime.fromtimestamp(now // 86400 * 86400, timezone.utc))
        except Exception:
            return False
        return any(d.get("kind") == "in" for d in deals)

    # ── the cycle ────────────────────────────────────────────────────────────
    def run_once(self, now: int | None = None) -> None:
        now = int(time.time()) if now is None else int(now)
        brokers = list(self.brokers.values())
        b0 = brokers[0]
        server_now = b0.clock.server_now() if hasattr(b0, "clock") else datetime.fromtimestamp(now, timezone.utc)
        open_any = any(st.ticket for st in self.states.values())
        allow, why, flat = SF.check(self.safety, brokers, server_now, self.cfg, has_open_positions=open_any)
        SF.save_state(self.safety, self.safety_path)
        if self.ledger:
            self.ledger.set_halt(None if allow else why, flatten=flat)    # other processes obey the same breaker
        if flat:
            self.flatten_all(why or "safety_breaker")
            return
        if not allow:
            for st in self.states.values():
                self.cancel(st, why or "safety_pause")
        if NG.ENABLED:
            blocked, nwhy = NG.blocked(now, self.news)
            if blocked:
                allow = False
                self.flatten_all(nwhy or "news")
            elif (nxt := NG.next_window(now, self.news)) and nxt - now <= 300:
                for st in self.states.values():
                    self.cancel(st, "news_pending")
        if self.ledger:
            for v in self.ledger.pending_closes(self.owner):
                self.close_displaced(v["key"], by=v.get("requested_for", "?"))
            self.ledger.sync(self.owner, {n: (n, st.ticket, st.entry_ts, st.risk_frac)
                                          for n, st in self.states.items() if st.ticket})

        for name in self.order:
            st, s = self.states[name], self.strat[name]
            day = now // 86400
            if st.day != day:
                if st.ticket:
                    day = st.day                     # never roll a day that still holds risk; keep managing it
                else:
                    st.reset(day)
            arming = not st.armed
            raw = self.feed.bars(s.symbol, days=s.history_days if arming else 2, include_forming=True)
            bars = Bars.from_arrays(s.symbol, *raw) if raw is not None and len(raw[0]) else None

            if not st.armed:
                if not allow or now < s.signal_time(day):
                    continue
                age = now - int(bars.ts[-1]) if bars is not None else float("inf")
                if age > self.cfg.max_bar_age_s:     # stale feed / market shut: retry, then give up after an hour
                    if now - s.signal_time(day) > 3600:
                        st.armed = st.finished = True
                        self.event(kind="no_data", strategy=name, bar_age=age)
                    continue
                st.armed = True
                try:
                    st.plan = s.arm(bars, day)
                except Exception as e:
                    log.exception("[%s] arming failed", name)
                    self.event(kind="arm_error", strategy=name, error=str(e))
                    st.finished = True
                    continue
                if st.plan is None:
                    self.event(kind="no_signal", strategy=name, day=day)
                    st.finished = True
                    continue
                self.event(kind="armed", strategy=name, plan=st.plan.__dict__)
                if self.adopt(st):                   # a restart must not give the day a second bite
                    if self.admit_fill(st, now):
                        self.manage(st, bars, now)
                    continue
                if self.traded_today(st, now):
                    st.finished = True
                    self.event(kind="already_traded", strategy=name)
                    continue
                self.place(st, bars, now)
                continue

            if st.finished and not st.ticket or st.plan is None or bars is None:
                continue
            if st.close_due and st.ticket:
                if self.adopt(st):
                    self.settle(st, st.close_due, self.brokers[name].close_ticket(s.symbol, st.ticket))
                else:
                    self.book(st, st.close_due)      # its broker stop/target took it meanwhile
                continue
            had = st.ticket
            if self.adopt(st):
                if st.orders:
                    self.cancel(st, "sibling filled")   # one-cancels-other
                if st.ticket != had and not self.admit_fill(st, now):
                    continue
                self.manage(st, bars, now)
                continue
            if had:
                self.cancel(st, "position closed")
                self.book(st, "broker")              # the stop or target did the work
                continue
            if st.orders and now >= st.plan.expiry_ts:
                self.cancel(st, "expired")
                st.finished = True

    def run_forever(self) -> None:
        b0 = next(iter(self.brokers.values()))
        while True:
            try:
                if hasattr(b0, "ensure_connected"):
                    b0.ensure_connected()            # a restarted terminal drops the IPC link for good
                self.run_once()
            except Exception:
                log.exception("cycle error")
            now = datetime.now(timezone.utc)
            nxt = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
            time.sleep(max(1.0, (nxt - now).total_seconds()))


def main() -> None:
    from tradebot.broker import MT5Broker, SimBroker
    from tradebot.feed import MT5Feed
    from tradebot.ops import AlreadyRunning, Notifier, TradeLog, acquire_singleton, keep_awake
    from tradebot.strategies import default_strategies

    ap = argparse.ArgumentParser(description="multi-strategy MT5 trading engine")
    ap.add_argument("--paper", action="store_true", help="simulated fills on the live MT5 feed")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()
    try:
        acquire_singleton(cfg.log_dir / "runtime.lock")
    except AlreadyRunning as e:
        log.error("not starting: %s", e)
        raise SystemExit(3)                          # a supervisor stops on 3 instead of restarting forever
    keep_awake()
    strategies = default_strategies()
    feed = MT5Feed(log_dir=cfg.log_dir)
    if args.paper or cfg.mode != "LIVE":
        brokers = {s.name: SimBroker(feed, s.magic, cfg.capital) for s in strategies}
    else:
        brokers = {s.name: MT5Broker(s.magic, dry_run=cfg.dry_run, tag=s.name, log_dir=cfg.log_dir) for s in strategies}
        if blocked := brokers[strategies[0].name].trading_allowed():
            log.error("TRADING OFF at the terminal/account: %s — every order would be rejected", ", ".join(blocked))
    ledger = Ledger(cfg.log_dir / "ledger.json", {s.name: s.rank for s in strategies}, cfg.max_open)
    Engine(strategies, brokers, feed, cfg, ledger, TradeLog(cfg.log_dir), Notifier.from_env()).run_forever()


if __name__ == "__main__":
    main()
