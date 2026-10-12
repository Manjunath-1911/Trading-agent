"""Brokers: MetaTrader 5 (live) and an in-process simulator (paper trading, tests).

Both expose the same surface the runtime uses. Order sizes are always given as RISK in account currency (the loss at
the stop); the broker converts that to lots with the terminal's own P&L math. One broker instance per strategy, each
with its own magic number, so two strategies on one symbol never see (or close) each other's positions.
"""
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("tradebot.broker")

LOT_ROUND_UP = max(float(os.getenv("LOT_ROUND_UP", "0") or 0), 0.0)   # take the next lot step if <= +x% over the stake
CLOSED_OK = ("CLOSED", "NO_POSITION", "DRY")


# ── simulator ────────────────────────────────────────────────────────────────

class SimBroker:
    """Fills resting stop orders when a bar trades through them and honours SL / TP on bar highs and lows (stop first
    when one bar touches both). Prices come from the same feed the runtime reads, so a paper run and a test replay
    see exactly what the strategy saw. 1 lot = 1 unit of price; P&L in account currency."""

    def __init__(self, feed, magic: int, balance: float = 10_000.0, login: int = 1):
        self.feed, self.magic, self.balance, self.login = feed, magic, balance, login
        self.pending, self.positions, self.closed = {}, {}, {}
        self.deals: list = []
        self._next = magic * 1000
        self._seen: dict = {}

    def _now(self) -> int:
        return int(getattr(self.feed, "now", 0) or time.time())     # a replay feed carries its own clock

    def _bars(self, symbol):
        return self.feed.bars(symbol, days=2, include_forming=True)

    def _sync(self, symbol: str) -> None:
        b = self._bars(symbol)
        if b is None:
            return
        ts, o, h, l, c, _ = b
        last = self._seen.get(symbol, -1)
        for k in range(len(ts)):
            if ts[k] <= last:
                continue
            for tkt, od in list(self.pending.items()):           # resting stops: fill on touch, at the level or the gap
                if od["symbol"] != symbol or (od["expiry"] and ts[k] >= od["expiry"]):
                    continue
                s = od["side"]
                if (s > 0 and h[k] >= od["level"]) or (s < 0 and l[k] <= od["level"]):
                    px = max(o[k], od["level"]) if s > 0 else min(o[k], od["level"])
                    self.pending.pop(tkt)
                    self._open(tkt, symbol, s, px, od["sl"], od["tp"], od["lots"], int(ts[k]))
            for tkt, p in list(self.positions.items()):           # stops first, then targets
                if p["symbol"] != symbol or ts[k] < p["open_ts"]:
                    continue
                s, sl, tp = p["side"], p["sl"], p["tp"]
                if sl and ((s > 0 and l[k] <= sl) or (s < 0 and h[k] >= sl)):
                    self._close(tkt, sl, int(ts[k]))
                elif tp and ((s > 0 and h[k] >= tp) or (s < 0 and l[k] <= tp)):
                    self._close(tkt, tp, int(ts[k]))
        if len(ts):
            # the forming bar's high/low keep moving: only bars that have CLOSED are marked as seen
            done = ts[ts + 60 <= self._now()]
            self._seen[symbol] = int(done[-1]) if len(done) else last

    def _open(self, tkt, symbol, side, px, sl, tp, lots, ts):
        self.positions[tkt] = dict(ticket=tkt, symbol=symbol, side=side, entry=px, sl=sl, tp=tp, lots=lots, open_ts=ts,
                                   direction="LONG" if side > 0 else "SHORT")
        self.deals.append(dict(kind="in", ts=ts, position_id=tkt, symbol=symbol, pnl=0.0))

    def _close(self, tkt, px, ts, lots=None):
        p = self.positions[tkt]
        lots = p["lots"] if lots is None else min(lots, p["lots"])
        pnl = p["side"] * (px - p["entry"]) * lots
        self.balance += pnl
        self.deals.append(dict(kind="out", ts=ts, position_id=tkt, symbol=p["symbol"], pnl=pnl))
        if lots >= p["lots"] - 1e-12:
            self.positions.pop(tkt)
            self.closed[tkt] = dict(pnl=self.closed.get(tkt, {}).get("pnl", 0.0) + pnl, price=px)
        else:
            p["lots"] -= lots
            self.closed.setdefault(tkt, dict(pnl=0.0, price=px))["pnl"] += pnl

    def _last(self, symbol) -> float:
        b = self._bars(symbol)
        return float(b[4][-1])

    # surface used by the runtime
    def account_balance(self):
        return self.balance

    def equity(self):
        fl = sum(p["side"] * (self._last(p["symbol"]) - p["entry"]) * p["lots"] for p in self.positions.values())
        return {"balance": self.balance, "equity": self.balance + fl, "floating": fl, "login": self.login}

    def risk_per_lot(self, symbol, entry, stop):
        return abs(entry - stop)

    def min_lots(self, symbol):
        return 0.01

    def market_order(self, symbol, side, risk_usd, stop, tp=0.0):
        self._sync(symbol)
        px = self._last(symbol)
        self._next += 1
        lots = risk_usd / max(abs(px - stop), 1e-9)
        self._open(self._next, symbol, side, px, stop, tp, lots, self._now())
        return {"status": "FILLED", "ticket": self._next, "fill_price": px}

    def stop_order(self, symbol, side, risk_usd, level, stop, tp=0.0, expiry_ts=0):
        self._sync(symbol)
        px = self._last(symbol)
        if (side > 0 and level <= px) or (side < 0 and level >= px):
            return {"status": "IMMEDIATE"}                        # already through: the backtest filled AT the level
        self._next += 1
        self.pending[self._next] = dict(symbol=symbol, side=side, level=level, sl=stop, tp=tp, expiry=expiry_ts,
                                        lots=risk_usd / max(abs(level - stop), 1e-9))
        return {"status": "PLACED", "ticket": self._next}

    def cancel_order(self, ticket):
        self.pending.pop(int(ticket), None)
        return {"status": "CANCELLED"}

    def list_positions(self, symbol=None):
        for s in {p["symbol"] for p in self.positions.values()} | ({symbol} if symbol else set()):
            self._sync(s)
        return [dict(p) for p in self.positions.values() if symbol is None or p["symbol"] == symbol]

    def list_pending_orders(self, symbol=None):
        return [dict(o, ticket=t) for t, o in self.pending.items() if symbol is None or o["symbol"] == symbol]

    def close_ticket(self, symbol, ticket):
        if int(ticket) not in self.positions:
            return {"status": "NO_POSITION"}
        self._close(int(ticket), self._last(symbol), self._now())
        return {"status": "CLOSED"}

    def modify_sltp(self, symbol, ticket, sl, tp):
        if int(ticket) in self.positions:
            self.positions[int(ticket)].update(sl=sl, tp=tp)
        return True

    def closed_result(self, ticket):
        return dict(self.closed.get(int(ticket), {"pnl": 0.0, "price": 0.0}))

    def deals_since(self, utc_from: datetime):
        t0 = utc_from.timestamp()
        return [d for d in self.deals if d["ts"] >= t0]


# ── MetaTrader 5 ─────────────────────────────────────────────────────────────

class MT5Broker:
    """Real orders through the MetaTrader5 terminal API (Windows). `dry_run` logs every order instead of sending it."""

    def __init__(self, magic: int, dry_run: bool = True, deviation: int = 20, tag: str = "bot",
                 symbol_map: Optional[dict] = None, log_dir: Optional[Path] = None):
        import MetaTrader5 as mt5                                  # Windows-only dependency, imported on demand
        from tradebot.mt5_clock import CLOCK_SYMBOLS, ServerClock
        self.mt5, self.magic, self.dry_run, self.deviation, self.tag = mt5, int(magic), dry_run, deviation, tag
        path, login = os.getenv("MT5_PATH", "").strip(), os.getenv("MT5_LOGIN", "").strip()
        kw = {"path": path} if path else {}
        # ATTACH FIRST, log in only if that fails: initialize(login=...) PERFORMS a login and a failed one drops the
        # session the terminal already had.
        if not mt5.initialize(**kw):
            if not login:
                raise RuntimeError("mt5.initialize() failed: %s (terminal running and logged in?)" % (mt5.last_error(),))
            kw.update(login=int(login), password=os.getenv("MT5_PASSWORD", ""), server=os.getenv("MT5_SERVER", ""))
            if not mt5.initialize(**kw):
                raise RuntimeError("mt5.initialize() failed: %s" % (mt5.last_error(),))
        self._init_kwargs = kw
        self.smap = dict(symbol_map or {})
        for kv in filter(None, os.getenv("MT5_SYMBOL_MAP", "").split(",")):
            k, v = kv.split("=")
            self.smap[k.strip()] = v.strip()
        self.clock = ServerClock(mt5, tuple(dict.fromkeys(CLOCK_SYMBOLS + tuple(self.smap.values()))),
                                 cache=(log_dir / "mt5_utc_offset.json") if log_dir else None)

    def ensure_connected(self) -> bool:
        """A restarted / auto-updated terminal drops the IPC link for good (every call returns None): re-initialize."""
        try:
            if self.mt5.account_info() is not None:
                return True
        except Exception:
            pass
        log.warning("MT5 connection lost — reconnecting")
        try:
            self.mt5.shutdown()
        except Exception:
            pass
        ok = bool(self.mt5.initialize(**self._init_kwargs))
        (log.info if ok else log.error)("MT5 reconnect %s", "ok" if ok else "FAILED: %s" % (self.mt5.last_error(),))
        return ok

    def _bsym(self, symbol: str) -> str:
        return self.smap.get(symbol, symbol)

    def _filling(self, bsym):
        mode = getattr(self.mt5.symbol_info(bsym), "filling_mode", 0)
        return (self.mt5.ORDER_FILLING_IOC if mode & 2 else
                self.mt5.ORDER_FILLING_FOK if mode & 1 else self.mt5.ORDER_FILLING_RETURN)

    def risk_per_lot(self, symbol: str, entry: float, stop: float) -> float:
        """Loss of a 1.00-lot stop-out from the terminal's own P&L math (tick_value specs are not always right)."""
        bsym = self._bsym(symbol)
        if self.mt5.symbol_info(bsym) is None or entry == stop:
            return 0.0
        calc = self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_BUY, bsym, 1.0, max(entry, stop), min(entry, stop))
        return abs(calc) if calc else 0.0

    def min_lots(self, symbol: str) -> float:
        return float(getattr(self.mt5.symbol_info(self._bsym(symbol)), "volume_min", 0.0) or 0.0)

    def lots_for_risk(self, symbol: str, entry: float, stop: float, risk_usd: float) -> float:
        """Lots so a stop-out loses ~risk_usd: FLOORED to the volume step (rounding to nearest oversizes the risk by up
        to half a step), optionally one step up when that stays within LOT_ROUND_UP of the stake."""
        info = self.mt5.symbol_info(self._bsym(symbol))
        rpl = self.risk_per_lot(symbol, entry, stop)
        if info is None or rpl <= 0:
            raise RuntimeError("cannot size %s" % symbol)
        step = info.volume_step or 0.01
        lots = math.floor(risk_usd / rpl / step + 1e-9) * step
        if LOT_ROUND_UP > 0 and (lots + step) * rpl <= (1 + LOT_ROUND_UP) * risk_usd:
            lots += step
        lots = max(info.volume_min, min(info.volume_max, lots))
        return round(lots, max(2, -int(math.floor(math.log10(step)))) if step < 1 else 0)

    def _send(self, req: dict) -> tuple[bool, object]:
        if self.dry_run:
            log.info("[DRY] %s", req)
            return True, None
        res = self.mt5.order_send(req)
        ok = res is not None and res.retcode == self.mt5.TRADE_RETCODE_DONE
        if not ok:
            log.error("order_send FAILED %s: %s", req.get("symbol"), getattr(res, "comment", res))
        return ok, res

    def market_order(self, symbol, side, risk_usd, stop, tp=0.0):
        bsym = self._bsym(symbol)
        self.mt5.symbol_select(bsym, True)
        info, tick = self.mt5.symbol_info(bsym), self.mt5.symbol_info_tick(bsym)
        price = tick.ask if side > 0 else tick.bid
        # spread-aware stop clamp: a BUY's stop is checked against the BID, a SELL's against the ASK; an index CFD's
        # spread can put a "valid" stop inside it (retcode 10016). Only ever widens the stop; size AFTER the clamp.
        buf = ((getattr(info, "trade_stops_level", 0) or 0) + 3) * (info.point or 0.0)
        stop = round(min(stop, tick.bid - buf) if side > 0 else max(stop, tick.ask + buf), info.digits)
        lots = self.lots_for_risk(symbol, price, stop, risk_usd)
        ok, res = self._send(dict(action=self.mt5.TRADE_ACTION_DEAL, symbol=bsym, volume=lots,
                                  type=self.mt5.ORDER_TYPE_BUY if side > 0 else self.mt5.ORDER_TYPE_SELL,
                                  price=price, sl=stop, tp=round(tp, info.digits) if tp else 0.0,
                                  deviation=self.deviation, magic=self.magic, comment=self.tag,
                                  type_time=self.mt5.ORDER_TIME_GTC, type_filling=self._filling(bsym)))
        if self.dry_run:
            return {"status": "DRY", "ticket": 0, "fill_price": price}
        return {"status": "FILLED" if ok else "REJECTED", "ticket": getattr(res, "order", 0),
                "fill_price": getattr(res, "price", 0.0) or price, "retcode": getattr(res, "retcode", -1)}

    def stop_order(self, symbol, side, risk_usd, level, stop, tp=0.0, expiry_ts=0):
        """Resting buy-stop / sell-stop: the entry breakout strategies are backtested on. IMMEDIATE when price is
        already through the level (the caller must not chase). The rule's expiry is placed AT the broker, so a dead
        agent's order still dies on time."""
        bsym = self._bsym(symbol)
        self.mt5.symbol_select(bsym, True)
        info, tick = self.mt5.symbol_info(bsym), self.mt5.symbol_info_tick(bsym)
        level = round(level, info.digits)
        if (side > 0 and level <= tick.ask) or (side < 0 and level >= tick.bid):
            return {"status": "IMMEDIATE"}
        buf = ((getattr(info, "trade_stops_level", 0) or 0) + 3) * (info.point or 0.0)
        stop = round(min(stop, level - buf) if side > 0 else max(stop, level + buf), info.digits)
        req = dict(action=self.mt5.TRADE_ACTION_PENDING, symbol=bsym, volume=self.lots_for_risk(symbol, level, stop, risk_usd),
                   type=self.mt5.ORDER_TYPE_BUY_STOP if side > 0 else self.mt5.ORDER_TYPE_SELL_STOP, price=level,
                   sl=stop, tp=round(tp, info.digits) if tp else 0.0, deviation=self.deviation, magic=self.magic,
                   comment=self.tag, type_time=self.mt5.ORDER_TIME_DAY, type_filling=self._filling(bsym))
        if expiry_ts and getattr(info, "expiration_mode", 0) & 4:
            req.update(type_time=self.mt5.ORDER_TIME_SPECIFIED, expiration=int(self.clock.to_server(expiry_ts)))
        ok, res = self._send(req)
        if self.dry_run:
            return {"status": "DRY", "ticket": 0}
        return {"status": "PLACED" if ok else "REJECTED", "ticket": getattr(res, "order", 0)}

    def list_positions(self, symbol=None, any_magic=False):
        raw = self.mt5.positions_get(symbol=self._bsym(symbol)) if symbol else self.mt5.positions_get()
        if raw is None:      # an ERROR (lost IPC, unknown symbol) — never "flat": that is the empty tuple
            raise RuntimeError("positions_get failed: %s" % (self.mt5.last_error(),))
        inv = {v: k for k, v in self.smap.items()}
        return [dict(ticket=p.ticket, symbol=inv.get(p.symbol, p.symbol), lots=p.volume, entry=p.price_open, sl=p.sl,
                     tp=p.tp, side=1 if p.type == self.mt5.POSITION_TYPE_BUY else -1, magic=p.magic,
                     direction="LONG" if p.type == self.mt5.POSITION_TYPE_BUY else "SHORT",
                     open_ts=int(self.clock.to_utc(int(p.time))))
                for p in raw if any_magic or p.magic == self.magic]

    def list_pending_orders(self, symbol=None):
        raw = self.mt5.orders_get(symbol=self._bsym(symbol)) if symbol else self.mt5.orders_get()
        if raw is None:
            raise RuntimeError("orders_get failed: %s" % (self.mt5.last_error(),))
        return [dict(ticket=o.ticket, symbol=o.symbol) for o in raw if o.magic == self.magic]

    def cancel_order(self, ticket):
        ok, _ = self._send(dict(action=self.mt5.TRADE_ACTION_REMOVE, order=int(ticket)))
        return {"status": "DRY" if self.dry_run else ("CANCELLED" if ok else "REJECTED")}

    def close_ticket(self, symbol, ticket, volume=None):
        pos = [p for p in self.list_positions(symbol) if p["ticket"] == int(ticket)]
        if not pos:
            return {"status": "NO_POSITION"}
        p, bsym = pos[0], self._bsym(symbol)
        tick = self.mt5.symbol_info_tick(bsym)
        ok, _ = self._send(dict(action=self.mt5.TRADE_ACTION_DEAL, symbol=bsym, position=int(ticket),
                                volume=p["lots"] if volume is None else min(volume, p["lots"]),
                                type=self.mt5.ORDER_TYPE_SELL if p["side"] > 0 else self.mt5.ORDER_TYPE_BUY,
                                price=tick.bid if p["side"] > 0 else tick.ask, deviation=self.deviation,
                                magic=self.magic, comment=self.tag + " close", type_time=self.mt5.ORDER_TIME_GTC,
                                type_filling=self._filling(bsym)))
        return {"status": "DRY" if self.dry_run else ("CLOSED" if ok else "REJECTED")}

    def modify_sltp(self, symbol, ticket, sl, tp):
        ok, _ = self._send(dict(action=self.mt5.TRADE_ACTION_SLTP, symbol=self._bsym(symbol), position=int(ticket),
                                sl=float(sl), tp=float(tp)))
        return ok

    def closed_result(self, ticket):
        deals = self.mt5.history_deals_get(position=int(ticket)) or ()
        outs = [d for d in deals if d.entry in (self.mt5.DEAL_ENTRY_OUT, self.mt5.DEAL_ENTRY_OUT_BY)]
        return {"pnl": float(sum(d.profit + d.swap + d.commission for d in deals)),
                "price": float(outs[-1].price) if outs else 0.0}

    def deals_since(self, utc_from: datetime):
        frm = datetime.fromtimestamp(self.clock.to_server(utc_from.timestamp()), tz=timezone.utc)
        to = datetime.fromtimestamp(self.clock.to_server(datetime.now(timezone.utc).timestamp()) + 300, tz=timezone.utc)
        out = []
        for d in self.mt5.history_deals_get(frm, to) or ():
            if d.magic == self.magic:
                out.append(dict(kind="in" if d.entry == self.mt5.DEAL_ENTRY_IN else "out", position_id=d.position_id,
                                ts=int(self.clock.to_utc(int(d.time))), pnl=float(d.profit + d.swap + d.commission)))
        return out

    def account_balance(self):
        ai = self.mt5.account_info()
        return float(ai.balance) if ai else float("nan")

    def equity(self):
        ai = self.mt5.account_info()
        return {} if ai is None else {"balance": float(ai.balance), "equity": float(ai.equity),
                                      "floating": float(ai.profit), "login": int(ai.login)}

    def account_mode(self) -> str:
        return {0: "DEMO", 1: "CONTEST", 2: "REAL"}.get(getattr(self.mt5.account_info(), "trade_mode", None), "UNKNOWN")

    def trading_allowed(self) -> list:
        """Permissions that would make every order fail: [] = all good."""
        ti, ai = self.mt5.terminal_info(), self.mt5.account_info()
        return [k for k, v in (("terminal Algo Trading", getattr(ti, "trade_allowed", False)),
                               ("account trading", getattr(ai, "trade_allowed", False)),
                               ("account expert trading", getattr(ai, "trade_expert", False))) if not v]
