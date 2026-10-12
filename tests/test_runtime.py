"""Full-day replays of the engine: synthetic 1m tape, minute-by-minute clock, simulated broker. Nothing is mocked at the
rule level — the strategy arms off the same bars it would see live, and the feed cannot see one second ahead."""
import json
from dataclasses import dataclass

import numpy as np

from conftest import DAY, breakout_up, day_bars
from tradebot.broker import SimBroker
from tradebot.feed import ReplayFeed
from tradebot.ledger import Ledger
from tradebot.runtime import Engine
from tradebot.strategies import MARKET, OpeningRangeBreakout, Plan
from tradebot.strategies.base import ny_ts


def replay(cfg, strategies, archive, minutes=(9 * 60, 16 * 60 + 30), engine=None, brokers=None, feed=None, ledger=None):
    feed = feed or ReplayFeed(archive)
    brokers = brokers or {s.name: SimBroker(feed, s.magic) for s in strategies}
    eng = engine or Engine(strategies, brokers, feed, cfg, ledger=ledger)
    for m in range(*minutes):
        feed.now = ny_ts(DAY, 0, 0) + m * 60 + 5
        eng.run_once(feed.now)
    return eng, brokers, feed


def events(cfg):
    return [json.loads(x) for x in (cfg.log_dir / "events.jsonl").read_text().splitlines()]


def test_orb_day_arms_rests_both_legs_fills_long_cancels_short_and_takes_target(cfg):
    orb = OpeningRangeBreakout()
    eng, b, _ = replay(cfg, [orb], {"US100": day_bars("US100", breakout_up)})
    ev = events(cfg)
    kinds = [e["kind"] for e in ev]
    armed = next(e for e in ev if e["kind"] == "armed")["plan"]
    assert (armed["level_up"], armed["level_dn"], armed["risk"]) == (110.0, 100.0, 10.0)
    assert kinds.count("stop_order") == 2                                 # both legs rest (one cancels the other)
    assert any(e["kind"] == "cancelled" and e["leg"] == "dn" for e in ev)
    ex = next(e for e in ev if e["kind"] == "exit")
    assert ex["reason"] == "broker" and abs(ex["r"] - 2.0) < 1e-9         # the 2R target, held AT the broker
    assert b["ORB"].balance > 10_000 and not b["ORB"].positions and not b["ORB"].pending


def test_flat_at_the_deadline_when_no_exit_is_hit(cfg):
    orb = OpeningRangeBreakout(rr=0.0)                                    # no target: only the hard flat can close it
    replay(cfg, [orb], {"US100": day_bars("US100", breakout_up)})
    ex = next(e for e in events(cfg) if e["kind"] == "exit")
    assert ex["reason"] == "time"


def test_a_restart_never_gives_the_day_a_second_bite(cfg):
    orb = OpeningRangeBreakout(rr=0.0)
    arch = {"US100": day_bars("US100", breakout_up)}
    eng, b, feed = replay(cfg, [orb], arch, minutes=(9 * 60, 11 * 60))   # filled, still open at 11:00
    assert len(b["ORB"].positions) == 1
    fresh = Engine([orb], b, feed, cfg)                                   # process restart: memory is gone
    replay(cfg, [orb], arch, minutes=(11 * 60, 16 * 60 + 30), engine=fresh, brokers=b, feed=feed)
    assert sum(1 for d in b["ORB"].deals if d["kind"] == "in") == 1      # adopted, not re-entered
    assert not b["ORB"].positions                                         # ... and still flattened on time


@dataclass
class MarketAt10:
    """A test double: market entry at 10:05 New York, long, 5-point stop."""
    name: str = "MKT"
    symbol: str = "US100"
    magic: int = 920001
    stake: float = 0.005
    rank: int = 1
    history_days: int = 2

    def signal_time(self, day):
        return ny_ts(day, 10, 5)

    def arm(self, bars, day):
        t = self.signal_time(day)
        return Plan(self.name, self.symbol, day, 1, MARKET, np.nan, np.nan, 5.0, 0.0, 0.0, t, t, ny_ts(day, 15, 55))


def test_a_market_fill_is_refused_and_closed_back_when_the_book_is_full(cfg, tmp_path):
    led = Ledger(tmp_path / "ledger.json", {"X": 9, "MKT": 1}, max_open=3)
    for i in range(3):
        led.admit("other", "X", "X%d" % i, i)                             # three higher-ranked positions elsewhere
    s = MarketAt10()
    eng, b, _ = replay(cfg, [s], {"US100": day_bars("US100", breakout_up)}, ledger=led)
    assert any(e["kind"] == "refused_fill" for e in events(cfg))
    assert not b["MKT"].positions and len(led.open_positions()) == 3      # the book never shows 4


def test_the_daily_breaker_flattens_everything(cfg):
    s = MarketAt10()
    arch = {"US100": day_bars("US100", breakout_up)}
    feed = ReplayFeed(arch)
    b = {s.name: SimBroker(feed, s.magic)}
    eng, _, _ = replay(cfg, [s], arch, minutes=(9 * 60, 10 * 60 + 10), brokers=b, feed=feed)
    assert len(b["MKT"].positions) == 1
    b["MKT"].balance -= 500                                               # -5% realised elsewhere on the account
    replay(cfg, [s], arch, minutes=(10 * 60 + 10, 10 * 60 + 12), engine=eng, brokers=b, feed=feed)
    assert not b["MKT"].positions and any(e["kind"] == "safety_flatten" for e in events(cfg))
