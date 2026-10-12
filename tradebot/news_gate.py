"""Red-folder news blackout: flat from T-15 to T+15 minutes around high-impact releases (optional, NEWS_GATE=1).

Many prop firms void trades opened or closed within minutes of a high-impact release. The calendar comes from the
public ForexFactory JSON feed (this week only), merged into a rolling on-disk cache. FAILS CLOSED: a cache that does not
cover `now` blocks new entries — a stale calendar cannot prove a release is absent.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.request
from pathlib import Path

FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
PRE_S = int(float(os.getenv("NEWS_PRE_MIN", "15")) * 60)
POST_S = int(float(os.getenv("NEWS_POST_MIN", "15")) * 60)
CURRENCIES = tuple(c.strip().upper() for c in os.getenv("NEWS_CCY", "USD").split(",") if c.strip())
ENABLED = os.getenv("NEWS_GATE", "0") == "1"


def _parse(ev) -> tuple | None:
    if str(ev.get("impact", "")).lower() != "high" or not ev.get("date"):
        return None
    try:
        t = dt.datetime.fromisoformat(str(ev["date"]))     # ISO WITH the New York offset: exact, no local tz involved
    except ValueError:
        return None
    return None if t.tzinfo is None else (int(t.timestamp()), str(ev.get("country", "")).upper(), str(ev.get("title", "")))


def read(cache: Path) -> list:
    try:
        return [tuple(x) for x in json.loads(cache.read_text(encoding="utf-8")).get("events", [])]
    except (FileNotFoundError, ValueError):
        return []


def refresh(cache: Path, timeout: int = 25) -> int:
    """Fetch this week's feed and MERGE it into the cache (keyed on time, currency, title)."""
    have = set(read(cache))
    with urllib.request.urlopen(urllib.request.Request(FEED, headers={"User-Agent": "tradebot/1.0"}), timeout=timeout) as r:
        have |= {p for p in map(_parse, json.load(r)) if p}
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps({"events": sorted(map(list, have))}), encoding="utf-8")
    os.replace(tmp, cache)
    return len(have)


def blocked(now: int, events: list) -> tuple[bool, str | None]:
    if not events:
        return True, "news_calendar_missing"
    if now > events[-1][0]:
        return True, "news_calendar_stale"
    for ts, ccy, title in events:
        if ccy in CURRENCIES and ts - PRE_S <= now <= ts + POST_S:
            return True, "news: %s %s" % (ccy, title)
    return False, None


def next_window(now: int, events: list) -> int:
    """Start of the next blackout at or after `now` (0 = none): resting entries are pulled before it opens."""
    return next((ts - PRE_S for ts, ccy, _ in events if ccy in CURRENCIES and ts - PRE_S >= now), 0)
