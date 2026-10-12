"""Strategy registry. The private book registers its own strategies here; the public repo ships one example."""
from tradebot.strategies.base import MARKET, STOP, Plan, Strategy
from tradebot.strategies.opening_range import OpeningRangeBreakout

__all__ = ["MARKET", "STOP", "Plan", "Strategy", "OpeningRangeBreakout", "default_strategies"]


def default_strategies() -> list:
    return [OpeningRangeBreakout()]
