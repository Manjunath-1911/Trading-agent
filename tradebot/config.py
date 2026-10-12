"""Runtime configuration from the environment (and an optional .env file).

Everything has a safe default: with no configuration the engine runs in PAPER mode and never sends an order.
Real orders need MODE=LIVE, DRY_RUN=0 **and** a LIVE_ARMED marker file in the project root — a deliberate,
machine-local act that a copied .env alone cannot supply.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal KEY=VALUE loader; never overrides variables already set in the process environment."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


load_env()


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(int(default))).strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass
class Config:
    mode: str = field(default_factory=lambda: os.getenv("MODE", "PAPER").upper())          # PAPER | LIVE
    dry_run: bool = field(default_factory=lambda: _b("DRY_RUN", True))
    capital: float = field(default_factory=lambda: _f("CAPITAL", 10_000))
    initial_balance: float = field(default_factory=lambda: _f("INITIAL_BALANCE", _f("CAPITAL", 10_000)))
    # account-level loss limits (equity, incl. floating P&L) — set under the prop firm's own limits
    max_daily_loss_pct: float = field(default_factory=lambda: _f("MAX_DAILY_LOSS_PCT", 3.75))
    max_overall_loss_pct: float = field(default_factory=lambda: _f("MAX_OVERALL_LOSS_PCT", 9.5))
    # drawdown cascade: stake x0.5 below -4% of the starting balance, x0.3 below -6%
    dd_cascade: str = field(default_factory=lambda: os.getenv("DD_CASCADE", "4:0.5,6:0.3"))
    collective_cap: float = field(default_factory=lambda: _f("RISK_CAP", 0.03))   # max open risk, fraction of equity
    max_open: int = field(default_factory=lambda: int(os.getenv("MAX_OPEN", "3")))
    max_market_lag_s: int = field(default_factory=lambda: int(os.getenv("MAX_MARKET_LAG", "120")))
    max_bar_age_s: int = field(default_factory=lambda: int(os.getenv("MAX_BAR_AGE", "300")))
    max_risk_overshoot: float = field(default_factory=lambda: _f("MAX_RISK_OVERSHOOT", 1.5))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", str(ROOT / "logs"))))
    gate_forced_dry: bool = False

    def __post_init__(self) -> None:
        if self.mode == "LIVE" and not self.dry_run and not (ROOT / "LIVE_ARMED").exists():
            self.dry_run, self.gate_forced_dry = True, True        # fail safe: monitor, never send
