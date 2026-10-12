"""Position sizing helpers: the drawdown cascade and the collective open-risk governor."""
from __future__ import annotations


def cascade_levels(spec: str) -> list[tuple[float, float]]:
    """"4:0.5,6:0.3" -> [(6.0, 0.3), (4.0, 0.5)] (deepest first). "" or "0" = off."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if part and part != "0":
            dd, m = part.split(":")
            out.append((float(dd), float(m)))
    return sorted(out, reverse=True)


def cascade_factor(balance: float, initial: float, spec: str) -> float:
    """Stake multiplier while the balance is below the STARTING balance (the prop firm's static floor hangs off it).
    Funded accounts mostly breach through slow drawdowns, not single bad days: cutting size on the way down is what
    keeps a bad month from becoming a breached account."""
    if not (initial and initial > 0 and balance == balance and balance > 0):
        return 1.0
    dd = 100.0 * (initial - balance) / initial
    for lvl, m in cascade_levels(spec):
        if dd >= lvl:
            return m
    return 1.0


def governor(want: float, committed: float, cap: float) -> float:
    """SCALE-IN mode: an entry that would breach the open-risk cap takes whatever room is left; 0.0 = no room."""
    free = cap - committed
    return 0.0 if free <= 1e-9 else min(want, free)
