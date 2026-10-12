from datetime import datetime, timezone

from tradebot import safety as SF
from tradebot.risk import cascade_factor, governor

NOW = datetime(2025, 10, 14, 15, tzinfo=timezone.utc)


class Acct:
    def __init__(self, balance=10_000.0, equity=10_000.0, login=1, ok=True):
        self.s = {"balance": balance, "equity": equity, "login": login} if ok else {}

    def equity(self):
        return self.s


def test_daily_breaker_trips_on_floating_loss_and_clears_next_day(cfg):
    st = SF.SafetyState()
    assert SF.check(st, [Acct()], NOW, cfg)[0]
    ok, why, flat = SF.check(st, [Acct(equity=9_600)], NOW, cfg)                 # -4.0% floating
    assert (ok, why, flat) == (False, "daily_breaker", True)
    nxt = NOW.replace(day=15)
    assert SF.check(st, [Acct(balance=9_600, equity=9_600)], nxt, cfg)[0]        # new server day, new budget


def test_overall_breaker_is_permanent_and_survives_a_restart(cfg, tmp_path):
    st = SF.SafetyState(initial_balance=10_000)
    assert SF.check(st, [Acct(balance=9_000, equity=9_000)], NOW, cfg)[1] == "overall_breaker"
    SF.save_state(st, tmp_path / "s.json")
    st2 = SF.load_state(tmp_path / "s.json", 10_000)
    assert SF.check(st2, [Acct(balance=11_000, equity=11_000)], NOW.replace(day=20), cfg)[0] is False


def test_a_new_account_gets_a_fresh_budget(cfg):
    st = SF.SafetyState(day="2025-10-14", day_start_balance=10_800, initial_balance=10_000, halted=True,
                        reason="overall_breaker", login=111)
    ok, _, _ = SF.check(st, [Acct(login=222)], NOW, cfg)
    assert ok and st.login == 222 and st.day_start_balance == 10_000


def test_missing_equity_pauses_entries_but_never_liquidates(cfg):
    assert SF.check(SF.SafetyState(), [Acct(ok=False)], NOW, cfg) == (False, "equity_unavailable", False)


def test_first_attach_with_open_positions_refuses_to_invent_a_day_anchor(cfg):
    assert SF.check(SF.SafetyState(), [Acct()], NOW, cfg, has_open_positions=True)[1] == "day_anchor_unavailable"


def test_corrupt_state_fails_closed(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert SF.load_state(p, 10_000).halted


def test_cascade_and_governor():
    assert cascade_factor(9_700, 10_000, "4:0.5,6:0.3") == 1.0
    assert cascade_factor(9_550, 10_000, "4:0.5,6:0.3") == 0.5
    assert cascade_factor(9_300, 10_000, "4:0.5,6:0.3") == 0.3
    assert cascade_factor(9_300, 10_000, "") == 1.0
    assert abs(governor(0.01, 0.025, 0.03) - 0.005) < 1e-12 and governor(0.01, 0.03, 0.03) == 0.0
