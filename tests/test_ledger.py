from tradebot.ledger import Ledger

RANK = {"A": 4, "B": 3, "C": 2, "D": 1}


def L(tmp_path, **k):
    return Ledger(tmp_path / "ledger.json", RANK, **k)


def test_three_admitted_fourth_judged(tmp_path):
    lg = L(tmp_path)
    for i, s in enumerate("BCD"):
        assert lg.admit("p1", s, s, i)["status"] == "ok"
    v = lg.admit("p2", "A", "A", 9)
    assert v["status"] == "displace" and v["victim"]["key"] == "D"          # the lowest rank goes
    assert {p["key"] for p in lg.open_positions()} == {"A", "B", "C"}
    assert [c["key"] for c in lg.pending_closes("p1")] == ["D"]             # the owner closes it next cycle


def test_lowest_rank_is_refused_and_equal_rank_never_displaces(tmp_path):
    lg = L(tmp_path)
    for i, s in enumerate("ABC"):
        lg.admit("p", s, s, i)
    assert lg.admit("p", "D", "D", 9)["status"] == "refuse"
    lg2 = L(tmp_path / "x", max_open=1)
    lg2.admit("p", "B", "B1", 1)
    assert lg2.admit("p", "B", "B2", 2)["status"] == "refuse"


def test_risk_cap_scales_then_refuses(tmp_path):
    lg = L(tmp_path, risk_cap=0.02, risk_pad=1.0)
    assert lg.admit("p", "A", "A", 1, risk_frac=0.012)["risk_frac"] == 0.012
    assert abs(lg.admit("p", "B", "B", 2, risk_frac=0.012)["risk_frac"] - 0.008) < 1e-12   # cut to the room left
    v = lg.admit("p", "C", "C", 3, risk_frac=0.005)
    assert v["status"] == "refuse" and v["why"] == "risk_cap"


def test_restore_undoes_a_displacement_for_a_trade_that_never_opened(tmp_path):
    lg = L(tmp_path)
    for i, s in enumerate("BCD"):
        lg.admit("p1", s, s, i)
    v = lg.admit("p2", "A", "A", 9)
    lg.restore("p2", "A", v["victim"])
    assert {p["key"] for p in lg.open_positions()} == {"B", "C", "D"} and lg.pending_closes("p1") == []


def test_sync_drops_closed_and_adopts_unknown_positions_without_judging(tmp_path):
    lg = L(tmp_path)
    lg.admit("p", "A", "A", 1)
    lg.sync("p", {"B": ("B", 7, 2, 0.01)})
    assert [p["key"] for p in lg.open_positions()] == ["B"]


def test_halt_flag_and_heartbeat(tmp_path):
    lg = L(tmp_path)
    assert lg.breaker_age() is None
    lg.set_halt("daily_breaker", flatten=True)
    assert lg.halt() == {"reason": "daily_breaker", "flatten": True} and lg.breaker_age() < 5
    lg.set_halt(None)
    assert lg.halt() is None
