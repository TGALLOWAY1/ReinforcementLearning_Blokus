"""Per-move time policy for the served Pentobi agent (D-019): a hard cap per
move, no search when only one move exists, and an optional per-game budget
that steps the level down to a floor instead of timing out."""

import pytest

from agents.time_budget import PentobiMoveBudget


def test_one_legal_move_needs_no_search():
    budget = PentobiMoveBudget(level=7, cap_ms=10_000)
    decision = budget.decide(legal_move_count=1, pieces_left=5)
    assert decision.search is False
    assert decision.tier == "trivial"
    assert decision.cap_ms == 0
    assert "one_legal_move" in decision.reasons
    assert decision.level == 7


def test_normal_move_gets_the_configured_level_under_the_cap():
    budget = PentobiMoveBudget(level=7, cap_ms=10_000)
    decision = budget.decide(legal_move_count=240, pieces_left=21)
    assert decision.search is True
    assert decision.tier == "normal"
    assert decision.cap_ms == 10_000
    assert decision.level == 7
    assert decision.reasons == []


def test_ledger_tracks_spent_time_without_a_game_budget():
    budget = PentobiMoveBudget(level=5, cap_ms=10_000)
    budget.record(1_200)
    budget.record(800)
    assert budget.spent_ms == 2_000
    assert budget.remaining_ms is None
    snap = budget.snapshot()
    assert snap == {"level": 5, "capMs": 10_000, "gameBudgetMs": None,
                    "gameSpentMs": 2_000, "gameRemainingMs": None}


def test_exhausted_game_budget_steps_down_to_the_floor_level():
    budget = PentobiMoveBudget(level=8, cap_ms=10_000, game_budget_ms=30_000, floor_level=3)
    budget.record(19_000)
    assert budget.remaining_ms == 11_000   # one full-cap move still affordable
    assert budget.decide(legal_move_count=50, pieces_left=10).level == 8
    budget.record(11_000)
    assert budget.remaining_ms == 0
    decision = budget.decide(legal_move_count=50, pieces_left=10)
    assert decision.level == 3
    assert decision.tier == "budget_exhausted"
    assert "game_budget_exhausted" in decision.reasons
    assert decision.cap_ms == 10_000


def test_budget_below_one_full_cap_steps_down_instead_of_shrinking_the_cap():
    """A shrunken cap would guarantee a timeout fallback on the boundary move
    (a loss attributable to the harness); stepping down keeps a real search."""
    budget = PentobiMoveBudget(level=6, cap_ms=10_000, game_budget_ms=25_000, floor_level=2)
    budget.record(5_000)
    normal = budget.decide(legal_move_count=50, pieces_left=10)
    assert normal.level == 6 and normal.cap_ms == 10_000 and normal.tier == "normal"
    budget.record(11_000)  # remaining 9 000 < cap
    decision = budget.decide(legal_move_count=50, pieces_left=10)
    assert decision.level == 2 and decision.cap_ms == 10_000
    assert decision.tier == "budget_exhausted" and "game_budget_below_cap" in decision.reasons


@pytest.mark.parametrize("kwargs", [
    {"level": 0}, {"level": 10}, {"level": 5, "cap_ms": 0},
    {"level": 5, "floor_level": 7}, {"level": 5, "game_budget_ms": 0},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PentobiMoveBudget(**kwargs)
