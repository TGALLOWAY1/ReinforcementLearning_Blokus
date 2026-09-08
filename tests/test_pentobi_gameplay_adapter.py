"""Natively served Pentobi agent (M4): the web backend builds it through the
same factory as every other gameplay agent, enforces the per-move cap with a
deterministic fallback, and reproduces the arena agent's moves."""

import threading
import time

import pytest

from agents import pentobi_agent as pa
from agents.greedy_agent import GreedyAgent
from engine.board import Board, Player
from engine.move_generator import LegalMoveGenerator
from schemas.game_state import AgentType
from webapi.gameplay_agent_factory import PentobiGameplayAdapter, build_deploy_gameplay_agent

GEN = LegalMoveGenerator()
needs_binary = pytest.mark.skipif(pa.find_binary() is None, reason="pentobi-gtp not installed")


def _place(board, move, player):
    board.place_piece(move.get_positions(GEN.piece_orientations_cache[move.piece_id]), player, move.piece_id)


def test_pentobi_is_a_gameplay_agent_type():
    assert AgentType("pentobi") is AgentType.PENTOBI


@needs_binary
def test_factory_builds_a_served_pentobi_adapter_that_plays_a_legal_move():
    adapter = build_deploy_gameplay_agent(AgentType.PENTOBI, {"level": 3, "seed": 5})
    assert isinstance(adapter, PentobiGameplayAdapter)
    board = Board()
    legal = GEN.get_legal_moves(board, Player.RED)
    move, stats = adapter.choose_move(board, Player.RED, legal, 10_000)
    assert move in legal
    assert stats["engine"] == "pentobi" and stats["level"] == 3
    assert stats["budgetTier"] == "normal" and stats["fallback"] is None
    assert stats["timeBudgetMs"] == 10_000 and 0 <= stats["timeSpentMs"] <= 10_000
    assert stats["pentobiVersion"] == "30.3"
    assert stats["topMoves"] == [] and stats["nodesEvaluated"] == 0
    assert stats["gameBudget"]["gameSpentMs"] == stats["timeSpentMs"]
    adapter.close()


@needs_binary
def test_backend_budget_argument_tightens_the_cap():
    adapter = build_deploy_gameplay_agent(AgentType.PENTOBI, {"level": 1, "seed": 5})
    board = Board()
    legal = GEN.get_legal_moves(board, Player.RED)
    _, stats = adapter.choose_move(board, Player.RED, legal, 500)
    assert stats["timeBudgetMs"] == 500
    adapter.close()


def test_single_legal_move_is_played_without_starting_the_engine():
    adapter = PentobiGameplayAdapter(level=9, seed=1)
    board = Board()
    legal = GEN.get_legal_moves(board, Player.RED)[:1]
    move, stats = adapter.choose_move(board, Player.RED, legal, 10_000)
    assert move == legal[0]
    assert stats["budgetTier"] == "trivial" and stats["fallback"] is None
    assert adapter.agent._engine is None


def test_no_legal_moves_returns_none():
    adapter = PentobiGameplayAdapter(level=9, seed=1)
    move, stats = adapter.choose_move(Board(), Player.RED, [], 10_000)
    assert move is None and stats["budgetTier"] == "trivial"


def test_missing_binary_fails_at_build_time(monkeypatch):
    monkeypatch.setattr(pa, "find_binary", lambda explicit=None: None)
    with pytest.raises(ValueError, match="pentobi-gtp"):
        build_deploy_gameplay_agent(AgentType.PENTOBI, {"level": 3})


def test_level_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        PentobiGameplayAdapter(level=10, seed=1)


def test_timeout_falls_back_to_the_greedy_move_within_the_cap(monkeypatch):
    adapter = PentobiGameplayAdapter(level=3, seed=1, cap_ms=300)
    released = threading.Event()
    aborted = []

    def slow_select(board, player, legal_moves):
        released.wait(5.0)
        return legal_moves[-1]

    monkeypatch.setattr(adapter.agent, "select_action", slow_select)
    monkeypatch.setattr(adapter.agent, "abort", lambda: aborted.append(True) or released.set())
    board = Board()
    legal = GEN.get_legal_moves(board, Player.RED)
    t0 = time.perf_counter()
    move, stats = adapter.choose_move(board, Player.RED, legal, 10_000)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert move == GreedyAgent(seed=0).select_action(board, Player.RED, legal)
    assert stats["fallback"] == "timeout" and aborted == [True]
    assert stats["timeBudgetMs"] == 300


def test_engine_error_falls_back_to_the_greedy_move_and_resets(monkeypatch):
    adapter = PentobiGameplayAdapter(level=3, seed=1)
    aborted = []

    def broken(board, player, legal_moves):
        raise pa.GtpError("rules disagreement")

    monkeypatch.setattr(adapter.agent, "select_action", broken)
    monkeypatch.setattr(adapter.agent, "abort", lambda: aborted.append(True))
    board = Board()
    legal = GEN.get_legal_moves(board, Player.RED)
    move, stats = adapter.choose_move(board, Player.RED, legal, 10_000)
    assert move == GreedyAgent(seed=0).select_action(board, Player.RED, legal)
    assert stats["fallback"] == "engine_error" and "rules disagreement" in stats["fallbackDetail"]
    assert aborted == [True]


@needs_binary
def test_exhausted_game_budget_drops_to_the_floor_level():
    # Budget affords exactly one full-cap move; the next move must step down
    # (never a shrunken cap that would force a timeout fallback).
    adapter = PentobiGameplayAdapter(level=3, seed=2, game_budget_ms=10_005, floor_level=1)
    board = Board()
    legal = GEN.get_legal_moves(board, Player.RED)
    move, stats = adapter.choose_move(board, Player.RED, legal, 10_000)
    assert move in legal and stats["level"] == 3 and stats["timeBudgetMs"] == 10_000
    legal = GEN.get_legal_moves(board, Player.BLUE)
    _place(board, move, Player.RED)
    legal = GEN.get_legal_moves(board, Player.BLUE)
    move, stats = adapter.choose_move(board, Player.BLUE, legal, 10_000)
    assert stats["level"] == 1 and stats["budgetTier"] == "budget_exhausted"
    assert adapter.agent.level == 1
    adapter.close()


@needs_binary
def test_served_adapter_reproduces_the_arena_agent_on_50_positions():
    """M4 gate: the agent a human plays is the agent the harness measured."""
    from analytics.tournament.arena_runner import AgentConfig, build_agent

    served = build_deploy_gameplay_agent(AgentType.PENTOBI, {"level": 3, "seed": 11})
    arena = build_agent(AgentConfig.from_dict({"name": "pentobi_l3", "type": "pentobi", "params": {"level": 3}}), seed=11)
    board = Board()
    player = Player.RED
    compared = 0
    passes = 0
    while compared < 50 and passes < 4:
        legal = GEN.get_legal_moves(board, player)
        if not legal:
            passes += 1
            player = Player((player.value % 4) + 1)
            continue
        passes = 0
        arena_move, _ = arena.choose_move(board, player, legal, thinking_time_ms=10_000)
        served_move, stats = served.choose_move(board, player, legal, 10_000)
        assert stats["fallback"] is None
        assert (served_move.piece_id, served_move.orientation, served_move.anchor_row, served_move.anchor_col) == \
            (arena_move.piece_id, arena_move.orientation, arena_move.anchor_row, arena_move.anchor_col)
        _place(board, served_move, player)
        compared += 1
        player = Player((player.value % 4) + 1)
    assert compared == 50
    served.close()
    arena.close()
