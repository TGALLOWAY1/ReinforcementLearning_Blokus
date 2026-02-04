"""
Tests for league 4-player match logic and Elo updates.
"""

from __future__ import annotations

import os
import tempfile
from numbers import Integral

import pytest

from agents.registry import AgentProtocol, build_baseline_agent
from league.league import League, MatchResult, build_league_agents


def test_play_match_requires_exactly_four_agents():
    """play_match must receive exactly 4 agent names and 4 agents."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        league = League(db_path=db_path)
        agents = [build_baseline_agent("random", seed=42) for _ in range(4)]
        names = [f"agent_{i}" for i in range(4)]

        with pytest.raises(ValueError, match="exactly 4"):
            league.play_match(names[:2], agents[:2], seed=42)
        with pytest.raises(ValueError, match="exactly 4"):
            league.play_match(names, agents[:3], seed=42)
        with pytest.raises(ValueError, match="exactly 4"):
            league.play_match(names * 2, agents * 2, seed=42)
    finally:
        league.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_play_match_returns_four_scores_and_individual_winner():
    """play_match returns MatchResult with 4 scores and winner = single highest scorer."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        league = League(db_path=db_path)
        agents = [build_baseline_agent("random", seed=42) for _ in range(4)]
        names = ["a", "b", "c", "d"]

        result = league.play_match(names, agents, seed=123, max_moves=500)

        assert len(result.scores) == 4
        assert set(result.scores.keys()) == {"a", "b", "c", "d"}
        # Scores may be int or numpy integer from engine
        assert all(isinstance(s, Integral) and s >= 0 for s in result.scores.values())
        # Winner is the single agent with highest score, or None if tie for first
        if result.winner is not None:
            assert result.winner in result.scores
            assert result.scores[result.winner] == max(result.scores.values())
            ties = [n for n, s in result.scores.items() if s == result.scores[result.winner]]
            assert len(ties) == 1
        assert result.moves_made >= 0
    finally:
        league.close()
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_build_league_agents_returns_three_names_without_rl():
    """build_league_agents without model returns 3 names (baselines only). With model, returns 4 (RL first)."""
    baseline_types = ["mcts", "random", "random"]
    agents, specs, ordered_4p_names = build_league_agents(
        baseline_types, seed=42, model=None, model_name=None
    )
    assert len(ordered_4p_names) == 3
    assert len(agents) == 3
    assert len(specs) == 3
    assert ordered_4p_names[0] == "mcts_baseline_0"
    assert "random_baseline_1" in ordered_4p_names
    assert "random_baseline_2" in ordered_4p_names


def test_build_league_agents_baseline_names_are_unique():
    """Baseline agents get unique names (e.g. random_baseline_0, random_baseline_1)."""
    baseline_types = ["random", "random", "random"]
    agents, specs, ordered_4p_names = build_league_agents(
        baseline_types, seed=42, model=None, model_name=None
    )
    assert len(set(ordered_4p_names)) == 3
    assert "random_baseline_0" in agents
    assert "random_baseline_1" in agents
    assert "random_baseline_2" in agents


def test_update_elo_after_4p_match_single_winner():
    """update_elo_after_4p_match records wins for winner vs each of the 3 others."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        league = League(db_path=db_path)
        for name in ["w", "x", "y", "z"]:
            league.db.add_agent(name, "test", version=None, checkpoint_path=None, initial_elo=1200.0)
        # MatchResult with a single winner
        result = MatchResult(winner="w", scores={"w": 50, "x": 40, "y": 30, "z": 20}, moves_made=100)
        league.update_elo_after_4p_match(result, seed=1)
        # Winner should have gained; others lost (we don't assert exact Elo, just no exception)
        w_id = league.db.get_agent_id("w")
        assert w_id is not None
        league.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_update_elo_after_4p_match_tie_still_updates_elo():
    """When winner is None (tie for first), we still run pairwise Elo updates; ratings change."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        league = League(db_path=db_path)
        for name in ["a", "b", "c", "d"]:
            league.db.add_agent(name, "test", version=None, checkpoint_path=None, initial_elo=1200.0)
        # a and b tie for first (10), c and d lower (8, 6). Pairwise: a,b draw; a,b beat c,d; c beats d.
        result = MatchResult(winner=None, scores={"a": 10, "b": 10, "c": 8, "d": 6}, moves_made=50)
        league.update_elo_after_4p_match(result, seed=2)
        # Ties still update Elo (we don't skip). At least some ratings changed.
        a_id = league.db.get_agent_id("a")
        c_id = league.db.get_agent_id("c")
        assert league.db.get_rating(a_id) != 1200.0 or league.db.get_rating(c_id) != 1200.0
        # a and b beat c and d, so a and b should gain and c and d should lose
        assert league.db.get_rating(a_id) > 1200.0
        assert league.db.get_rating(c_id) < 1200.0
        league.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
