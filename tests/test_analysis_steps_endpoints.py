"""
Test StrategyLogger read-only analytics endpoints.
"""

import asyncio
import os

import pytest

from webapi.app import get_analysis_steps, get_analysis_summary


@pytest.fixture(autouse=True)
def _strategy_log_dir(monkeypatch):
    """Point to verification_run logs (flat layout with game_001)."""
    monkeypatch.setenv("STRATEGY_LOG_DIR", "logs/verification_run")


def test_get_analysis_steps():
    """GET /api/analysis/{game_id}/steps returns paginated StepLog entries."""
    result = asyncio.run(get_analysis_steps("game_001", limit=5, offset=0))
    assert "game_id" in result
    assert result["game_id"] == "game_001"
    assert "steps" in result
    assert "total" in result
    assert "limit" in result
    assert "offset" in result
    assert result["limit"] == 5
    assert result["offset"] == 0
    if result["total"] > 0:
        assert len(result["steps"]) <= 5
        step = result["steps"][0]
        assert "turn_index" in step
        assert "player_id" in step
        assert "action" in step
        assert "legal_moves_before" in step
        assert "legal_moves_after" in step
        assert "metrics" in step


def test_get_analysis_steps_pagination():
    """Steps endpoint paginates correctly."""
    full = asyncio.run(get_analysis_steps("game_001", limit=1000, offset=0))
    if full["total"] == 0:
        pytest.skip("No steps in logs")
    page1 = asyncio.run(get_analysis_steps("game_001", limit=3, offset=0))
    page2 = asyncio.run(get_analysis_steps("game_001", limit=3, offset=3))
    assert len(page1["steps"]) <= 3
    assert len(page2["steps"]) <= 3
    if page1["steps"] and page2["steps"]:
        assert page1["steps"][0]["turn_index"] != page2["steps"][0]["turn_index"] or \
               page1["steps"][0]["player_id"] != page2["steps"][0]["player_id"]


def test_get_analysis_summary():
    """GET /api/analysis/{game_id}/summary returns mobility curve and deltas."""
    result = asyncio.run(get_analysis_summary("game_001"))
    assert "game_id" in result
    assert result["game_id"] == "game_001"
    assert "mobilityCurve" in result
    assert "deltas" in result
    assert "totalSteps" in result
    if result["totalSteps"] > 0:
        assert len(result["mobilityCurve"]) == result["totalSteps"]
        assert len(result["deltas"]) == result["totalSteps"]
        curve = result["mobilityCurve"][0]
        assert "turn_index" in curve
        assert "player_id" in curve
        assert "legal_moves_before" in curve
        assert "legal_moves_after" in curve


def test_get_analysis_steps_missing_game():
    """Steps for non-existent game returns empty."""
    result = asyncio.run(get_analysis_steps("nonexistent-game-xyz-999", limit=10, offset=0))
    assert result["game_id"] == "nonexistent-game-xyz-999"
    assert result["steps"] == []
    assert result["total"] == 0


def test_get_analysis_summary_missing_game():
    """Summary for non-existent game returns empty aggregates."""
    result = asyncio.run(get_analysis_summary("nonexistent-game-xyz-999"))
    assert result["game_id"] == "nonexistent-game-xyz-999"
    assert result["mobilityCurve"] == []
    assert result["deltas"] == []
    assert result["totalSteps"] == 0
