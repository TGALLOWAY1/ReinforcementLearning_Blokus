import pytest
from fastapi import HTTPException

from schemas.game_state import AgentType, GameConfig, Player, PlayerConfig
from webapi.deploy_validation import (
    DEPLOY_DIFFICULTY_TO_MS,
    DEPLOY_TIME_BUDGET_CAP_MS,
    normalize_deploy_game_config,
)


def _valid_deploy_config() -> GameConfig:
    return GameConfig(
        players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.MCTS, agent_config={"difficulty": "easy"}),
            PlayerConfig(player=Player.GREEN, agent_type=AgentType.MCTS, agent_config={"difficulty": "medium"}),
            PlayerConfig(player=Player.YELLOW, agent_type=AgentType.MCTS, agent_config={"difficulty": "hard"}),
        ],
        auto_start=True,
    )


def test_deploy_rejects_invalid_agent_types():
    cfg = GameConfig(
        players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.RANDOM, agent_config={}),
            PlayerConfig(player=Player.GREEN, agent_type=AgentType.MCTS, agent_config={"difficulty": "medium"}),
            PlayerConfig(player=Player.YELLOW, agent_type=AgentType.MCTS, agent_config={"difficulty": "hard"}),
        ],
        auto_start=True,
    )
    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(cfg)
    assert exc.value.status_code == 400
    assert "only supports agent_type values" in str(exc.value.detail)


def test_deploy_rejects_wrong_player_count():
    cfg = GameConfig(
        players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.MCTS, agent_config={"difficulty": "easy"}),
            PlayerConfig(player=Player.GREEN, agent_type=AgentType.MCTS, agent_config={"difficulty": "medium"}),
        ],
        auto_start=True,
    )
    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(cfg)
    assert exc.value.status_code == 400
    assert "exactly 4 players" in str(exc.value.detail)


def test_deploy_maps_difficulty_presets_to_time_budget():
    cfg = _valid_deploy_config()
    normalized = normalize_deploy_game_config(cfg)

    budgets = {}
    for p in normalized.players:
        if p.agent_type == AgentType.MCTS:
            budgets[p.agent_config["difficulty"]] = p.agent_config["time_budget_ms"]

    assert budgets["easy"] == DEPLOY_DIFFICULTY_TO_MS["easy"]
    assert budgets["medium"] == DEPLOY_DIFFICULTY_TO_MS["medium"]
    assert budgets["hard"] == DEPLOY_DIFFICULTY_TO_MS["hard"]


def test_deploy_rejects_budget_over_cap():
    cfg = _valid_deploy_config()
    cfg.players[1].agent_config = {"time_budget_ms": DEPLOY_TIME_BUDGET_CAP_MS + 1}

    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(cfg)
    assert exc.value.status_code == 400
    assert "exceeds deploy cap" in str(exc.value.detail)
