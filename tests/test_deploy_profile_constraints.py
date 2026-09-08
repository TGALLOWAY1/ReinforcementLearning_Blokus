import pytest
from fastapi import HTTPException

from schemas.game_state import AgentType, GameConfig, Player, PlayerConfig
from mcts.champion_profile import CHALLENGE_CHAMPION_PROFILE
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


def test_deploy_allows_challenge_champion_profile():
    cfg = GameConfig(
        players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.MCTS, agent_config={"profile": CHALLENGE_CHAMPION_PROFILE}),
            PlayerConfig(player=Player.GREEN, agent_type=AgentType.MCTS, agent_config={"profile": CHALLENGE_CHAMPION_PROFILE}),
            PlayerConfig(player=Player.YELLOW, agent_type=AgentType.MCTS, agent_config={"profile": CHALLENGE_CHAMPION_PROFILE}),
        ],
        auto_start=True,
    )

    normalized = normalize_deploy_game_config(cfg)

    for player_cfg in normalized.players:
        if player_cfg.agent_type == AgentType.MCTS:
            assert player_cfg.agent_config["profile"] == CHALLENGE_CHAMPION_PROFILE
            assert player_cfg.agent_config["time_budget_ms"] == 30000


def test_deploy_rejects_mixed_challenge_and_difficulty_modes():
    cfg = _valid_deploy_config()
    cfg.players[1].agent_config = {"profile": CHALLENGE_CHAMPION_PROFILE}

    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(cfg)
    assert exc.value.status_code == 400
    assert "cannot mix" in str(exc.value.detail)


# --- natively served Pentobi (M4) --------------------------------------------
def _pentobi_config(level=3, extra=None) -> GameConfig:
    cfg = {"level": level}
    cfg.update(extra or {})
    return GameConfig(
        players=[
            PlayerConfig(player=Player.RED, agent_type=AgentType.HUMAN, agent_config={}),
            PlayerConfig(player=Player.BLUE, agent_type=AgentType.PENTOBI, agent_config=dict(cfg)),
            PlayerConfig(player=Player.GREEN, agent_type=AgentType.PENTOBI, agent_config=dict(cfg)),
            PlayerConfig(player=Player.YELLOW, agent_type=AgentType.PENTOBI, agent_config=dict(cfg)),
        ],
        auto_start=True,
    )


def test_deploy_accepts_pentobi_seats_and_defaults_the_cap():
    from webapi.deploy_validation import PENTOBI_TIME_CAP_MS

    normalized = normalize_deploy_game_config(_pentobi_config(level=7))
    for p in normalized.players[1:]:
        assert p.agent_type == AgentType.PENTOBI
        assert p.agent_config["level"] == 7
        assert p.agent_config["time_budget_ms"] == PENTOBI_TIME_CAP_MS


def test_deploy_rejects_pentobi_level_out_of_range():
    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(_pentobi_config(level=10))
    assert exc.value.status_code == 400
    assert "level" in str(exc.value.detail)


def test_deploy_rejects_pentobi_cap_over_the_human_protocol_cap():
    from webapi.deploy_validation import PENTOBI_TIME_CAP_MS

    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(_pentobi_config(level=3, extra={"time_budget_ms": PENTOBI_TIME_CAP_MS + 1}))
    assert exc.value.status_code == 400
    assert "cap" in str(exc.value.detail)


def test_deploy_keeps_pentobi_seed_and_game_budget():
    normalized = normalize_deploy_game_config(
        _pentobi_config(level=5, extra={"seed": 42, "game_budget_ms": 120000, "time_budget_ms": 8000}))
    cfg = normalized.players[1].agent_config
    assert cfg == {"level": 5, "seed": 42, "game_budget_ms": 120000, "time_budget_ms": 8000}


def test_pentobi_seed_must_be_an_integer():
    with pytest.raises(HTTPException) as exc:
        normalize_deploy_game_config(_pentobi_config(level=3, extra={"seed": "abc"}))
    assert exc.value.status_code == 400 and "seed" in str(exc.value.detail)
    normalized = normalize_deploy_game_config(_pentobi_config(level=3, extra={"seed": "42"}))
    assert normalized.players[1].agent_config["seed"] == 42


def test_pentobi_binary_and_book_are_never_taken_from_the_request():
    normalized = normalize_deploy_game_config(
        _pentobi_config(level=3, extra={"binary": "/tmp/evil", "use_book": True}))
    cfg = normalized.players[1].agent_config
    assert "binary" not in cfg and "use_book" not in cfg


def test_shared_pentobi_seat_normaliser_validates_budget_fields():
    from webapi.deploy_validation import PENTOBI_TIME_CAP_MS, normalize_pentobi_agent_config

    assert normalize_pentobi_agent_config({}) == {"level": 3, "time_budget_ms": PENTOBI_TIME_CAP_MS}
    assert normalize_pentobi_agent_config({"level": "5", "game_budget_ms": 60000, "floor_level": 2}) == {
        "level": 5, "game_budget_ms": 60000, "floor_level": 2, "time_budget_ms": PENTOBI_TIME_CAP_MS}
    for bad in ({"game_budget_ms": 0}, {"game_budget_ms": "x"}, {"level": 3, "floor_level": 4},
                {"floor_level": 0}, {"level": 10}, {"time_budget_ms": PENTOBI_TIME_CAP_MS + 1}):
        with pytest.raises(HTTPException) as exc:
            normalize_pentobi_agent_config(bad)
        assert exc.value.status_code == 400
