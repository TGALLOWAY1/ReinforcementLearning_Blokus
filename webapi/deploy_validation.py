"""
Validation and normalization rules for deploy-mode gameplay configuration.
"""

from __future__ import annotations

from typing import Dict

from fastapi import HTTPException

from schemas.game_state import AgentType, GameConfig, PlayerConfig

DEPLOY_TIME_BUDGET_CAP_MS = 9000
DEPLOY_DIFFICULTY_TO_MS: Dict[str, int] = {
    "easy": 200,
    "medium": 450,
    "hard": 900,
}
_DEPLOY_REQUIRED_DIFFICULTIES = {"easy", "medium", "hard"}


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _normalize_difficulty(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value not in DEPLOY_DIFFICULTY_TO_MS:
        valid = ", ".join(sorted(DEPLOY_DIFFICULTY_TO_MS.keys()))
        raise _bad_request(f"Invalid MCTS difficulty '{raw}'. Allowed values: {valid}.")
    return value


def _normalize_time_budget_ms(raw: object) -> int:
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        raise _bad_request("Invalid time budget. 'time_budget_ms' must be an integer.")

    if budget < 1:
        raise _bad_request("Invalid time budget. 'time_budget_ms' must be >= 1.")
    if budget > DEPLOY_TIME_BUDGET_CAP_MS:
        raise _bad_request(
            f"time_budget_ms={budget} exceeds deploy cap ({DEPLOY_TIME_BUDGET_CAP_MS}ms)."
        )
    return budget


def normalize_deploy_game_config(config: GameConfig) -> GameConfig:
    """
    Validate and normalize a create-game payload for APP_PROFILE=deploy.

    Rules:
    - Exactly 4 players
    - Exactly 1 human + 3 mcts
    - MCTS difficulties must be exactly easy/medium/hard (one each)
    - Non-deploy agent types are rejected
    - MCTS budgets are normalized from preset or validated against deploy cap
    """
    if len(config.players) != 4:
        raise _bad_request("Deploy mode requires exactly 4 players (1 human + 3 MCTS).")

    human_players = []
    mcts_players = []

    for player_cfg in config.players:
        if player_cfg.agent_type == AgentType.HUMAN:
            human_players.append(player_cfg)
        elif player_cfg.agent_type == AgentType.MCTS:
            mcts_players.append(player_cfg)
        else:
            raise _bad_request(
                "Deploy mode only supports agent_type values: 'human' and 'mcts'."
            )

    if len(human_players) != 1 or len(mcts_players) != 3:
        raise _bad_request(
            "Deploy mode requires exactly 1 human player and exactly 3 MCTS players."
        )

    seen_difficulties = set()
    for player_cfg in mcts_players:
        agent_config = dict(player_cfg.agent_config or {})
        difficulty_raw = agent_config.get("difficulty")
        budget_raw = agent_config.get("time_budget_ms")

        if budget_raw is not None:
            budget_ms = _normalize_time_budget_ms(budget_raw)
            if difficulty_raw is not None:
                difficulty = _normalize_difficulty(str(difficulty_raw))
            else:
                # Infer nearest deploy difficulty label for consistent telemetry/config.
                difficulty = min(
                    DEPLOY_DIFFICULTY_TO_MS.keys(),
                    key=lambda k: abs(DEPLOY_DIFFICULTY_TO_MS[k] - budget_ms),
                )
        elif difficulty_raw is not None:
            difficulty = _normalize_difficulty(str(difficulty_raw))
            budget_ms = DEPLOY_DIFFICULTY_TO_MS[difficulty]
        else:
            difficulty = "medium"
            budget_ms = DEPLOY_DIFFICULTY_TO_MS[difficulty]

        if budget_ms > DEPLOY_TIME_BUDGET_CAP_MS:
            raise _bad_request(
                f"time_budget_ms={budget_ms} exceeds deploy cap ({DEPLOY_TIME_BUDGET_CAP_MS}ms)."
            )

        seen_difficulties.add(difficulty)
        agent_config["difficulty"] = difficulty
        agent_config["time_budget_ms"] = budget_ms
        player_cfg.agent_config = agent_config

    if seen_difficulties != _DEPLOY_REQUIRED_DIFFICULTIES:
        required = ", ".join(sorted(_DEPLOY_REQUIRED_DIFFICULTIES))
        raise _bad_request(
            f"Deploy mode requires exactly one MCTS player for each difficulty: {required}."
        )

    human_cfg: PlayerConfig = human_players[0]
    human_cfg.agent_config = dict(human_cfg.agent_config or {})

    return config
