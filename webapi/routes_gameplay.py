"""
Route registration for deploy-safe gameplay endpoints.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Callable, List

from fastapi import FastAPI

from schemas.game_state import AgentInfo, GameCreateResponse, GameState, MoveResponse

AsyncHandler = Callable[..., Awaitable[Any]]
SyncHandler = Callable[..., Any]


def register_gameplay_routes(
    app: FastAPI,
    *,
    health: SyncHandler,
    root: AsyncHandler,
    create_game: AsyncHandler,
    get_game: AsyncHandler,
    make_move: AsyncHandler,
    finish_game: AsyncHandler,
    delete_game: AsyncHandler,
    get_agents: AsyncHandler,
    list_games: AsyncHandler,
    advance_turn: AsyncHandler,
    pass_turn: AsyncHandler,
    list_arena_runs: AsyncHandler,
    get_arena_run: AsyncHandler,
    get_champion: AsyncHandler,
) -> None:
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", root, methods=["GET"])
    app.add_api_route("/api/games", create_game, methods=["POST"], response_model=GameCreateResponse)
    app.add_api_route("/api/games/{game_id}", get_game, methods=["GET"], response_model=GameState)
    app.add_api_route("/api/games/{game_id}/move", make_move, methods=["POST"], response_model=MoveResponse)
    app.add_api_route("/api/games/{game_id}/pass", pass_turn, methods=["POST"], response_model=MoveResponse)
    app.add_api_route("/api/games/{game_id}/finish", finish_game, methods=["POST"])
    app.add_api_route("/api/games/{game_id}", delete_game, methods=["DELETE"])
    app.add_api_route("/api/agents", get_agents, methods=["GET"], response_model=List[AgentInfo])
    # Champion metadata is deploy-safe (read-only registry metadata) and powers the
    # public "Play the Champion" demo's opponent banner.
    app.add_api_route("/api/champion", get_champion, methods=["GET"])
    app.add_api_route("/api/games", list_games, methods=["GET"], response_model=List[GameState])
    app.add_api_route("/api/games/{game_id}/advance_turn", advance_turn, methods=["POST"], response_model=MoveResponse)
    # Arena leaderboard reads are deploy-safe (read-only, no sensitive data) and power the
    # "Current Best" opponent picker in the game-setup UI, so they register here rather than
    # in the research-only group.
    app.add_api_route("/api/arena-runs", list_arena_runs, methods=["GET"])
    app.add_api_route("/api/arena-runs/{run_id}", get_arena_run, methods=["GET"])
