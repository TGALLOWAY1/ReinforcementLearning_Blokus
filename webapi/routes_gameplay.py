"""
Route registration for deploy-safe gameplay endpoints.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, List

from fastapi import FastAPI, WebSocket

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
    get_agents: AsyncHandler,
    list_games: AsyncHandler,
    websocket_endpoint: Callable[[WebSocket, str], Awaitable[None]],
) -> None:
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", root, methods=["GET"])
    app.add_api_route("/api/games", create_game, methods=["POST"], response_model=GameCreateResponse)
    app.add_api_route("/api/games/{game_id}", get_game, methods=["GET"], response_model=GameState)
    app.add_api_route("/api/games/{game_id}/move", make_move, methods=["POST"], response_model=MoveResponse)
    app.add_api_route("/api/games/{game_id}/finish", finish_game, methods=["POST"])
    app.add_api_route("/api/agents", get_agents, methods=["GET"], response_model=List[AgentInfo])
    app.add_api_route("/api/games", list_games, methods=["GET"], response_model=List[GameState])
    app.add_api_websocket_route("/ws/games/{game_id}", websocket_endpoint)
