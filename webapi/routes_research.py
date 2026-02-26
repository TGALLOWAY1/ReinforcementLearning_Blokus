"""
Route registration for research-only API endpoints.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Callable

from fastapi import FastAPI

AsyncHandler = Callable[..., Awaitable[Any]]


def register_research_routes(
    app: FastAPI,
    *,
    health_check_db: AsyncHandler,
    mongo_debug: AsyncHandler,
    get_game_analysis: AsyncHandler,
    get_game_replay: AsyncHandler,
    get_analysis_steps: AsyncHandler,
    get_analysis_summary: AsyncHandler,
    get_history: AsyncHandler,
    get_trends: AsyncHandler,
    list_training_runs: AsyncHandler,
    get_training_run: AsyncHandler,
    list_agents: AsyncHandler,
    get_training_run_evaluations: AsyncHandler,
) -> None:
    app.add_api_route("/api/health/db", health_check_db, methods=["GET"])
    app.add_api_route("/debug/mongo", mongo_debug, methods=["GET"])
    app.add_api_route("/api/analysis/{game_id}", get_game_analysis, methods=["GET"])
    app.add_api_route("/api/analysis/{game_id}/replay", get_game_replay, methods=["GET"])
    app.add_api_route("/api/analysis/{game_id}/steps", get_analysis_steps, methods=["GET"])
    app.add_api_route("/api/analysis/{game_id}/summary", get_analysis_summary, methods=["GET"])
    app.add_api_route("/api/history", get_history, methods=["GET"])
    app.add_api_route("/api/trends", get_trends, methods=["GET"])
    app.add_api_route("/api/training-runs", list_training_runs, methods=["GET"])
    app.add_api_route("/api/training-runs/{run_id}", get_training_run, methods=["GET"])
    app.add_api_route("/api/training-runs/agents/list", list_agents, methods=["GET"])
    app.add_api_route("/api/training-runs/{run_id}/evaluations", get_training_run_evaluations, methods=["GET"])
