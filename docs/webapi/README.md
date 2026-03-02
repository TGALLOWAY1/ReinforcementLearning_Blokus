# Blokus RL Web API

FastAPI backend for game orchestration, analysis routes, and MongoDB-backed research data.

## Profiles

The app supports two profiles via `APP_PROFILE`:

- `research` (default): gameplay routes + research/data routes + MongoDB health endpoints
- `deploy`: gameplay-safe routes only (no research data endpoints)

## Quick Start

From the project root:

```bash
python3 run_server.py
```

API docs: `http://localhost:8000/docs`

Alternative startup:

```bash
APP_PROFILE=research python3 -m uvicorn webapi.app:app --reload --host 0.0.0.0 --port 8000
```

## MongoDB Configuration

Environment variables used by `webapi/db/mongo.py`:

- `MONGODB_URI` (default: `mongodb://localhost:27017`)
- `MONGODB_DB_NAME` (default: `blokusdb`)

Example:

```bash
export MONGODB_URI="mongodb://localhost:27017"
export MONGODB_DB_NAME="blokusdb"
```

## Gameplay Endpoints

- `GET /health`
- `GET /`
- `POST /api/games`
- `GET /api/games/{game_id}`
- `GET /api/games`
- `GET /api/agents`
- `POST /api/games/{game_id}/move`
- `POST /api/games/{game_id}/pass`
- `POST /api/games/{game_id}/advance_turn`
- `POST /api/games/{game_id}/finish`

### Example: Create Game

```http
POST /api/games
Content-Type: application/json

{
  "players": [
    {"player": "RED", "agent_type": "human"},
    {"player": "BLUE", "agent_type": "random"},
    {"player": "YELLOW", "agent_type": "heuristic"},
    {"player": "GREEN", "agent_type": "mcts", "agent_config": {"time_budget_ms": 500}}
  ],
  "auto_start": true
}
```

### Example: Make Move

```http
POST /api/games/{game_id}/move
Content-Type: application/json

{
  "player": "RED",
  "move": {
    "piece_id": 1,
    "orientation": 0,
    "anchor_row": 0,
    "anchor_col": 0
  }
}
```

### Example: Advance One AI Turn

```http
POST /api/games/{game_id}/advance_turn
```

## Research Endpoints (`APP_PROFILE=research`)

- `GET /api/health/db`
- `GET /debug/mongo`
- `GET /api/analysis/{game_id}`
- `GET /api/analysis/{game_id}/replay`
- `GET /api/analysis/{game_id}/steps`
- `GET /api/analysis/{game_id}/summary`
- `GET /api/history`
- `GET /api/trends`
- `GET /api/training-runs`
- `GET /api/training-runs/{run_id}`
- `GET /api/training-runs/agents/list`
- `GET /api/training-runs/{run_id}/evaluations`

## Important Notes

- WebSocket routes are not currently registered in `webapi/app.py`; gameplay interaction is REST-driven.
- Human players are represented by `agent_type: "human"` and progress turns via move/pass endpoints.

## Relevant Files

- `webapi/app.py`: app factory, game manager wiring, route handlers
- `webapi/routes_gameplay.py`: gameplay route registration
- `webapi/routes_research.py`: research route registration
- `webapi/db/mongo.py`: MongoDB connection lifecycle
- `schemas/game_state.py`: request/response models

