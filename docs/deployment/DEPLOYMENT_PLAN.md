# Deployment Plan (Option A: Vercel + External MCTS Compute)

## Phase 0 Repo Map

## Core engine/rules
- `engine/game.py`: Top-level Blokus game orchestration, turn flow, scoring, legal move access.
- `engine/board.py`: Canonical board state and placement legality rules (first-move corner, corner-only adjacency, no edge contact).
- `engine/move_generator.py`: Legal move generation with frontier and bitboard optimizations.

## Agents / MCTS
- `agents/random_agent.py`: Baseline random legal move policy.
- `agents/heuristic_agent.py`: Heuristic move chooser.
- `agents/fast_mcts_agent.py`: Current real-time MCTS-like agent with `iterations` + `time_limit` but no structured per-move telemetry return.
- `mcts/mcts_agent.py`: Full MCTS implementation with internal stats.

## API/backend
- `webapi/app.py`: Main FastAPI app, game creation, move handling, websocket updates, AI turn loop.
- `schemas/game_state.py`: API/game schemas currently used by web backend and frontend.
- `webapi/db/mongo.py`: Mongo connection lifecycle.
- `webapi/db/models.py`: Existing Mongo models focused on training/eval history (not gameplay analytics yet).

## Frontend/game flow
- `frontend/src/pages/Play.tsx`: Main gameplay page.
- `frontend/src/components/GameConfigModal.tsx`: Game/bot setup UI.
- `frontend/src/store/gameStore.ts`: Client API + websocket state management.
- `frontend/src/App.tsx`: Routing.

## Current gaps vs. requested goal
- No explicit 1s / 3s / 5s MCTS difficulty presets with fixed budget semantics.
- No unified `think(...)->{move,stats}` interface surfaced to game loop and UI/API.
- No persistence model for completed games and per-move think telemetry.
- No analysis/trends API endpoints for gameplay analytics.
- No gameplay analysis page in frontend.
- No separate external compute service for MCTS with `POST /think`, `GET /health`.
- No Vercel-oriented deployment docs/env wiring for split architecture.

## Proposed minimal-change implementation
- Reuse `FastMCTSAgent`; add a telemetry-capable `think` method while preserving existing `select_action` signature for compatibility.
- Extend `schemas/game_state.py` minimally for player config passthrough + analysis payloads.
- In `webapi/app.py`:
  - wire per-player MCTS budgets via `agent_config.time_budget_ms`.
  - collect per-move stats for AI and user timing.
  - persist game + move records to Mongo collections at game end.
  - add `/api/analysis/{game_id}` and `/api/trends`.
- Add frontend `Analysis` page and route; keep existing Play flow, add game-over navigation/button.
- Add `engine-service/` FastAPI microservice exposing `POST /think` + `GET /health` and sharing project modules.
- Add `ENGINE_URL` support in web API; for MCTS turns call engine-service over HTTP with timeout/retry, fallback to local think.
- Add local two-process dev script/documentation and Vercel deployment notes.
- Add regression tests for:
  - legal move validity for MCTS output,
  - time budget behavior tolerance,
  - analysis/trends payload shape.

## Risks and mitigations
- **Risk: MCTS runtime jitter** in CI/dev affecting strict budget tests.
  - **Mitigation:** use tolerance windows and assert upper-bound soft margin.
- **Risk: Mongo unavailable in local test env.**
  - **Mitigation:** keep persistence tests isolated/mocked or skip when DB not configured.
- **Risk: External engine timeout/network failure stalls turn loop.**
  - **Mitigation:** add short request timeout + bounded retry + local fallback + explicit skip/log behavior.
- **Risk: schema/UI mismatch for new analysis payloads.**
  - **Mitigation:** lightweight API shape tests and typed frontend interfaces.
