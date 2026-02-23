# Deploy Slice Manifest (APP_PROFILE=deploy)

This document defines the minimum runtime surface for deploy-mode gameplay:
`1 human vs 3 MCTS` with serverless-safe budgets.

## Required Runtime Modules

### Engine
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/engine/board.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/engine/game.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/engine/move_generator.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/engine/pieces.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/engine/bitboard.py`

### Gameplay Agents (Deploy)
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/agents/fast_mcts_agent.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/agents/gameplay_protocol.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/agents/gameplay_fast_mcts.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/agents/gameplay_human.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/webapi/gameplay_agent_factory.py`

### Schemas (Gameplay State / Moves)
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/schemas/game_state.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/schemas/move.py`

### API Runtime (Deploy Routes)
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/api-runtime/app.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/webapi/app.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/webapi/routes_gameplay.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/webapi/deploy_validation.py`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/webapi/profile.py`

Deploy-mode endpoints:
- `GET /health`
- `POST /api/games`
- `GET /api/games/{game_id}`
- `POST /api/games/{game_id}/move`
- `GET /api/agents`
- `GET /api/games`
- `WS /ws/games/{game_id}`

### Frontend Runtime (Play Flow)
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/App.tsx`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/pages/Play.tsx`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/components/GameConfigModal.tsx`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/components/Board.tsx`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/components/PieceTray.tsx`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/store/gameStore.ts`
- `/Users/tjgalloway/Programming Projects 2025/blokus_rl/frontend/src/constants/gameConstants.ts`

## Excluded in Deploy Mode

The following remain in the repo and research runtime, but are not registered in deploy mode:

- Analysis and history endpoints:
  - `/api/analysis/{game_id}`
  - `/api/history`
  - `/api/trends`
- Training endpoints:
  - `/api/training-runs`
  - `/api/training-runs/{run_id}`
  - `/api/training-runs/agents/list`
  - `/api/training-runs/{run_id}/evaluations`
- DB debug/health endpoints:
  - `/api/health/db`
  - `/debug/mongo`

## Explicit Non-Goals for Deploy Slice

- No changes to training contracts in:
  - `/Users/tjgalloway/Programming Projects 2025/blokus_rl/agents/registry.py`
  - `/Users/tjgalloway/Programming Projects 2025/blokus_rl/rl/*`
  - `/Users/tjgalloway/Programming Projects 2025/blokus_rl/league/*`
- No relocation/renaming of engine modules.
