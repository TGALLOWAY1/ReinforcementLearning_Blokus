# Stage 3: Self-Play League, Engine Service, Frontend Analysis/History

## Overview

This PR implements Stage 3 of the Blokus RL training pipeline, introducing a self-play league system with checkpoint-based opponents, an external engine service for MCTS compute, frontend analysis/history pages, and comprehensive deployment documentation.

## Key Features

### 🎮 League System (4-Player Individual Competition)

- **4-player matches**: League now supports 4 individual players (not teams), with winner determined by highest individual score
- **Pairwise Elo updates**: All pairs of players get Elo updates based on relative scores (wins/draws/losses), including proper handling of ties
- **Checkpoint-based opponents**: League can use trained RL checkpoints as opponents alongside baseline agents (MCTS, random, heuristic)
- **League manager**: `LeagueManager` class for orchestrating matches, opponent selection, and Elo tracking

### 🤖 Self-Play Training Pipeline

- **4-player training**: RL agent trains against 3 opponents (configurable: MCTS, fast_mcts, random, heuristic)
- **League evaluation**: Periodic evaluation runs 4-player league matches to track RL agent performance
- **Checkpoint integration**: Training saves checkpoints that can be used as league opponents
- **Concise logging**: Clean progress logs showing steps, estimated games, league W/L/D, and Elo ratings

### 🚀 Engine Service (External MCTS Compute)

- **Standalone service**: `engine-service/app.py` provides HTTP API for MCTS computation
- **Think endpoint**: `POST /think` accepts game state and returns move + telemetry
- **Health check**: `GET /health` for service monitoring
- **Fallback support**: Web API can use external engine service or fall back to local computation

### 📊 Frontend Enhancements

- **Analysis page**: Game-by-game analysis with move-by-move telemetry, trends, and statistics
- **History page**: List of completed games with filtering and navigation
- **Game config modal**: Enhanced configuration for MCTS difficulty presets and agent selection
- **Routing**: New routes for `/analysis/:gameId` and `/history`

### 📈 Analytics & Metrics

- **Comprehensive metrics**: Blocking, center control, corners, mobility, option quality, pieces, proximity, territory
- **Phase analysis**: Game phase detection and phase-based metrics
- **Aggregation**: Tools for aggregating metrics across games and agents
- **Tournament support**: Elo tracking and matchup analysis

### 🧪 Testing & Quality

- **League tests**: Tests for 4-player matches, Elo updates (including ties), and agent building
- **Payload tests**: Tests for analysis and history API payloads
- **Fast MCTS tests**: Tests for MCTS thinking and telemetry
- **Benchmarks**: Performance benchmarks for move generation and environment

### 📚 Documentation

- **Deployment notes**: Vercel + external engine service deployment guide
- **Deployment plan**: Architecture overview and implementation roadmap
- **Verification checklist**: Pre-deployment verification steps
- **Training entrypoints**: Documentation for training pipeline usage

## Technical Changes

### Backend

- `league/league.py`: 4-player match support, pairwise Elo updates, `build_league_agents` with ordered names
- `rl/train.py`: 4-player training loop, league evaluation, concise logging, games estimation
- `webapi/app.py`: Engine service integration, analysis/history endpoints, game persistence
- `agents/registry.py`: Enhanced agent building with unique names for duplicate types
- `schemas/game_state.py`: Extended schemas for analysis payloads and player configs

### Frontend

- `frontend/src/pages/Analysis.tsx`: New analysis page component
- `frontend/src/pages/History.tsx`: New history page component
- `frontend/src/pages/Play.tsx`: Updates for navigation to analysis/history
- `frontend/src/components/GameConfigModal.tsx`: Enhanced configuration UI
- `frontend/src/App.tsx`: New routes for analysis and history

### Infrastructure

- `engine-service/app.py`: New FastAPI service for external MCTS compute
- `scripts/load_test_engine_service.py`: Load testing for engine service
- `scripts/run_sample_games.py`: Sample game runner for testing
- `rl/training_preflight.py`: Pre-flight checks for training setup

### Configuration

- `configs/v1_rl_vs_mcts.yaml`: Stage 2 training config (RL vs fast_mcts + 2 random)
- `configs/portfolio_fast_mcts.yaml`: Portfolio optimization config
- `configs/smoke_mcts.yaml`: Smoke test config with MCTS

## Testing

- ✅ League 4-player matches and Elo updates
- ✅ Analysis and history API payloads
- ✅ Fast MCTS thinking and telemetry
- ✅ Training pipeline with league evaluation
- ✅ Engine service health and think endpoints

## Deployment

See `docs/DEPLOYMENT_NOTES.md` and `docs/DEPLOYMENT_PLAN.md` for:
- Vercel frontend deployment
- External engine service setup
- Environment variable configuration
- Local development with two-process architecture

## Breaking Changes

- League `play_match` now requires 4 agents (not 2) - signature changed from `(agent1_name, agent2_name, agent1, agent2)` to `(agent_names: List[str], agents: List[AgentProtocol])`
- `build_league_agents` now returns `(agents, specs, ordered_4p_names)` tuple (added third return value)

## Migration Guide

If you have code using the old league API:
- Update `play_match` calls to pass 4 agents: `league.play_match([name0, name1, name2, name3], [agent0, agent1, agent2, agent3], ...)`
- Update `build_league_agents` unpacking: `agents, specs, ordered_names = build_league_agents(...)`

## Related Issues

- Implements 4-player league system
- Adds checkpoint-based opponent support
- Enables external MCTS compute service
- Provides game analysis and history UI

## Checklist

- [x] Code follows project style guidelines
- [x] Tests added/updated and passing
- [x] Documentation updated
- [x] No breaking changes (or migration guide provided)
- [x] Deployment documentation included
