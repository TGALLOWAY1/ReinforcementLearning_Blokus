# Cleanup Verification Steps

Run these commands after the cleanup pass to ensure nothing is broken.

## Prerequisites

- Python 3.10+ with project dependencies installed (`pip install -r requirements.txt` or equivalent)
- Node.js 18+ for frontend

## 1. Frontend Build / Typecheck

```bash
cd frontend
npm ci
npm run build
```

**Success:** Build completes with exit code 0, no TypeScript errors.

Optional typecheck:
```bash
npx tsc --noEmit
```

## 2. WebAPI Server Start

```bash
# From repo root
PYTHONPATH=. python run_server.py
```

**Success:** Server starts, logs "Starting Blokus Web API server...", responds to `GET /health`.

Stop with Ctrl+C.

## 3. Unit Tests

```bash
# From repo root
PYTHONPATH=. pytest tests/ -v -x --ignore=tests/test_api_runtime_smoke.py 2>/dev/null || true
# Or a minimal subset:
PYTHONPATH=. pytest tests/test_engine.py tests/test_mobility_metrics.py -v
```

**Success:** Tests pass (or known failures are documented).

## 4. Smoke: Start Game and Get Legal Moves

Start the server first (`PYTHONPATH=. python run_server.py` or `python3 run_server.py`), then:

```bash
# Create a game and fetch legal moves
curl -s -X POST http://localhost:8000/api/games \
  -H "Content-Type: application/json" \
  -d '{"players":[{"player":"RED","agent_type":"human"},{"player":"BLUE","agent_type":"mcts"},{"player":"YELLOW","agent_type":"mcts"},{"player":"GREEN","agent_type":"mcts"}]}' | jq -r '.game_id'
# Use the returned game_id in:
curl -s http://localhost:8000/api/games/<GAME_ID> | jq '.legal_moves | length'
```

**Success:** Game created, legal_moves count > 0.

**Note:** Requires WebAPI running on port 8000. Use `PYTHONPATH=. python run_server.py` from repo root.

## 5. Training Dry-Run (Optional)

```bash
# Quick preflight check
PYTHONPATH=. python rl/training_preflight.py --config configs/v1_rl_vs_mcts.yaml
```

**Success:** Preflight completes without import or config errors.

For a short training dry-run (if desired):
```bash
PYTHONPATH=. python rl/train.py --config configs/v1_rl_vs_mcts.yaml --total-timesteps 100
```

**Success:** Training starts, no module/import errors.

## 6. Deploy Profile (Optional)

```bash
# Run API in deploy profile (gameplay routes only)
PYTHONPATH=. python api-runtime/app.py
```

**Success:** Deploy-mode API starts, `/api/agents` returns deploy-allowed agents.

---

## Summary Checklist

- [ ] Frontend builds
- [ ] WebAPI starts
- [ ] Core unit tests pass
- [ ] Smoke: create game, get legal moves
- [ ] Training preflight succeeds
- [ ] (Optional) Deploy profile starts
