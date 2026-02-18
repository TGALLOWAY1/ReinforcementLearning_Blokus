# Verification Checklist

## Local smoke tests
- [ ] Start API: `python run_server.py`
- [ ] Start frontend: `cd frontend && npm run dev`
- [ ] Start engine service: `uvicorn engine-service.app:app --host 0.0.0.0 --port 8100`
- [ ] Set `ENGINE_URL=http://localhost:8100` for API process.
- [ ] Create game preset **Human vs MCTS (1s/3s/5s)** and complete game.
- [ ] Confirm AI move logs include `timeBudgetMs`, `timeSpentMs`, `nodesEvaluated`, `maxDepthReached`.
- [ ] Open `/analysis/:gameId` and verify per-move + aggregates render.
- [ ] Confirm trends block loads from `/api/trends`.

## Deployment smoke tests (Vercel + external compute)
- [ ] Deploy frontend to Vercel.
- [ ] Deploy API to Vercel Python runtime (or managed FastAPI endpoint) with Mongo env vars.
- [ ] Deploy engine service to long-running host (Render/Fly/Railway).
- [ ] Configure API env `ENGINE_URL=https://<engine-service>`.
- [ ] Verify `GET /health` on engine service returns `{ ok: true }`.
- [ ] Play one full game in production and open analysis page.

## Edge-case checks
- [ ] Game end state persists and is queryable by `/api/analysis/{gameId}`.
- [ ] Restart/new game does not leak move records across game IDs.
- [ ] Browser refresh during game reconnects and continues state updates.
- [ ] Temporary engine-service failure falls back to local MCTS and game continues.
- [ ] Network blip during AI move does not crash turn loop.

## Load sanity script
- [ ] Run 50 think requests per budget and capture p50/p95 + error rate.

Example:
```bash
python scripts/load_test_engine_service.py --engine-url http://localhost:8100 --requests 50 --budgets 1000 3000 5000
```
