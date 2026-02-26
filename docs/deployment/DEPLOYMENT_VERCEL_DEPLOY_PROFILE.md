# Vercel Deployment (Deploy Profile)

This guide deploys the minimal gameplay runtime (`1 human vs 3 MCTS`) while keeping
research/training code in-repo and unchanged.

## Profiles

- `APP_PROFILE=research` (default): full API surface (analysis/history/training/db routes).
- `APP_PROFILE=deploy`: gameplay-only routes and deploy constraints.

## Required Environment Variables

Backend/API:
- `APP_PROFILE=deploy`
- `ENGINE_URL` (optional; if set, API calls external `engine-service` for think)

Frontend:
- `VITE_APP_PROFILE=deploy`
- `VITE_API_URL` (public API base URL)
- `VITE_WS_URL` (optional; if omitted, derived from `VITE_API_URL`)

Optional (research profile only):
- `MONGODB_URI`
- `MONGODB_DB_NAME`

## Compute Modes

### Mode A: Local MCTS in API (Fully Serverless)
- Leave `ENGINE_URL` unset.
- API computes AI moves using `FastMCTSAgent` through deploy gameplay adapters.
- Time budgets are capped by backend deploy validation.

### Mode B: External Engine Service
- Set `ENGINE_URL=https://<engine-service-host>`.
- API sends `/think` requests to engine-service first.
- On engine failure/timeouts, API falls back to local MCTS.

## Local Smoke Test

1. Start deploy API runtime:
```bash
APP_PROFILE=deploy PYTHONPATH=. python3 api-runtime/app.py
```

2. Create a valid deploy game:
```bash
curl -sS -X POST http://localhost:8000/api/games \
  -H 'Content-Type: application/json' \
  -d '{
    "players": [
      {"player":"RED","agent_type":"human","agent_config":{}},
      {"player":"BLUE","agent_type":"mcts","agent_config":{"difficulty":"easy"}},
      {"player":"GREEN","agent_type":"mcts","agent_config":{"difficulty":"medium"}},
      {"player":"YELLOW","agent_type":"mcts","agent_config":{"difficulty":"hard"}}
    ],
    "auto_start": true
  }'
```

3. Verify health:
```bash
curl -sS http://localhost:8000/health
```

4. Verify invalid config is rejected:
```bash
curl -sS -X POST http://localhost:8000/api/games \
  -H 'Content-Type: application/json' \
  -d '{"players":[{"player":"RED","agent_type":"human"},{"player":"BLUE","agent_type":"random"}]}' 
```

Expected: HTTP `400` with clear validation message.

## Vercel Notes

- `vercel.json` routes `/api/*`, `/ws/*`, and `/health` to `api-runtime/app.py`.
- Frontend is built from `/frontend` and served as SPA.
- Keep `APP_PROFILE=deploy` in Vercel project env for API runtime.
