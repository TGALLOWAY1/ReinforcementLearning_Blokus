# Deployment Notes (Vercel + External Engine Service)

## Architecture
- `frontend/` (Vite React) deployed to Vercel as static site.
- `webapi/app.py` deployed as Python API runtime.
- `engine-service/app.py` deployed to a long-running host.

## Required environment variables
- `MONGODB_URI`
- `MONGODB_DB_NAME`
- `ENGINE_URL` (e.g. `https://blokus-engine.example.com`)
- Frontend: `VITE_API_URL` and optional `VITE_WS_URL`

## Local two-process run
1. Start compute service:
   ```bash
   uvicorn engine-service.app:app --host 0.0.0.0 --port 8100
   ```
2. Start API with external engine enabled:
   ```bash
   ENGINE_URL=http://localhost:8100 python run_server.py
   ```
3. Start frontend:
   ```bash
   cd frontend && npm run dev
   ```

## Vercel
- Set frontend project root to `frontend/`.
- Configure API deployment to expose FastAPI endpoints.
- Add `ENGINE_URL` in API environment settings.
