# CryptoScope — Crypto Market Analysis (Render + Vercel)

A deployable full-stack interface for the existing Python cryptocurrency analysis pipeline. The backend exposes a FastAPI REST API; the frontend is a responsive React + TypeScript + Vite dashboard.

## Project layout

- `backend/main.py` — REST API, validation, CORS, health check, optional protected refresh endpoint.
- `frontend/` — Vite/React dashboard, charting, rankings, search and asset history.
- `src/`, `data/`, `reports/` — existing analysis pipeline and sample generated datasets.
- `render.yaml` — Render service blueprint.

## Run locally

Requires Python 3.11+ and Node.js 20+.

### 1. Backend

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs. Health: http://localhost:8000/health.

### 2. Frontend

```bash
cd frontend
npm install
# Windows PowerShell:
$env:VITE_API_BASE_URL="http://localhost:8000"
# macOS/Linux:
# export VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

Vite prints the local URL. To build: `npm run build`.

## Deploy backend to Render

1. Push this repository to GitHub (do not commit `.env` files or secrets).
2. In Render, choose **New + → Blueprint** and select the repository, or create a **Web Service**.
3. Blueprint settings are in `render.yaml`. If creating manually:
   - Root Directory: leave blank (recommended, so the existing `src/`, `data/`, and `reports/` are included).
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Health Check Path: `/health`
4. Add environment variable `CORS_ORIGINS` with your deployed Vercel origin, e.g. `https://your-project.vercel.app` (no trailing slash). For multiple origins, comma-separate them.
5. Optional: set `REFRESH_TOKEN` to a long random secret to enable `POST /api/refresh`. Without it, refresh is intentionally disabled. The current dataset is bundled with the repository; Render's ephemeral filesystem means generated changes are not durable across restarts/redeploys unless you configure persistent storage.

**Important:** use the repository root as Render's root directory. The analysis modules and datasets live outside `backend/`.

## Deploy frontend to Vercel

1. Import the same GitHub repository into Vercel.
2. Set **Root Directory** to `frontend`.
3. Framework preset: Vite.
4. Build command: `npm run build`; output directory: `dist`.
5. Add environment variable `VITE_API_BASE_URL` = your Render service origin, e.g. `https://crypto-market-analysis-api.onrender.com` (no trailing slash).
6. Deploy. If you change environment variables, redeploy so Vite bakes them into the frontend build.
7. Copy the final Vercel origin into Render's `CORS_ORIGINS` and redeploy the backend if required.

The frontend `vercel.json` enables SPA route fallback. All API requests go directly to the Render backend.

## API endpoints

- `GET /health` — service and dataset availability.
- `GET /api/overview` — headline metrics and coin list.
- `GET /api/coins?search=bitcoin&sort_by=volatility&limit=25&offset=0`
- `GET /api/coins/{coin_id}` — single-coin summary.
- `GET /api/coins/{coin_id}/history?limit=300` — price and moving-average series.
- `GET /api/outliers?limit=100`
- `POST /api/refresh?top_n=20&days=5` (send the secret in the `X-Refresh-Token` header) — optional protected pipeline run; requires `REFRESH_TOKEN`.

## Production notes

- Configure the exact frontend origin in CORS rather than `*` for deployment.
- Do not place CoinGecko API keys or the refresh token in `VITE_*` variables. Anything prefixed `VITE_` is public in the browser bundle.
- Public CoinGecko rate limits apply when the pipeline is refreshed. Start with a small `top_n`.
- This project reports historical statistical measures and does not make guaranteed predictions or provide investment advice.
- SQLite and generated CSVs are local-file persistence. For a durable live product, use a managed database/object storage and a scheduled worker; Render's ephemeral filesystem should not be treated as permanent storage.
