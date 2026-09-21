# Deployment

## Local / demo
```bash
# 1. Generate the demo bundle (regenerate anytime; deterministic given the config seed)
pip install -r ml/requirements.txt
python ml/inference/build_demo_bundle.py       # writes data/demo/*.json + model_<region>.pkl

# 2. Backend
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload    # http://localhost:8000/api/health

# 3. Frontend (Vite scaffold under frontend_src/ — see docs/demo.md for the
#    in-chat React-artifact version that needs no build step)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

## Cloud targets
| Layer | Target | Notes |
|---|---|---|
| Frontend | Vercel / Netlify | static Vite build, `VITE_API_BASE_URL` env var |
| Backend | Render / Railway / Fly.io | `backend/Dockerfile`, set `OCEANEMBED_BUNDLE_PATH` |
| ML artifacts | ship inside backend image (demo) → S3/GCS (production) | `data/demo/*.pkl`, `*.json` |
| Database (optional) | managed Postgres | only if you enable app-state (section 35) |

## Environment variables
See `.env.example` at repo root — `OCEANEMBED_ENV`, `OCEANEMBED_BUNDLE_PATH`,
`OCEANEMBED_CORS_ORIGINS`, `VITE_API_BASE_URL`, plus optional Postgres vars.

## Regenerating with real data later
Replace `ml/data/data_pipeline.py: generate_raw_fields()` with real NetCDF readers
(Copernicus Marine GLORYS, PODAAC/NOAA satellite products, Argo GDAC) — every
downstream function (QC, feature matrix, model, evaluation, API) is written against
the same array shapes, so nothing else needs to change to go from demo to production
data.
