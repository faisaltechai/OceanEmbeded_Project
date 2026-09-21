# OceanEmbed — Architecture

```
Satellite/Ocean Data (synthetic proxy in demo mode)
   -> ml/data/data_pipeline.py            [ingest, QC, regrid, normalize, feature-fuse]
   -> ml/models/model.py                  [ocean embedding + physics-informed decoder + MC-Dropout head]
   -> ml/training/train.py                [temporal train/test split, gradient-descent training]
   -> ml/evaluation/{evaluate,services}.py [RMSE/Bias/Corr vs holdout + vs synthetic ARGO/GLORYS,
                                            thermocline, heat content, extreme events, ARGO guidance]
   -> ml/active_learning/                 [extends the uncertainty/ARGO-guidance layer above into
                                            acquisition scoring, spatial-diversity deployment picks,
                                            platform/depth guidance, adaptive re-sampling cadence]
   -> ml/inference/{inference,build_demo_bundle}.py
                                           [full-grid cached inference -> data/demo/*.json]
   -> backend/app (FastAPI)               [serves the precomputed bundle + live /predict via pickled model]
   -> frontend (React artifact / Vite scaffold)
                                           [map, depth slider, digital profile, uncertainty,
                                            events, ARGO guidance, adaptive sampling, explainability,
                                            validation]
```

## Why this split
- **ml/** never imports anything web-related. It is pure NumPy/Pandas so it can run
  anywhere (including this offline sandbox) and later swap to PyTorch/Xarray without
  touching the API layer.
- **backend/app/services** intentionally re-uses `ml/evaluation/services.py` directly
  rather than re-implementing scientific logic in the API layer — one definition of
  RMSE/thermocline/heat-content/event-detection/ARGO-scoring, imported by both the
  offline evaluation scripts and the live API.
- **ml/active_learning/ extends rather than replaces** `ArgoGuidanceService`: the
  original scoring (uncertainty + sparsity + anomaly) still powers
  `/api/argo/guidance` unchanged; the active-learning layer adds the
  thermocline-gradient / event-risk components, spatial diversity, and
  platform/depth/frequency guidance on top, reusing `thermocline_metrics` and
  `ExtremeEventDetector` rather than duplicating them (see docs/active-learning.md).
- **Caching / no retrain-per-request (section 37/38):** `build_demo_bundle.py` is the
  only place that calls `.train()`. The API layer only ever reads the resulting JSON
  or calls `.predict()` / `.predict_with_uncertainty()` on an already-trained,
  pickled model (`/api/predict`).

## Data flow for one map click (jury demo)
1. User clicks a lat/lon on the map (frontend).
2. Frontend calls `GET /api/ocean/profile?region=...&lat=...&lon=...`.
3. Backend snaps to the nearest precomputed grid cell (`DataStore.nearest_grid_index`
   in production mode; in this demo bundle, `get_profile` matches against the small
   set of precomputed `profile_samples`).
4. Response includes: predicted profile, MC-Dropout uncertainty band, confidence
   category per depth, thermocline depth/temperature/gradient, ocean heat content —
   all numbers computed once during `build_demo_bundle.py`, never invented per-request.

## Deployment targets (section 64)
- Frontend: Vite build → Vercel/Netlify (static hosting is sufficient; no server-side
  rendering needed).
- Backend: any container host (Render/Railway/Fly) running the `backend/Dockerfile`.
- ML artifacts: the two `model_<region>.pkl` files + `backend_bundle.json` ship inside
  the backend image for demo mode; production mode would move these to object storage
  (S3/GCS) and load lazily.
- Database: optional Postgres for app-state (users, saved views, audit logs) — the
  scientific core (map/profile/events/guidance/validation) never depends on it.

## Known architectural simplifications (demo mode, all documented)
- Grid resolution: 1.0° instead of the target 0.25° (config-driven single-line change).
- 6 demo dates instead of a continuous daily archive.
- `/api/ocean/map` in demo mode only serves the depth levels the bundle precomputed
  (every 3rd of the 16 standard levels) to keep the JSON small; production mode would
  compute any (date, depth) pair on demand from the pretrained model and cache it.
- `/api/active-learning/*` is precomputed for the single most recent demo date only
  (same reasoning as `/api/ocean/map` above); `number_of_points`/`platform`/
  `minimum_distance` are still honored live by filtering the precomputed candidate
  pool — see docs/active-learning.md.
