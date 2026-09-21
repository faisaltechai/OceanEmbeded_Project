# OceanEmbed

**A Physics-Informed, Uncertainty-Aware, Explainable AI Platform for Subsurface Ocean Intelligence**

Smart India Hackathon 2026 · Problem Statement **SIH26066** · Theme: Disaster Management
Team: **BWU TECH TITANS**

> Reconstructing the hidden ocean beneath satellite-observed surfaces.

## What this is
OceanEmbed reconstructs daily 3D subsurface ocean temperature (0–1000 m, 16 standard
levels) for the Bay of Bengal and Arabian Sea from surface satellite observations
(SST, SSS, SSH/SLA, currents, winds) using a physics-informed deep learning model,
with uncertainty estimation, Smart ARGO observation guidance, extreme-event detection,
an interactive Digital Ocean Profile, explainable AI, and a validation dashboard
comparing against ARGO and GLORYS.

## ⚠️ Read this before anything else
This prototype was built in a sandboxed environment **with no network access**, so
PyTorch, FastAPI, and real Copernicus/GLORYS/Argo/satellite data could not be
downloaded or installed while authoring it. Rather than fake those pieces:

- The ML model (`ml/models/model.py`) is a **real, from-scratch NumPy** implementation
  of the specified architecture, trained with hand-derived backprop and an actually-
  applied physics-informed loss — it really learns, on real (if synthetic) data.
- The dataset is a **clearly labeled synthetic proxy** ("DEMO / PROTOTYPE DATASET"),
  built to have realistic oceanographic structure (tanh mixed-layer/thermocline
  shape, sparse Argo-like sampling, smoothed GLORYS-like reference) — see
  `ml/data/data_pipeline.py`'s module docstring for full disclosure.
- The FastAPI backend (`backend/`) is complete, typed, documented source code that
  **was not executed live** in the authoring environment (no network to `pip install
  fastapi`) — it is ready to run as-is anywhere with normal internet access.
- Every number surfaced anywhere in the app (RMSE, correlation, event thresholds,
  ARGO guidance scores, explainability contributions) was actually computed by the
  pipeline in this repo. Nothing is hand-typed.

Full disclosure and the reasoning behind every one of these decisions:
`docs/requirements.md`.

## Quickstart
```bash
# 1) Generate the demo dataset + train the model + compute all metrics (~10s)
pip install -r ml/requirements.txt
python ml/inference/build_demo_bundle.py

# 2) Run the real test suite (no pytest needed)
python tests/test_ml_pipeline.py

# 3) Backend API (needs network access to install FastAPI)
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload
# -> http://localhost:8000/api/health

# 4) Frontend
#    See docs/demo.md for the zero-build-step interactive demo (a self-contained
#    artifact), or scaffold the full Vite app described in docs/architecture.md.
```

## Project layout
```
oceanembed/
├── ml/                 # data pipeline, physics-informed model, training/eval/inference
├── backend/            # FastAPI app: demo bundle + live satellite gateway (see docs/live-integrations.md)
├── frontend_src/        # canonical tab-based UI (oceanembed_demo.html) + the merged globe build
│   ├── oceanembed_demo.html    # RULE #1 baseline: tabs, locked palette, live-wired
│   ├── oceanembed_merged.html  # same, + re-themed Three.js globe landing page
│   └── assets/oceanembed-client.js  # shared fetch client used by all three UIs
├── frontend_variants/   # OceanEmbed_Final-1.html: scrolling/globe UI, also live-wired
├── data/demo/           # generated: backend_bundle.json, frontend_demo.json, model_*.pkl
├── docs/                # architecture + methodology docs, incl. live-integrations.md
├── tests/               # ML core + region-logic tests run offline; API tests need FastAPI
├── docker-compose.yml
└── .env.example
```

## Three frontend variants
You asked for all three to be preserved and wired up rather than picking one:
1. **`frontend_src/oceanembed_demo.html`** — tab-based nav, exact locked color
   palette, closest match to the brief's RULE #1. Fully live-wired: backend
   health badge, live SST/Argo overlays on Overview, a working PDF report
   button, a live NOAA cyclone tab, and an Adaptive Sampling tab (Active
   Learning deployment recommendations — see `docs/active-learning.md`).
2. **`frontend_variants/OceanEmbed_Final-1.html`** — the scrolling, Three.js
   globe build. Live-wired at the Overview section (live strip + report
   button); Twin/Explain/Depth-Skill/Events/Argo sections still run on the
   embedded demo dataset.
3. **`frontend_src/oceanembed_merged.html`** — variant 1's tabs/palette/live
   wiring, with variant 2's rotating globe ported into the landing page and
   re-themed to the locked palette (deep-ocean core, sky-blue glow, sand
   markers). Click an active marker to jump straight into that basin's
   live dashboard.

See `docs/live-integrations.md` for exactly what was and wasn't possible to
test end-to-end in the environment this was built in, and a first-run
checklist for when you deploy it somewhere with real internet access.

## Feature status
See `docs/requirements.md` for the full feature matrix (✅ implemented vs. ⚠ partial,
nothing marked done that isn't).

## Research references preserved from the SIH submission
- DORS (2022) — DOI: 10.3390/rs14133198
- Convformer (2024) — DOI: 10.3390/rs16132422
- Physics-guided South China Sea reconstruction (2025) — DOI: 10.3390/rs17172954
- TS-Cast (2026) — DOI: 10.5194/os-22-2161-2026
- Deep-learning hydrographic profile retrieval (2020) — DOI: 10.3390/rs12193151
- CNN-based vertical temperature profile reconstruction (2025)
- Dynamical-statistical interior temperature/salinity retrieval (2021)
- Reference datasets: GLORYS Global Ocean Reanalysis, ARGO Ocean Data Products

## License
MIT — see `LICENSE`.
