# OceanEmbed — Requirements & Feature Matrix

Source of truth: `SIH_2026.pptx` (SIH26066, team BWU TECH TITANS) + the accompanying build brief.

## 1. Problem
Reconstruct daily 3D subsurface ocean temperature (0–1000 m, 15 standard levels) for the
Bay of Bengal and Arabian Sea from surface satellite observations (SST, SSS, SSH/SLA,
currents, winds), validated against ARGO and GLORYS, using a physics-informed,
uncertainty-aware deep learning approach.

## 2. This build's environment constraints (read before anything else)
This prototype was built inside a sandboxed execution environment with:
- **No network access** — no `pip install torch/fastapi/xarray`, no downloading real
  Copernicus GLORYS / PODAAC satellite / Argo GDAC data.
- Only `numpy`, `pandas`, `scipy`, `scikit-learn` preinstalled.

Engineering decisions made as a result (all documented in `docs/data-pipeline.md` and
`docs/model.md`):
- The ML model is a **from-scratch NumPy implementation** of the specified
  encoder → embedding → decoder → uncertainty-head architecture, trained with
  hand-derived backprop (not a stand-in — it actually learns, actually applies
  physics-informed loss terms during training, and actually produces the numbers
  used everywhere in the app). A `torch.nn.Module` port is a drop-in replacement
  once a networked environment is available (see `ml/requirements.txt`).
- The dataset is **synthetic-but-physically-styled** (a parameterised tanh
  mixed-layer/thermocline profile + sparse "Argo-like" sampling + a smoothed
  "GLORYS-like" reference), clearly labeled everywhere as
  `DEMO / PROTOTYPE DATASET`, never presented as live satellite data.
- The FastAPI backend is complete, typed source code but **was not executed live
  in this chat** (no network to `pip install fastapi` here); it is written to run
  unmodified with `pip install -r backend/requirements.txt && uvicorn app.main:app`.
- The live, in-chat interactive demo is delivered as a self-contained React
  artifact that embeds the real JSON output of the pipeline (nothing in it is
  hand-typed/fabricated) so the SIH jury flow (map → depth → profile → ARGO/GLORYS
  compare → uncertainty → events → guidance → explainability → validation) can be
  clicked through today, independent of any server.

## 3. Feature matrix

| # | Feature | Status | Where |
|---|---|---|---|
| 1 | Data integration / harmonization pipeline | ✅ implemented (synthetic sources) | `ml/data/data_pipeline.py` |
| 2 | Daily / 0.25°-style standardization | ✅ (demo uses 1.0° for speed; config-driven) | `ml/configs/config.yaml` |
| 3 | Surface inputs (SST/SSS/SSH/currents/winds) | ✅ | `data_pipeline.py` |
| 4 | Ocean embedding (deep learning) | ✅ | `ml/models/model.py` |
| 5 | 15-level 0–1000m temperature reconstruction | ✅ (16 levels incl. surface) | `model.py` |
| 6 | Physics-informed loss (used in training, not just claimed) | ✅ | `model.py: _physics_grad`, `docs/physics-informed-learning.md` |
| 7 | Uncertainty (MC-Dropout) | ✅ | `model.py: predict_with_uncertainty` |
| 8 | Smart ARGO guidance | ✅ scored from real computed uncertainty/density/anomaly | `ml/evaluation/services.py: ArgoGuidanceService` |
| 9 | Extreme event detection | ✅ threshold-based, documented as prototype methodology | `services.py: ExtremeEventDetector` |
| 10 | Digital Ocean Profile (OceanEmbed vs ARGO vs GLORYS) | ✅ backend endpoint + frontend chart | `api/ocean.py`, frontend artifact |
| 11 | Thermocline analysis | ✅ steepest-gradient definition | `services.py: thermocline_metrics` |
| 12 | Ocean Heat Content | ✅ ρ·cp·∫(T−Tref)dz, documented, not claimed operational | `services.py: ocean_heat_content` |
| 13 | Explainable AI | ✅ feature ablation (real method, not arbitrary weights) | `model.py: feature_ablation_importance` |
| 14 | Event-aware learning (normal vs extreme metrics) | ⚠ partial — validation splits reported; a dedicated extreme-period retrain is future scope | `docs/` future scope |
| 15 | Data pipeline (ingest→QC→regrid→normalize→fuse) | ✅ | `data_pipeline.py` |
| 16 | Regional scope (BoB + Arabian Sea, modular) | ✅ | `config.yaml: regions` |
| 17 | ML stack decision (NumPy now, PyTorch later) | ✅ documented | `ml/requirements.txt` |
| 18 | Validation dashboard (RMSE/Bias/Corr, model vs baseline vs ARGO) | ✅ real computed numbers | `ml/evaluation/evaluate.py` |
| 19 | Map interface (zoom/pan/depth/date/layers) | ✅ (grid heatmap in the artifact; full pan/zoom is a frontend/ Vite follow-up) | frontend artifact |
| 20 | Depth slider / date control | ✅ | frontend artifact |
| 21 | Dashboard navigation (Overview/Map/Profile/Events/Uncertainty/ARGO/Validation/XAI/Data/About) | ✅ | frontend artifact |
| 22 | Landing page | ✅ | frontend artifact |
| 23 | Backend API (all endpoints in section 33 of the brief) | ✅ source complete; not live-executed here (no network) | `backend/app/api/*` |
| 24 | Database (Postgres for app metadata) | ⚠ schema sketch only — not required for the scientific core | `docs/architecture.md` |
| 25 | Docker / docker-compose | ✅ | `backend/Dockerfile`, `docker-compose.yml` |
| 26 | Tests | ✅ minimal but real (shape/metric sanity, not placeholders) | `tests/` |
| 27 | Docs | ✅ this folder | `docs/` |
| 28 | **Active Learning & Adaptive Sampling** (uncertainty → acquisition score → spatially-diverse, platform/depth-aware deployment recommendations → adaptive re-sampling cadence) | ✅ extends `ArgoGuidanceService`; weights/thresholds are documented, configurable prototype parameters, not a validated deployment doctrine | `ml/active_learning/`, `backend/app/api/active_learning.py`, `docs/active-learning.md` |
| 29 | Multi-Modal Ocean Data Fusion (satellite + ARGO + BGC-Argo fusion layer) | ❌ not started — next upgrade phase | — |
| 30 | Interactive Counterfactual Analysis ("Counterfactual Lab") | ❌ not started — next upgrade phase | — |
| 31 | Advanced 3D/4D WebGL ocean visualization (depth/time-aware volume view) | ❌ not started — next upgrade phase | — |
| 32 | *(audit finding, not part of this upgrade)* `frontend_src/assets/oceanembed-client.js` calls `/ocean/map`, `/ocean/profile`, and `/argo?lat=&lon=`, none of which exist in `backend/app/api/ocean.py` / `argo.py` (those only expose `/ocean/surface`, `/ocean/point`, `/argo/guidance`) — a pre-existing gap between the live-gateway endpoints added later and the DataStore-backed map/profile endpoints `docs/architecture.md` describes | ⚠ pre-existing, left as found; the in-chat demo works around it by rendering directly from the embedded bundle rather than calling these routes | `backend/app/api/ocean.py`, `backend/app/api/argo.py` |

Legend: ✅ implemented and runnable · ⚠ partially implemented / documented, not fully built · ❌ not yet started — never silently claimed done.
