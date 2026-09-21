# Active Learning & Adaptive Sampling

**Status: IMPLEMENTED** (backend + ML + API + frontend on the canonical demo +
tests + docs). See the caveats at the end for what is demo-mode-limited.

## What this is

Turns the uncertainty engine OceanEmbed already had (`model.py:
predict_with_uncertainty`, `services.py: ArgoGuidanceService`) into an
autonomous observation-planning layer: for the region/date the pipeline
already reconstructs a full grid for, it answers "if we could deploy N more
real ocean-observing platforms right now, where should they go, what kind of
platform, at what depths, and how often should we re-sample?"

It does **not** introduce a new uncertainty algorithm. Every number is
derived from quantities the app already computes:

- MC-Dropout uncertainty (`model.py`)
- the anomaly/sparsity convention `ArgoGuidanceService` already scores on
- `services.py: thermocline_metrics` (steepest-gradient definition, unchanged)
- `services.py: ExtremeEventDetector`'s z-score / mixed-layer-collapse
  ingredients, extended with one new method, `event_risk_grid()`, that
  reuses the same `_anomaly_fields()` helper `detect()` uses, just returns a
  continuous 0-1 grid instead of a binary top-N-capped event list

## Where the code lives

```
ml/active_learning/
├── __init__.py            build_active_learning_bundle() — the one entry point
├── uncertainty_map.py      per-cell metrics grid (uncertainty, anomaly,
│                           thermocline_gradient, event_risk, observation
│                           density, distance-to-nearest-observation)
├── acquisition.py          AcquisitionFunction — weighted, explainable score
├── sampling_optimizer.py   spatial-diversity selection + platform/depth rules
├── adaptive_frequency.py   uncertainty -> re-sampling cadence
└── models.py               SamplingRecommendation dataclass (the JSON shape)

backend/app/api/active_learning.py   GET /api/active-learning/recommendations
                                      GET /api/active-learning/map
tests/test_active_learning.py        17 tests, all passing (see below)
```

`ml/inference/build_demo_bundle.py` calls `build_active_learning_bundle()`
once per region, at the same point it already computes `ArgoGuidanceService`
scores, and stores the result under `regions.<region>.active_learning` in
both `data/demo/backend_bundle.json` and `data/demo/frontend_demo.json`.
**It is never computed per API request** — the "train != inference, cache
the rest" rule in `docs/architecture.md` applies here too.

## Methodology

### 1. Per-cell metrics (`uncertainty_map.py`)
For the most recent demo date, every grid cell gets:
`uncertainty_score`, `prediction_variance` (`= uncertainty_score**2`),
`temperature_anomaly` (vs. the demo-period climatology, same convention as
`ArgoGuidanceService`), `thermocline_gradient` (looped call to the existing
`thermocline_metrics`, not a new formula), `event_risk` (continuous, from
`ExtremeEventDetector.event_risk_grid`), `observation_density` (passed
through from `ml/inference/inference.py`), and `distance_to_observation_deg`
(nearest synthetic-Argo hit).

### 2. Acquisition score (`acquisition.py`)
```
acquisition = w_uncertainty        * norm(uncertainty_score)
            + w_anomaly            * norm(|temperature_anomaly|)
            + w_thermocline_gradient * norm(|thermocline_gradient|)
            + w_event_risk         * norm(event_risk)
            + w_observation_gap    * norm(1 - norm(observation_density))
```
All five components are independently min-max normalized over the grid
(same convention as `ArgoGuidanceService._minmax`). Default weights
(`ml/configs/config.yaml: active_learning.acquisition_weights`) are
`{uncertainty: 0.35, anomaly: 0.20, thermocline_gradient: 0.20,
event_risk: 0.15, observation_gap: 0.10}`.

**These weights are a documented prototype allocation, not a scientifically
optimized or validated one.** Anyone retuning them should say so the same
way. `AcquisitionFunction` always renormalizes whatever weights it's given
to sum to 1, so a partial override in config never silently changes the
total scale.

`explain()` builds the `reason` string from whichever normalized components
actually crossed a threshold (0.6 by default) at that cell — never a canned
string, same convention as `ArgoGuidanceService.explain`.

### 3. Spatial diversity (`sampling_optimizer.py`)
Never "top N pixels": a greedy selector walks the acquisition-ranked cell
list and skips any candidate within `min_distance_deg` (default 1.5°) of an
already-selected point, because nearby high-score cells are usually the same
real-world feature (e.g. one eddy edge) — picking several adjacent pixels
wastes several platforms on one feature instead of covering the region.
The demo precomputes a pool of `default_top_n = 20` diverse candidates per
region so the API has real headroom to serve `number_of_points` requests
without ever recomputing anything (see API section below).

### 4. Platform recommendation (`sampling_optimizer.py: recommend_platform`)
A **documented heuristic, not a validated deployment doctrine**:

1. `event_risk >= event_risk_threshold` (default 0.7) → **AUV** (fastest
   response platform)
2. else `|thermocline_gradient| >= gradient_threshold_c_per_m` (default
   0.05 °C/m) → **Glider** (repeated transects resolve fine vertical
   structure better than one profile)
3. else uncertainty concentrated below `deep_focus_m` (default 300 m) →
   **BGC-Argo** (long-duration profiling float with biogeochemical sensors)
4. else → **Argo** (standard profiling float)

All three thresholds are configurable
(`ml/configs/config.yaml: active_learning.platform_rules`).

### 5. Depth recommendation (`recommend_depths`)
Surface + the cell's own computed thermocline depth (from
`thermocline_metrics`, never invented) + the two deepest standard levels
from the model's own depth configuration — every value snapped to an actual
model depth level, nothing fabricated. When the computed thermocline depth
ties to the surface level (common in this synthetic dataset's shallow
tropical mixed layer), the list naturally has 3 unique depths instead of 4 —
left as-is rather than forced to look fuller than the real computation gives.

### 6. Adaptive sampling frequency (`adaptive_frequency.py`)
Reuses the exact Low/Medium/High uncertainty bands
`ml/inference/inference.py: confidence_category` already defines, extended
with an event-flagged "priority" tier:

| Band | Uncertainty (°C) | Cadence | Interval |
|---|---|---|---|
| normal | ≤ `low_max_c` (0.35) | routine | 10 days |
| increased | `low_max_c`–`medium_max_c` (0.35–0.8) | closer watch | 5 days |
| intensive | > `medium_max_c` (0.8) | frequent | 2 days |
| priority | `event_risk >= event_risk_threshold` | immediate | 1 day |

## API

```
GET /api/active-learning/recommendations?region=<region>
    [&number_of_points=N][&platform=AUV|Glider|Argo|BGC-Argo]
    [&minimum_distance=deg][&date=YYYY-MM-DD][&depth=m]

GET /api/active-learning/map?region=<region>
```

Both read the precomputed bundle only — **no model inference happens on
either request**:
- `number_of_points` slices the precomputed, already-ranked candidate pool.
- `platform` filters to one recommended platform.
- `minimum_distance` re-runs the greedy spatial filter on the *already
  computed* candidate list's own lat/lon (pure distance math) to enforce a
  **coarser** spacing than the 1.5° baked in at build time. A *finer*
  request than that can't be honored without recomputing from the full
  grid — the endpoint serves what it has rather than fabricating points
  that were already discarded at build time.
- `date` outside the single precomputed date returns
  `{"status": "demo", "fallback_reason": "..."}` with an empty
  recommendation list rather than silently serving the wrong date.
- `depth` isn't used to filter recommendations (each recommendation already
  carries its own real, per-cell `recommended_depths_m`) — the response
  says so in a `note` field rather than silently ignoring the parameter.

## Frontend

A new **Adaptive Sampling** tab in the canonical demo
(`frontend_src/oceanembed_demo.html`, per this repo's own README §"RULE #1
baseline"): an acquisition-score heatmap with numbered markers for the
ranked sites, a ranked table (score / platform badge / re-sampling cadence),
a reasoning + recommended-depths detail panel, and a "View Digital
Profile →" action that jumps to the nearest precomputed Digital Ocean
Profile — mirroring the existing Ocean Map tab's click-to-profile pattern
exactly (`drawMap()` → `drawSamplingMap()`). Runs entirely off the embedded
demo bundle, same as the existing ARGO Guidance / Events / Uncertainty tabs
(not live-fetched) — `frontend_src/assets/oceanembed-client.js` also gained
`activeLearningRecommendations()` / `activeLearningMap()` methods for future
live-wiring or porting to the other two frontend variants.

**Not yet done:** the other two frontend variants
(`frontend_variants/oceanembed_merged.html`, `OceanEmbed_Final-1.html`)
don't have this tab yet — same "partial coverage across variants" pattern
this project already has for other features.

## Tests

`tests/test_active_learning.py` — 17 tests, all passing: acquisition
bounds/ranking/weight-renormalization/explain(), spatial-diversity
min-distance + ranking + "fewer than requested when the grid is too small",
all four platform-recommendation branches, depth-recommendation snapping,
all four adaptive-frequency bands, `event_risk_grid` flagging an injected
anomaly, per-cell metrics shape/consistency, and a full
`build_active_learning_bundle()` end-to-end integration test (rank
ordering, spatial diversity actually enforced, all depths are real model
levels, all platforms/frequency bands are valid). Pre-existing suites
(`test_ml_pipeline.py`, `test_regions_global.py`) still pass unchanged —
the only pre-existing code touched was a non-behavioral refactor of
`ExtremeEventDetector` to extract `_anomaly_fields()` so `detect()` and the
new `event_risk_grid()` share one climatology computation instead of two.

## Honest limitations

- **Weights and thresholds are prototype defaults**, not fit to any
  ground-truth "which platform actually reduced error most" outcome — there
  is no such labeled dataset here to fit them against.
- **In this specific synthetic dataset**, the injected uncertainty hot-spot
  and the injected event-risk hot-spot are co-located (both are artifacts of
  the same synthetic perturbation the data generator adds), so the top-ranked
  recommendations for both demo regions skew heavily toward the AUV / priority
  branch. The Glider / BGC-Argo / Argo branches and the normal/increased/
  intensive frequency bands are real, working code paths — they're exercised
  by `tests/test_active_learning.py`'s targeted unit tests with synthetic
  inputs built to hit each branch, not by this particular dataset's top-8.
  Weights were **not** retuned to force a more "diverse-looking" demo output
  — doing so would be hard-coding the appearance of completeness, which this
  project's own ground rules explicitly forbid.
- **Demo-mode date/depth limitation**: like every other map/profile endpoint
  in this prototype, the active-learning bundle is precomputed for the most
  recent of the 6 demo dates only. A production deployment would compute this
  from the pretrained model on any date on a schedule (e.g. nightly), not per
  request — see `docs/architecture.md`'s "train != inference" principle.
- **`distance_to_observation_deg`** is a simple planar lat/lon Euclidean
  distance, adequate at the scale of these regional grids (a few hundred km
  across) but not a great-circle distance — not used anywhere a more precise
  distance would matter.
