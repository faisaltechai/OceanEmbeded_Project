# Validation

Computed in `ml/evaluation/evaluate.py`, never invented:

- **Temporal holdout**: the last of 6 demo dates is held out entirely from training
  (`ml/data/data_pipeline.py: temporal_split`) — no random shuffling across time, so
  there is no leakage between train and test periods.
- **Model vs. baseline**: `climatology_baseline()` (per-depth mean profile from the
  training period — the simplest defensible baseline per the brief's section 23) is
  compared against OceanEmbed on the exact same held-out samples.
- **Independent ARGO comparison**: `evaluate_against_argo()` compares model predictions
  at each synthetic-Argo profile's (date, lat, lon) against that profile — a comparison
  that never touched training.
- **Depth-wise breakdown**: RMSE/Bias/Correlation reported at every one of the 16
  standard depth levels, for both model and baseline (`validate_depthwise`).

Actual numbers from the last pipeline run (regenerable any time via
`python ml/inference/build_demo_bundle.py`) are in `data/demo/backend_bundle.json ->
regions.<region>.metrics` and `.argo_validation` — see the Validation tab in the demo
UI for the live figures, which will differ slightly run-to-run only if you change the
random seed in `config.yaml`.

**Honest limitation:** all of the above is against the synthetic proxy dataset, not
real Argo/GLORYS. The evaluation *code* is the real, reusable deliverable — point it
at real NetCDF-derived arrays and the same functions produce real validation numbers.
