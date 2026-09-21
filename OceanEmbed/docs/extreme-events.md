# Extreme Ocean Event Detection

**Prototype methodology — threshold-based**, per `ml/evaluation/services.py:
ExtremeEventDetector`. A grid cell/date is flagged only when it passes at least one of:

1. **Temperature anomaly**: at some depth, |actual − demo-period climatology| exceeds
   BOTH `subsurface_heat_anomaly_std` standard deviations (config, default 1.5σ) AND a
   minimum absolute departure of 0.8°C. Requiring both avoids over-flagging from the
   small (6-date) demo climatology's noisy standard deviation.
2. **Mixed-layer collapse**: the mixed-layer depth at that date is ≤60% of its
   demo-period climatological value (a shallow, collapsed mixed layer is a common
   precursor/co-occurrence of subsurface heat events).

Each event reports its `type` (Heat/Cold Anomaly), `severity` (High/Medium from the
z-score / MLD-ratio magnitude), the specific depth and reason string, and
`detection_method` so nothing is presented as an authoritative classification. Results
are capped to the top N most severe per region (config: `max_events_per_region=25`) for
demo readability — this cap is documented, not a silent drop.

**Honest limitation:** this is not a validated marine-heatwave or cyclone-detection
algorithm (e.g. no Hobday et al. MHW definition, no cyclone track ingestion) — it is a
transparent statistical threshold prototype, explicitly labeled as such per the SIH
brief's requirement.
