# Smart ARGO Guidance

`ml/evaluation/services.py: ArgoGuidanceService` scores every grid cell using three
quantities the pipeline actually computes (never hardcoded):

```
priority_score = w_uncertainty * norm(uncertainty)
               + w_density     * norm(1 - observation_density)
               + w_anomaly     * norm(|temperature_anomaly|)
```

- `uncertainty`: mean MC-Dropout std across depths for that cell/date.
- `observation_density`: count of synthetic-Argo hits at that grid cell across all
  demo dates (stand-in for real Argo float density).
- `anomaly`: (temperature − demo-period climatology) / climatology std, averaged
  across depths.

Weights (`w_uncertainty=0.5, w_density=0.3, w_anomaly=0.2`) are configurable and
documented here rather than buried in code. Each returned zone includes a short
plain-language `reason` built from which component(s) crossed a 0.6-normalized
threshold, answering "why is this location recommended?" directly from the numbers
that produced the score — never a canned string.

**Honest limitation:** "observation density" is measured against the synthetic Argo
subsample, not real Argo GDAC float positions.
