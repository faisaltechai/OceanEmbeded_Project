"""
OceanEmbed - Active Learning & Adaptive Sampling
==================================================
Turns the existing MC-Dropout uncertainty engine (ml/models/model.py) and
Smart ARGO guidance (ml/evaluation/services.py) into an autonomous
observation-planning layer. For every region/date the pipeline already
has a full-grid temperature+uncertainty field for, this package produces:

  1. a full per-cell metrics grid                 (uncertainty_map.py)
  2. an explainable, config-weighted acquisition score
                                                     (acquisition.py)
  3. a spatially-diverse, platform/depth-aware ranked deployment list
                                                     (sampling_optimizer.py)
  4. a per-recommendation adaptive re-sampling cadence
                                                     (adaptive_frequency.py)

IMPORTANT: no new uncertainty algorithm is introduced anywhere in this
package. Every number here is derived from quantities `model.py:
predict_with_uncertainty`, `services.py: ArgoGuidanceService`,
`services.py: thermocline_metrics`, and `services.py: ExtremeEventDetector`
already compute. This module is additive: it does not change the
existing `/api/argo/guidance` behavior or the ArgoGuidanceService it's
built on. See docs/active-learning.md for the full methodology write-up
and the honest limitations of each prototype threshold/weight below.
"""
from __future__ import annotations
import numpy as np

from ml.evaluation.services import ExtremeEventDetector, thermocline_metrics
from ml.inference.inference import confidence_category
from .uncertainty_map import compute_grid_metrics
from .acquisition import AcquisitionFunction, DEFAULT_WEIGHTS
from .sampling_optimizer import (
    select_diverse_locations, recommend_platform, recommend_depths, PLATFORM_RULES_DOC,
    DEFAULT_PLATFORM_RULES,
)
from .adaptive_frequency import recommend_frequency, DEFAULT_INTERVALS_DAYS
from .models import SamplingRecommendation

__all__ = [
    "compute_grid_metrics", "AcquisitionFunction", "DEFAULT_WEIGHTS",
    "select_diverse_locations", "recommend_platform", "recommend_depths",
    "recommend_frequency", "SamplingRecommendation", "PLATFORM_RULES_DOC",
    "build_active_learning_bundle",
]


def build_active_learning_bundle(temperature_field, uncertainty_field, mixed_layer_depth_field,
                                  depths, lats, lons, dates, date_index, observation_density,
                                  argo_records, config=None):
    """
    The single entry point both `ml/inference/build_demo_bundle.py` (precompute
    time) and, if ever needed, a live backend job would call. Never called
    per-API-request in this project's "train != inference, cache the rest"
    architecture (see docs/architecture.md) -- the backend only ever reads the
    dict this returns, already stored in the bundle.

    config: the `active_learning` sub-dict of ml/configs/config.yaml (or None
    to fall back to this package's own documented defaults).
    """
    config = config or {}
    event_detector = ExtremeEventDetector(std_threshold=config.get("event_std_threshold", 1.5))

    metrics = compute_grid_metrics(
        temperature_field, uncertainty_field, mixed_layer_depth_field,
        depths, lats, lons, dates, date_index, observation_density, argo_records, event_detector,
    )

    acquisition_fn = AcquisitionFunction(config.get("acquisition_weights"))
    acquisition, components = acquisition_fn.score(metrics)

    div_cfg = config.get("spatial_diversity", {})
    top_n = int(div_cfg.get("default_top_n", 8))
    min_distance_deg = float(div_cfg.get("min_distance_deg", 1.5))
    selected = select_diverse_locations(acquisition, lats, lons, top_n=top_n, min_distance_deg=min_distance_deg)

    platform_rules = {**DEFAULT_PLATFORM_RULES, **config.get("platform_rules", {})}
    freq_cfg = config.get("frequency_thresholds", {})
    low_thresh = float(freq_cfg.get("low_max_c", 0.35))
    high_thresh = float(freq_cfg.get("medium_max_c", 0.8))
    intervals = {**DEFAULT_INTERVALS_DAYS, **freq_cfg.get("intervals_days", {})}

    recommendations = []
    for rank, (i, j) in enumerate(selected, start=1):
        comp_here = {k: float(v[i, j]) for k, v in components.items()}
        reason = acquisition_fn.explain(comp_here)
        platform = recommend_platform(
            uncertainty_field[date_index, i, j, :], depths,
            metrics["thermocline_gradient"][i, j], metrics["event_risk"][i, j], platform_rules,
        )
        cell_thermocline_depth = thermocline_metrics(depths, temperature_field[date_index, i, j, :])["thermocline_depth_m"]
        rec_depths = recommend_depths(depths, cell_thermocline_depth)
        freq = recommend_frequency(
            metrics["uncertainty_score"][i, j],
            is_extreme_event=bool(metrics["event_risk"][i, j] >= platform_rules["event_risk_threshold"]),
            low_thresh=low_thresh, high_thresh=high_thresh, intervals=intervals,
        )
        rec = SamplingRecommendation(
            rank=rank, latitude=float(lats[i]), longitude=float(lons[j]),
            priority_score=round(float(acquisition[i, j]), 4),
            uncertainty=round(float(metrics["uncertainty_score"][i, j]), 4),
            reason=reason, recommended_platform=platform, recommended_depths_m=rec_depths,
            adaptive_sampling=freq,
        )
        recommendations.append(rec.to_dict())

    confidence_grid = [[confidence_category(v) for v in row] for row in metrics["uncertainty_score"]]

    return {
        "status": "model-derived",
        "method": (
            "acquisition_score = weighted sum of normalized uncertainty, |anomaly|, "
            "|thermocline_gradient|, event_risk, and observation_gap (all min-max "
            "normalized over the grid); weights are configurable prototype "
            "parameters, not a scientifically optimized allocation -- see "
            "docs/active-learning.md and ml/configs/config.yaml: active_learning."
        ),
        "platform_methodology": PLATFORM_RULES_DOC,
        "weights_used": acquisition_fn.weights,
        "spatial_diversity": {"min_distance_deg": min_distance_deg, "top_n": top_n},
        "date_used": dates[date_index],
        "recommendations": recommendations,
        "grid": {
            "lats": [float(x) for x in lats], "lons": [float(x) for x in lons],
            "uncertainty_score": metrics["uncertainty_score"].round(4).tolist(),
            "acquisition_score": acquisition.round(4).tolist(),
            "confidence_category": confidence_grid,
            "components": {k: v.round(4).tolist() for k, v in components.items()},
        },
    }
