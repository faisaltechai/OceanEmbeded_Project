"""
OceanEmbed - Active Learning: acquisition function
=====================================================
Combines the per-cell metrics from uncertainty_map.py into a single,
explainable "how valuable would a new real observation be here" score:

    Acquisition Score =
        w_uncertainty          * norm(uncertainty_score)
      + w_anomaly              * norm(|temperature_anomaly|)
      + w_thermocline_gradient * norm(|thermocline_gradient|)
      + w_event_risk           * norm(event_risk)
      + w_observation_gap      * norm(1 - norm(observation_density))

Weights are entirely configuration-driven (ml/configs/config.yaml:
active_learning.acquisition_weights) and are PROTOTYPE DEFAULTS, not a
scientifically optimized/validated allocation -- see docs/active-learning.md.

This deliberately EXTENDS ArgoGuidanceService (ml/evaluation/services.py,
which scores uncertainty + sparsity + anomaly) rather than replacing it:
ArgoGuidanceService is untouched and still powers the original
`/api/argo/guidance` endpoint. This module adds the two components the
active-learning upgrade brief asks for beyond it (thermocline_gradient,
event_risk) and generalizes the explanation/weighting machinery around it.
"""
from __future__ import annotations
import numpy as np

DEFAULT_WEIGHTS = {
    "uncertainty": 0.35,
    "anomaly": 0.20,
    "thermocline_gradient": 0.20,
    "event_risk": 0.15,
    "observation_gap": 0.10,
}

_LABELS = {
    "uncertainty": "high prediction uncertainty",
    "anomaly": "strong temperature anomaly vs. climatology",
    "thermocline_gradient": "steep thermocline gradient",
    "event_risk": "elevated extreme-event risk",
    "observation_gap": "sparse nearby observations",
}


def _minmax01(a):
    """Same min-max convention as ArgoGuidanceService._minmax, tolerant of NaN
    (e.g. a region with zero recorded Argo observations)."""
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    out = (a - lo) / (hi - lo + 1e-9)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


class AcquisitionFunction:
    def __init__(self, weights: dict | None = None):
        merged = {**DEFAULT_WEIGHTS, **(weights or {})}
        total = sum(merged.values()) or 1.0
        # Weights always renormalized to sum to 1, even if config supplies
        # odd values -- documented behavior, not a silent "correction".
        self.weights = {k: v / total for k, v in merged.items()}

    def score(self, metrics: dict):
        """metrics: the dict returned by uncertainty_map.compute_grid_metrics.
        Returns (acquisition_grid, components) where components are the
        individual normalized (0-1) contributions, useful both for the
        weighted sum and for `explain()` below."""
        components = {
            "uncertainty": _minmax01(metrics["uncertainty_score"]),
            "anomaly": _minmax01(np.abs(metrics["temperature_anomaly"])),
            "thermocline_gradient": _minmax01(np.abs(metrics["thermocline_gradient"])),
            "event_risk": _minmax01(metrics["event_risk"]),
            "observation_gap": _minmax01(1 - _minmax01(metrics["observation_density"])),
        }
        acquisition = sum(self.weights[k] * components[k] for k in self.weights)
        return acquisition, components

    @staticmethod
    def explain(components_at_cell: dict, threshold: float = 0.6) -> str:
        """Builds the plain-language `reason` string from whichever
        components actually crossed the threshold at this cell -- never a
        canned string, same convention as ArgoGuidanceService.explain."""
        reasons = [_LABELS[k] for k, v in components_at_cell.items() if k in _LABELS and v > threshold]
        return "; ".join(reasons) if reasons else "moderate combined score across all factors"
