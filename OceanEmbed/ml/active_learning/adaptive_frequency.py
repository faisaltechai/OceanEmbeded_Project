"""
OceanEmbed - Active Learning: adaptive sampling frequency
============================================================
Maps a cell's uncertainty (+ whether it's currently flagged as a
high-event-risk cell) to a recommended re-sampling cadence. The low/medium
uncertainty boundaries default to the SAME thresholds already used for
`confidence_category` (ml/inference/inference.py) -- reused, not
reinvented -- extended with an "intensive"/"priority" tier for the
Low-confidence and event-flagged cases:

    LOW UNCERTAINTY      (<= low_thresh)            -> normal sampling
    MEDIUM UNCERTAINTY   (low_thresh - high_thresh)  -> increased sampling
    HIGH UNCERTAINTY     (> high_thresh)             -> intensive sampling
    EXTREME EVENT        (event_risk over threshold) -> priority sampling

All thresholds and intervals are configuration-driven
(ml/configs/config.yaml: active_learning.frequency_thresholds) and are
documented prototype defaults, not an operationally validated cadence.
"""
from __future__ import annotations

DEFAULT_INTERVALS_DAYS = {
    "normal": 10,
    "increased": 5,
    "intensive": 2,
    "priority": 1,
}


def recommend_frequency(uncertainty_score, is_extreme_event, low_thresh=0.35, high_thresh=0.8,
                         intervals=None):
    intervals = {**DEFAULT_INTERVALS_DAYS, **(intervals or {})}
    if is_extreme_event:
        level = "priority"
        reason = "cell is currently flagged as an elevated-risk event (see event_risk)"
    elif uncertainty_score > high_thresh:
        level = "intensive"
        reason = f"uncertainty {uncertainty_score:.2f}\u00b0C is in the Low-confidence band (> {high_thresh}\u00b0C)"
    elif uncertainty_score > low_thresh:
        level = "increased"
        reason = f"uncertainty {uncertainty_score:.2f}\u00b0C is in the Medium-confidence band ({low_thresh}-{high_thresh}\u00b0C)"
    else:
        level = "normal"
        reason = f"uncertainty {uncertainty_score:.2f}\u00b0C is in the High-confidence band (<= {low_thresh}\u00b0C)"
    return {
        "priority_level": level,
        "recommended_sampling_interval_days": intervals[level],
        "reason": reason,
    }
