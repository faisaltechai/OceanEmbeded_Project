"""
OceanEmbed - Active Learning: spatial diversity + platform/depth guidance
===========================================================================
Turns a per-cell acquisition score grid into a ranked, spatially-diverse
list of deployment recommendations. Never just "top N pixels": nearby
high-score cells are usually the SAME real-world feature (e.g. one eddy
edge), so picking several adjacent pixels wastes several platforms on one
feature instead of covering the region.
"""
from __future__ import annotations
import numpy as np

DEFAULT_PLATFORM_RULES = {
    "deep_focus_m": 300.0,
    "gradient_threshold_c_per_m": 0.05,
    "event_risk_threshold": 0.7,
}

PLATFORM_RULES_DOC = (
    "Prototype rule-based platform choice (a documented heuristic for this demo, "
    "NOT a scientifically validated deployment doctrine): a cell whose normalized "
    "event_risk is at/above `event_risk_threshold` goes to the fastest-response "
    "platform (AUV); otherwise a cell whose thermocline gradient magnitude is at/"
    "above `gradient_threshold_c_per_m` goes to a Glider (repeated transects "
    "resolve fine vertical structure better than a single profile); otherwise a "
    "cell whose uncertainty is concentrated below `deep_focus_m` goes to a "
    "profiling float capable of biogeochemical sensing (BGC-Argo); everything "
    "else defaults to a standard Argo float. All three thresholds are "
    "configurable in ml/configs/config.yaml: active_learning.platform_rules."
)


def select_diverse_locations(acquisition, lats, lons, top_n=8, min_distance_deg=1.5):
    """
    Greedy selection: rank all cells by score descending, then walk the
    ranked list adding a cell only if it is >= min_distance_deg (simple
    lat/lon Euclidean distance -- adequate at this small regional-grid
    scale, not a great-circle claim) from every already-selected cell.
    Returns a list of (i, j) grid indices in selection order (highest-
    priority first). Returns fewer than top_n only if the grid genuinely
    doesn't have that many mutually-diverse candidates.
    """
    n_lat, n_lon = acquisition.shape
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    flat_order = np.argsort(acquisition.ravel())[::-1]
    selected: list[tuple[int, int]] = []
    for idx in flat_order:
        i, j = np.unravel_index(idx, acquisition.shape)
        lat_i, lon_j = lats[i], lons[j]
        if all(np.hypot(lat_i - lats[si], lon_j - lons[sj]) >= min_distance_deg for si, sj in selected):
            selected.append((int(i), int(j)))
        if len(selected) >= top_n:
            break
    return selected


def recommend_platform(uncertainty_profile, depths, thermocline_gradient, event_risk, rules=None):
    """
    uncertainty_profile: 1D array over depths for this ONE cell (the full
    per-depth uncertainty, not the depth-mean used for scoring) -- used only
    to decide whether uncertainty is concentrated above or below
    `deep_focus_m`. See PLATFORM_RULES_DOC for the full decision rule.
    """
    rules = {**DEFAULT_PLATFORM_RULES, **(rules or {})}
    depths = np.asarray(depths, dtype=float)
    deep_mask = depths >= rules["deep_focus_m"]
    shallow_mask = ~deep_mask

    if event_risk >= rules["event_risk_threshold"]:
        return "AUV"
    if abs(thermocline_gradient) >= rules["gradient_threshold_c_per_m"]:
        return "Glider"
    deep_unc = float(np.mean(uncertainty_profile[deep_mask])) if deep_mask.any() else 0.0
    shallow_unc = float(np.mean(uncertainty_profile[shallow_mask])) if shallow_mask.any() else 0.0
    if deep_unc > shallow_unc:
        return "BGC-Argo"
    return "Argo"


def recommend_depths(depths, thermocline_depth_m, n_deep_anchors=2):
    """
    Recommended sampling depths for a deployment at this cell: the surface,
    the cell's own computed thermocline depth (services.thermocline_metrics
    -- never invented), and the `n_deep_anchors` deepest standard levels
    from the model's own depth configuration (ml/configs/config.yaml:
    depths_m), so subsurface structure below the thermocline is still
    captured. All values are snapped to actual standard levels -- nothing
    here is a depth that isn't already one of the model's own levels.
    """
    levels = sorted(set(float(d) for d in depths))
    nearest_therm = min(levels, key=lambda d: abs(d - thermocline_depth_m))
    anchors = {levels[0], nearest_therm, *levels[-n_deep_anchors:]}
    return sorted(anchors)
