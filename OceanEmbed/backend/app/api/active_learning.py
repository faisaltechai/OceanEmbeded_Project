"""
OceanEmbed - Active Learning & Adaptive Sampling API
=======================================================
Serves the precomputed active-learning bundle (ml/active_learning/, run once
per region/date in ml/inference/build_demo_bundle.py -- NEVER recomputed or
re-inferred here) and applies cheap, model-free request-time filtering on
top of it:
  - `number_of_points` -- slice the precomputed, already-ranked candidate pool
  - `platform`         -- filter to a single recommended platform
  - `minimum_distance` -- re-run the spatial-diversity filter on the already-
                           computed candidate list's own lat/lon (pure
                           distance math, no model access) to enforce a
                           COARSER spacing than what was precomputed
This keeps the "never retrain / never recompute an expensive prediction on
a request" contract (docs/architecture.md) while still honoring every query
parameter the upgrade brief asks for.

Demo-mode limitation (documented, not hidden): the bundle only precomputes
this for the most recent date in the demo period, same as every other
map/profile endpoint in this prototype. A `date` outside that range returns
`status: "demo"` with an explicit `fallback_reason` rather than silently
serving the wrong date's numbers (section 7 of the upgrade brief: "never
silently fall back from LIVE to DEMO").
"""
from __future__ import annotations
import math
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()

VALID_PLATFORMS = {"AUV", "Glider", "Argo", "BGC-Argo"}


def _greedy_min_distance(points: list[dict], min_distance_deg: float) -> list[dict]:
    """`points` are already sorted by priority/rank; this re-applies a
    (possibly stricter) spatial-diversity filter using only their own
    lat/lon -- no model or grid access needed, so it's safe to run per
    request."""
    selected: list[dict] = []
    for p in points:
        if all(
            math.hypot(p["latitude"] - s["latitude"], p["longitude"] - s["longitude"]) >= min_distance_deg
            for s in selected
        ):
            selected.append(p)
    return selected


def _get_active_learning(request: Request, region: str):
    store = request.app.state.store
    r = store.get_region(region)
    al = r.get("active_learning")
    if al is None:
        raise HTTPException(status_code=404, detail=f"No active-learning bundle for region '{region}'")
    return r, al


@router.get("/recommendations")
def get_recommendations(
    request: Request,
    region: str,
    number_of_points: int | None = Query(default=None, ge=1),
    platform: str | None = None,
    minimum_distance: float | None = Query(default=None, gt=0),
    date: str | None = None,
    depth: float | None = None,
):
    r, al = _get_active_learning(request, region)

    if date is not None and date != al["date_used"]:
        return {
            "region": region, "status": "demo",
            "fallback_reason": (
                f"active-learning recommendations are precomputed for {al['date_used']} only in demo "
                "mode (see docs/architecture.md's 'train != inference, cache the rest' pattern); "
                f"requested date '{date}' is not available without a production deployment that "
                "computes on demand from the pretrained model."
            ),
            "date_used": al["date_used"], "recommendations": [],
        }

    recs = al["recommendations"]

    if platform is not None:
        if platform not in VALID_PLATFORMS:
            raise HTTPException(status_code=422, detail=f"platform must be one of {sorted(VALID_PLATFORMS)}")
        recs = [rec for rec in recs if rec["recommended_platform"] == platform]

    depth_note = None
    if depth is not None:
        # Each recommendation already carries its own `recommended_depths_m`
        # derived from that cell's real profile+thermocline -- there's no
        # honest way to further filter by a single requested depth without
        # inventing per-depth scores, so this is documented rather than
        # silently ignored.
        depth_note = (
            "`depth` is not used to filter recommendations: each recommendation already "
            "carries its own recommended_depths_m computed from that cell's real profile."
        )

    if minimum_distance is not None:
        base_min_distance = al["spatial_diversity"]["min_distance_deg"]
        if minimum_distance > base_min_distance:
            recs = _greedy_min_distance(recs, minimum_distance)
        # A minimum_distance SMALLER than what was precomputed can't be honored
        # without recomputing from the full grid (points closer together were
        # already discarded at build time) -- served as-is, which already
        # satisfies the finer constraint; never silently over-served.

    if number_of_points is not None:
        recs = recs[: number_of_points]

    out = {
        "region": region, "region_name": r["region_name"], "status": al["status"],
        "date_used": al["date_used"], "n_existing_argo_profiles": r["n_argo_profiles_total"],
        "method": al["method"], "platform_methodology": al["platform_methodology"],
        "weights_used": al["weights_used"], "spatial_diversity": al["spatial_diversity"],
        "recommendations": recs,
    }
    if depth_note:
        out["note"] = depth_note
    return out


@router.get("/map")
def get_active_learning_map(request: Request, region: str):
    """
    Full-grid uncertainty/acquisition heatmaps + the individual normalized
    components (anomaly, thermocline_gradient, event_risk, observation_gap)
    for the Adaptive Sampling map view. Same precomputed, no-recompute
    contract as /recommendations above.
    """
    r, al = _get_active_learning(request, region)
    return {
        "region": region, "status": al["status"], "date_used": al["date_used"],
        "method": al["method"], "weights_used": al["weights_used"],
        **al["grid"],
    }
