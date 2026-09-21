from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/guidance")
def get_argo_guidance(request: Request, region: str):
    store = request.app.state.store
    r = store.get_region(region)
    return {
        "region": region, "n_existing_argo_profiles": r["n_argo_profiles_total"],
        "recommended_observation_zones": r["argo_guidance"],
        "scoring": "priority_score = w_u*uncertainty + w_d*(1-observation_density) + w_a*anomaly, "
                   "all min-max normalized (see ml/evaluation/services.py:ArgoGuidanceService)",
    }
