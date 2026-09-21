from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
def get_validation(request: Request, region: str):
    store = request.app.state.store
    r = store.get_region(region)
    return {
        "region": region,
        "overall": r["metrics"]["overall"],
        "depthwise_model": r["metrics"]["depthwise_model"],
        "depthwise_baseline_climatology": r["metrics"]["depthwise_baseline"],
        "against_argo": r["argo_validation"],
        "n_test_samples": r["metrics"]["n_test_samples"],
        "notes": "model/baseline computed on a TEMPORALLY held-out date (last demo date); "
                 "against_argo computed independently against synthetic Argo-like profiles.",
    }
