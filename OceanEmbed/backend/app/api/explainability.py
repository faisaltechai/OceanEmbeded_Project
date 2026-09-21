from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
def get_explainability(request: Request, region: str):
    store = request.app.state.store
    r = store.get_region(region)
    return {
        "region": region,
        "method": "feature ablation (each surface variable replaced by its dataset mean; "
                   "importance = mean absolute shift in the predicted profile)",
        "contributions": r["explainability"],
    }
