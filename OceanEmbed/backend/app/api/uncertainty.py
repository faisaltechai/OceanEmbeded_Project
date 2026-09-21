from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
def get_uncertainty(request: Request, region: str):
    store = request.app.state.store
    r = store.get_region(region)
    ml = r["map_layers"]
    return {
        "region": region, "date": ml["date_used"], "lats": r["lats"], "lons": r["lons"],
        "uncertainty_mean_over_depth": ml["uncertainty_surface_mean"],
        "method": "Monte-Carlo Dropout (N stochastic forward passes, std across passes)",
        "units": "degC",
    }
