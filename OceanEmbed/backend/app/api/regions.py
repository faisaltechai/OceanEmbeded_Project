from fastapi import APIRouter, Request, HTTPException

from app.core.regions_global import REGIONS, region_for_point

router = APIRouter()


@router.get("/regions")
def list_regions(request: Request):
    """
    Full global region catalogue (dropdown scope from the brief). Regions
    with a precomputed demo bundle also get grid_shape/bbox from the live
    store; the rest are real bbox/name metadata with live satellite clients
    wired up, but no synthetic map grid -- `has_demo_bundle: false` tells
    the frontend to route Overview/Profile calls through the live gateway
    only for those, rather than expecting a map layer that doesn't exist.
    """
    store = request.app.state.store
    out = []
    for key, meta in REGIONS.items():
        entry = {"key": key, "name": meta["name"], "bbox": meta["bbox"], "has_demo_bundle": meta["has_demo_bundle"]}
        if meta["has_demo_bundle"] and key in store.region_keys():
            r = store.get_region(key)
            entry["grid_shape"] = [len(r["lats"]), len(r["lons"])]
        out.append(entry)
    return {"regions": out}


@router.get("/regions/locate")
def locate_region(lat: float, lon: float):
    """Supports 'click anywhere on Earth' -- resolves a clicked point to its
    containing ocean basin region key."""
    key = region_for_point(lat, lon)
    if not key:
        raise HTTPException(404, "No known ocean region contains this point.")
    return {"lat": lat, "lon": lon, "region": key, "region_name": REGIONS[key]["name"]}


@router.get("/dates")
def list_dates(request: Request, region: str):
    store = request.app.state.store
    r = store.get_region(region)
    return {"region": region, "dates": r["dates"]}
