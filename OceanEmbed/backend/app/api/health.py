import os

from fastapi import APIRouter, Request

from app.cache.redis_cache import get_cache

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    store = request.app.state.store
    cache = get_cache()
    await cache._ensure_backend()  # force backend resolution so status is accurate, not "unknown"
    live_sources_configured = {
        "copernicus_marine": bool(os.getenv("COPERNICUS_MARINE_USERNAME")),
        "nasa_earthdata": bool(os.getenv("EARTHDATA_BEARER_TOKEN")),
        "cds_era5": bool(os.getenv("CDS_API_KEY")),
        "cdse_sentinel3": bool(os.getenv("CDSE_USERNAME")),
        "noaa_coastwatch_erddap": True,  # public, no key needed
        "argovis": True,  # public search; API key optional for higher rate limits
        "noaa_nhc_cyclones": True,  # public, no key needed
    }
    return {
        "status": "ok",
        "dataset_label": store.dataset_label,
        "regions_loaded": store.region_keys(),
        "cache_backend": cache.backend_name,
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "live_sources_configured": live_sources_configured,
    }
