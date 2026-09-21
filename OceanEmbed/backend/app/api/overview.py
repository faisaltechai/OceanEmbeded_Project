from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.services import gateway
from app.services.satellite import argo as argo_client
from app.services.satellite.erddap import noaa_coastwatch_sst_client

router = APIRouter()


@router.get("/overview")
async def get_overview(request: Request, region: str):
    """
    Overview panel cards. Scientific numbers (RMSE, model confidence,
    Argo-count-used-in-training, event counts) come from the precomputed
    pipeline bundle -- those are not live-refreshable without retraining and
    were never claimed to be. The clearly-live-refreshable cards (current
    SST at the region centroid, count of Argo floats reporting nearby right
    now) go through the live gateway so the "LIVE" badge is only ever true
    when a live call actually succeeded this run.
    """
    store = request.app.state.store
    try:
        r = store.get_region(region)
    except HTTPException:
        raise

    bbox = r["bbox"]  # [south, west, north, east]
    center_lat = (bbox[0] + bbox[2]) / 2
    center_lon = (bbox[1] + bbox[3]) / 2
    ml = r["map_layers"]

    async with httpx.AsyncClient() as client:
        sst_client = noaa_coastwatch_sst_client()

        live_sst = await gateway.resolve(
            cache_key=f"overview:sst:{region}",
            live_fn=lambda: sst_client.fetch_point(client, center_lat, center_lon),
            demo_fn=lambda: {
                "source": store.dataset_label, "updated": ml["date_used"],
                "resolution": "1.0 deg (demo grid)", "confidence": "Medium",
                "data": {"value": r["overview_cards"]["avg_surface_temperature_c"]},
            },
            ttl_bucket="sst",
        )
        live_argo = await gateway.resolve(
            cache_key=f"overview:argo:{region}",
            live_fn=lambda: argo_client.fetch_nearby_floats(client, center_lat, center_lon, radius_km=500),
            demo_fn=lambda: {
                "source": store.dataset_label, "updated": ml["date_used"],
                "resolution": "point observations (demo)", "confidence": "Medium",
                "data": {"count": r["overview_cards"]["active_argo_floats"]},
            },
            ttl_bucket="argo",
        )

    return {
        "region": region,
        "region_name": r["region_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cards": {
            "average_surface_temperature_c": {
                "value": (live_sst["data"] or {}).get("value") if live_sst["status"] == "live" else r["overview_cards"]["avg_surface_temperature_c"],
                "status": live_sst["status"], "source": live_sst["source"], "updated": live_sst["updated"],
            },
            "ocean_heat_content_j_m2": {
                "value": r["overview_cards"]["ocean_heat_content_j_m2"],
                "status": "demo", "source": store.dataset_label, "updated": ml["date_used"],
                "note": "Derived from the reconstruction model; not a directly-observed satellite quantity.",
            },
            "detected_events": {"value": len(r["events"]), "status": "demo", "source": store.dataset_label, "updated": ml["date_used"]},
            "high_confidence_coverage_pct": {"value": r["overview_cards"]["high_confidence_coverage_pct"], "status": "demo", "source": store.dataset_label, "updated": ml["date_used"]},
            "model_rmse_c": {"value": r["metrics"]["overall"]["rmse"], "status": "demo", "source": store.dataset_label, "updated": ml["date_used"]},
            "active_argo_floats": {
                "value": (live_argo["data"] or {}).get("count") if live_argo["status"] == "live" else r["overview_cards"]["active_argo_floats"],
                "status": live_argo["status"], "source": live_argo["source"], "updated": live_argo["updated"],
            },
            "live_data_streams": {
                "value": sum(1 for s in [live_sst, live_argo] if s["status"] == "live"),
                "status": "computed", "source": "OceanEmbed gateway",
                "updated": datetime.now(timezone.utc).isoformat(),
            },
            "last_updated": {"value": datetime.now(timezone.utc).isoformat(), "status": "computed"},
        },
    }
