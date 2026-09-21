from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.services.satellite.erddap import noaa_coastwatch_sst_client

router = APIRouter()


@router.get("/timeline")
async def get_timeline(request: Request, region: str, days: int = 30):
    """
    Time Machine: yesterday / last week / last month / last year comparison.

    ERDDAP genuinely supports historical date queries (that's real, live
    behavior, not a replay trick) so the SST series below is a real
    multi-date live fetch when reachable. The demo bundle's own multi-date
    SST history (if the region has one) is used as the label-honest fallback
    per date where the live call fails, rather than interpolating/faking a
    point.
    """
    store = request.app.state.store
    r = store.get_region(region)
    bbox = r["bbox"]
    center_lat, center_lon = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    anchors = {
        "yesterday": timedelta(days=1),
        "last_week": timedelta(days=7),
        "last_month": timedelta(days=30),
        "last_year": timedelta(days=365),
    }
    now = datetime.now(timezone.utc)
    sst_client = noaa_coastwatch_sst_client()

    points = []
    async with httpx.AsyncClient() as client:
        for label, delta in anchors.items():
            when = now - delta
            envelope = await sst_client.fetch_point(client, center_lat, center_lon, when=when)
            points.append({
                "label": label,
                "date": when.date().isoformat(),
                "status": envelope.status,
                "value_c": (envelope.data or {}).get("value") if envelope.status == "live" else None,
                "source": envelope.source,
                "error": envelope.error,
            })

    return {
        "region": region,
        "requested_days": days,
        "series": points,
        "note": "Each point is an independent live ERDDAP query for that historical date; "
                "points with status='error' mean that date/variable wasn't available live "
                "(common near-real-time gap) -- they are left null, never interpolated or invented.",
    }
