import httpx
from fastapi import APIRouter

from app.services import gateway
from app.services.satellite.nhc_cyclones import fetch_active_storms

router = APIRouter()


@router.get("/cyclones")
async def get_cyclones():
    """
    Live active storm tracks (NOAA NHC, Atlantic/E-Pacific coverage today).
    No synthetic cyclone data exists in this project -- if the live feed is
    unreachable and nothing is cached yet, this returns an empty, clearly-
    labeled result rather than inventing storms, since RULE #2 leaves no
    room for a "demo cyclone" here the way the SST/temperature bundle has a
    labeled synthetic proxy.
    """
    async with httpx.AsyncClient() as client:
        result = await gateway.resolve(
            cache_key="cyclones:active",
            live_fn=lambda: fetch_active_storms(client),
            demo_fn=lambda: {
                "source": "NOAA National Hurricane Center", "updated": None,
                "resolution": "point track, ~6-hourly advisories", "confidence": "Low",
                "data": {"count": 0, "storms": []},
            },
            ttl_bucket="cyclone",
        )
    return result
