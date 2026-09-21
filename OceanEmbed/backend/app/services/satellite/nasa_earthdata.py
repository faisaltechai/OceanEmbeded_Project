"""
NASA EarthData integration: CMR granule search + PO.DAAC data access.

The Common Metadata Repository (CMR) search API
(https://cmr.earthdata.nasa.gov/search/) is genuinely public -- no login
required to search *what granules exist*. Actually *downloading* a granule
requires an Earthdata Login (EDL) bearer token, obtained via
https://urs.earthdata.nasa.gov -- free account, then a token from
https://urs.earthdata.nasa.gov/profile/apps -- because NASA gates the data
itself, not the catalogue.

This client does both: a key-free CMR search (so "what's available" always
works), and a bearer-token-gated granule fetch (so we degrade to
status="error" with a clear message, never a silent fake value, when
EARTHDATA_BEARER_TOKEN isn't set).
"""
from __future__ import annotations

import os

import httpx

from .envelope import SourceEnvelope, fetch_json, now_iso

CMR_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Real, current short_names for commonly used products.
SENTINEL3_SLSTR_SST_SHORT_NAME = "SENTINEL-3A_SLSTR_L2_SST"
GHRSST_SHORT_NAME = "MUR-JPL-L4-GLOB-v4.1"


async def search_granules(
    client: httpx.AsyncClient,
    short_name: str,
    lat: float,
    lon: float,
    temporal_days: int = 3,
    page_size: int = 5,
) -> SourceEnvelope:
    from datetime import datetime, timedelta, timezone

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=temporal_days)
    params = {
        "short_name": short_name,
        "point": f"{lon},{lat}",
        "temporal": f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')},{end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "page_size": page_size,
        "sort_key": "-start_date",
    }
    try:
        payload = await fetch_json(client, CMR_SEARCH_URL, params=params)
        entries = payload.get("feed", {}).get("entry", [])
        return SourceEnvelope(
            status="live",
            source="NASA EarthData CMR",
            updated=now_iso(),
            resolution="varies by product",
            confidence="High" if entries else "Medium",
            data={
                "short_name": short_name,
                "granule_count": len(entries),
                "granules": [
                    {"id": e.get("id"), "title": e.get("title"), "time_start": e.get("time_start"),
                     "links": [l.get("href") for l in e.get("links", []) if l.get("href")]}
                    for e in entries
                ],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error", source="NASA EarthData CMR", updated=now_iso(),
            resolution="varies by product", confidence="Low", data=None, error=str(exc),
        )


async def fetch_granule_data(client: httpx.AsyncClient, granule_url: str) -> SourceEnvelope:
    """Download an actual granule (e.g. a NetCDF file). Requires EDL bearer token."""
    token = os.getenv("EARTHDATA_BEARER_TOKEN")
    if not token:
        return SourceEnvelope(
            status="error", source="NASA EarthData (PO.DAAC)", updated=now_iso(),
            resolution="varies by product", confidence="Low", data=None,
            error="EARTHDATA_BEARER_TOKEN not configured. Generate one at "
                  "urs.earthdata.nasa.gov/profile/apps and set it in .env.",
        )
    try:
        resp = await client.get(granule_url, headers={"Authorization": f"Bearer {token}"}, follow_redirects=True)
        resp.raise_for_status()
        return SourceEnvelope(
            status="live", source="NASA EarthData (PO.DAAC)", updated=now_iso(),
            resolution="varies by product", confidence="High",
            data={"granule_url": granule_url, "content_length_bytes": len(resp.content),
                  "content_type": resp.headers.get("content-type")},
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error", source="NASA EarthData (PO.DAAC)", updated=now_iso(),
            resolution="varies by product", confidence="Low", data=None, error=str(exc),
        )
