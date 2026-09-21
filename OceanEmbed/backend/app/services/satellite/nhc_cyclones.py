"""
Live active-cyclone tracks via NOAA's National Hurricane Center.

https://www.nhc.noaa.gov/CurrentStorms.json is a real, public, key-free
JSON feed NHC publishes and updates continuously during storm season --
the most practical "no fake data" cyclone source available without a paid
subscription. It covers Atlantic + East/Central Pacific; for Indian Ocean /
West Pacific cyclones (relevant to the Bay of Bengal scope) the equivalent
public feed is the Joint Typhoon Warning Center (JTWC) at
https://www.metoc.navy.mil/jtwc/jtwc.html?best-tracks (no clean JSON API,
HTML/text products) or IMD's public cyclone bulletins -- both noted as a
follow-up integration rather than faked here.
"""
from __future__ import annotations

import httpx

from .envelope import SourceEnvelope, fetch_json, now_iso

NHC_CURRENT_STORMS_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"


async def fetch_active_storms(client: httpx.AsyncClient) -> SourceEnvelope:
    try:
        payload = await fetch_json(client, NHC_CURRENT_STORMS_URL)
        storms = payload.get("activeStorms", [])
        return SourceEnvelope(
            status="live",
            source="NOAA National Hurricane Center",
            updated=payload.get("issuance", now_iso()),
            resolution="point track, ~6-hourly advisories",
            confidence="High",
            data={
                "count": len(storms),
                "storms": [
                    {
                        "name": s.get("name"),
                        "basin": s.get("basin"),
                        "classification": s.get("classification"),
                        "lat": s.get("latitudeNumeric"),
                        "lon": s.get("longitudeNumeric"),
                        "intensity_kt": s.get("intensity"),
                        "pressure_mb": s.get("pressure"),
                        "movement": s.get("movement"),
                        "public_advisory_url": s.get("publicAdvisory", {}).get("url"),
                    }
                    for s in storms
                ],
            },
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error", source="NOAA National Hurricane Center", updated=now_iso(),
            resolution="point track, ~6-hourly advisories", confidence="Low", data=None, error=str(exc),
        )
