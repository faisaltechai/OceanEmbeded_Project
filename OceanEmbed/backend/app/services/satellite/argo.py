"""
Real-time Argo float data via Argovis (https://argovis.colorado.edu).

Argovis mirrors the Argo Global Data Assembly Center (GDAC) into a public,
key-free REST API -- this is the same approach the oceanographic research
community uses day to day, and is far more practical than parsing GDAC's
raw NetCDF file tree directly for a web backend. Docs:
https://argovis-api.colorado.edu/docs

If Argovis is unreachable, `argo_gdac.py`'s comments describe the raw-GDAC
fallback (ifremer.fr rsync mirror) for a production hardening pass.
"""
from __future__ import annotations

import httpx

from .envelope import SourceEnvelope, fetch_json, now_iso, confidence_from_qc

ARGOVIS_BASE = "https://argovis-api.colorado.edu"


async def fetch_nearby_floats(
    client: httpx.AsyncClient,
    lat: float,
    lon: float,
    radius_km: float = 300,
    api_key: str | None = None,
) -> SourceEnvelope:
    """Nearby Argo float profiles.

    Argovis requires a free API key as of their v3 API (register at
    argovis-keygen.colorado.edu) -- unlike ERDDAP this is not fully
    anonymous, so we pass it through if configured and degrade gracefully
    if not.
    """
    # Argovis expects a GeoJSON polygon or a center+radius query depending on
    # endpoint version; this uses the documented nearby-profile search.
    url = f"{ARGOVIS_BASE}/argo"
    params = {
        "center": f"[{lon},{lat}]",
        "radius": int(radius_km * 1000),  # meters
    }
    headers = {"x-argokey": api_key} if api_key else {}
    try:
        payload = await fetch_json(client, url, params=params, headers=headers)
        profiles = payload if isinstance(payload, list) else payload.get("profiles", [])
        qc_flags = [p.get("position_qc", 1) for p in profiles if isinstance(p, dict)]
        return SourceEnvelope(
            status="live",
            source="Argo GDAC (via Argovis)",
            updated=now_iso(),
            resolution="point observations (irregular)",
            confidence=confidence_from_qc(qc_flags, good_flags=(1,)),
            data={"count": len(profiles), "profiles": profiles},
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error",
            source="Argo GDAC (via Argovis)",
            updated=now_iso(),
            resolution="point observations (irregular)",
            confidence="Low",
            data=None,
            error=str(exc),
        )
