"""
Two more real integrations, grouped here because both are "search/request
now, data arrives asynchronously" APIs rather than instant REST GETs -- an
important, real constraint to design the scheduler and cache around (see
scheduler/jobs.py).

1) ERA5 reanalysis via the Copernicus Climate Data Store (CDS).
   The `cdsapi` client submits a request that CDS queues and processes
   (often 30s-few minutes), then returns a download URL. It is NOT
   instant, so it must never sit in the request/response path of a user
   clicking a button -- the scheduler pre-fetches it on a cadence and the
   API layer only ever serves the last cached result.

2) Sentinel-3 granule search via the Copernicus Data Space Ecosystem (CDSE)
   OData API -- search is public; download requires an OAuth2 token from
   CDSE's identity service (real, current as of the 2023 migration off the
   old SciHub).
"""
from __future__ import annotations

import os

import httpx

from .envelope import SourceEnvelope, fetch_json, now_iso

CDSE_IDENTITY_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def _cds_credentials_configured() -> bool:
    return bool(os.getenv("CDS_API_KEY"))


def _blocking_cds_request(variable: str, lat: float, lon: float, date_str: str) -> str:
    """Runs in a worker thread. Requires `pip install cdsapi` and ~/.cdsapirc
    or CDSAPI_KEY/CDSAPI_URL env vars. Returns a local file path.
    """
    import cdsapi
    import tempfile

    c = cdsapi.Client()
    out_path = tempfile.mktemp(suffix=".nc")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variable,
            "date": date_str,
            "time": "00:00",
            "area": [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],  # N,W,S,E
            "format": "netcdf",
        },
        out_path,
    )
    return out_path


async def fetch_era5_point(variable: str, lat: float, lon: float, date_str: str) -> SourceEnvelope:
    import asyncio

    if not _cds_credentials_configured():
        return SourceEnvelope(
            status="error", source="Copernicus CDS (ERA5)", updated=now_iso(),
            resolution="0.25 deg", confidence="Low", data=None,
            error="CDS_API_KEY not configured. Register free at cds.climate.copernicus.eu "
                  "and set CDS_API_KEY (and CDSAPI_URL if non-default) in .env.",
        )
    try:
        path = await asyncio.to_thread(_blocking_cds_request, variable, lat, lon, date_str)
        import xarray as xr
        ds = xr.open_dataset(path)
        value = float(ds[list(ds.data_vars)[0]].isel(time=0).sel(
            latitude=lat, longitude=lon, method="nearest").values)
        return SourceEnvelope(
            status="live", source="Copernicus CDS (ERA5 Reanalysis)", updated=date_str,
            resolution="0.25 deg", confidence="High",
            data={"variable": variable, "value": value, "lat": lat, "lon": lon},
        )
    except ImportError as exc:
        return SourceEnvelope(
            status="error", source="Copernicus CDS (ERA5)", updated=now_iso(),
            resolution="0.25 deg", confidence="Low", data=None,
            error=f"Missing optional dependency: {exc}. pip install cdsapi xarray netCDF4",
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error", source="Copernicus CDS (ERA5)", updated=now_iso(),
            resolution="0.25 deg", confidence="Low", data=None, error=str(exc),
        )


async def search_sentinel3(client: httpx.AsyncClient, lat: float, lon: float, top: int = 5) -> SourceEnvelope:
    """Public OData search -- no auth needed for search itself."""
    point_filter = (
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') "
        f"and contains(Name,'S3') and contains(Name,'SL_2_WST')"
    )
    params = {"$filter": point_filter, "$top": top, "$orderby": "ContentDate/Start desc"}
    try:
        payload = await fetch_json(client, CDSE_ODATA_URL, params=params)
        products = payload.get("value", [])
        return SourceEnvelope(
            status="live", source="Copernicus Data Space (Sentinel-3 SLSTR SST)", updated=now_iso(),
            resolution="1 km", confidence="High" if products else "Medium",
            data={"count": len(products),
                  "products": [{"name": p.get("Name"), "id": p.get("Id"),
                                 "sensed": p.get("ContentDate", {}).get("Start")} for p in products]},
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error", source="Copernicus Data Space (Sentinel-3 SLSTR SST)", updated=now_iso(),
            resolution="1 km", confidence="Low", data=None, error=str(exc),
        )


async def get_cdse_access_token(client: httpx.AsyncClient) -> str | None:
    """OAuth2 password grant against CDSE identity -- needed only to
    download (not search) Sentinel-3 products. Real, current endpoint.
    """
    username, password = os.getenv("CDSE_USERNAME"), os.getenv("CDSE_PASSWORD")
    if not (username and password):
        return None
    resp = await client.post(CDSE_IDENTITY_TOKEN_URL, data={
        "client_id": "cdse-public", "grant_type": "password",
        "username": username, "password": password,
    })
    resp.raise_for_status()
    return resp.json()["access_token"]
