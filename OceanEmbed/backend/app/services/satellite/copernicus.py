"""
Copernicus Marine Service integration.

Unlike ERDDAP/Argovis, Copernicus Marine Service (CMEMS) does not offer a
simple anonymous REST GET for subsetting -- the supported path is the
official `copernicusmarine` Python toolbox (pip install copernicusmarine),
which handles auth, catalogue resolution, and subsetting against their
S3-backed store. This is real, current (2024+) Copernicus Marine practice,
not a stand-in.

Auth: free account at https://data.marine.copernicus.eu -> credentials go
in COPERNICUS_MARINE_USERNAME / COPERNICUS_MARINE_PASSWORD env vars, which
the toolbox reads automatically.

Because the toolbox does synchronous, potentially slow I/O (it downloads a
NetCDF subset to disk, or a Zarr byte-range for `open_dataset`), we run it
in a worker thread via `asyncio.to_thread` so it doesn't block the FastAPI
event loop -- the correct pattern for wrapping a sync SDK in an async app.
"""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone

from .envelope import SourceEnvelope, now_iso

# Dataset IDs are the real, current Copernicus Marine product IDs as of the
# 2024 catalogue reorganization. Confirm at https://data.marine.copernicus.eu
# before relying on them long-term -- CMEMS periodically retires dataset
# versions (e.g. *_P1D-m suffixes bump).
SST_DATASET_ID = "cmems_obs-sst_glo_phy_nrt_l4_P1D-m"
SSH_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs_P1D"
CURRENTS_DATASET_ID = "cmems_mod_glo_phy_anfc_merged-uv_PT1H-i"


def _credentials_configured() -> bool:
    import os
    return bool(os.getenv("COPERNICUS_MARINE_USERNAME") and os.getenv("COPERNICUS_MARINE_PASSWORD"))


def _blocking_subset(dataset_id: str, variables: list[str], lat: float, lon: float,
                      buffer_deg: float = 0.5) -> dict:
    """Runs in a worker thread. Requires `pip install copernicusmarine`."""
    import copernicusmarine  # imported lazily: optional heavy dependency

    with tempfile.TemporaryDirectory() as tmp:
        result = copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=variables,
            minimum_longitude=lon - buffer_deg,
            maximum_longitude=lon + buffer_deg,
            minimum_latitude=lat - buffer_deg,
            maximum_latitude=lat + buffer_deg,
            output_directory=tmp,
            output_filename="subset.nc",
            force_download=True,
            disable_progress_bar=True,
        )
        import xarray as xr  # optional heavy dependency, only needed on this path
        ds = xr.open_dataset(f"{tmp}/subset.nc")
        latest = ds.isel(time=-1) if "time" in ds.dims else ds
        point = latest.sel(latitude=lat, longitude=lon, method="nearest")
        out = {var: float(point[var].values) for var in variables if var in point}
        time_val = None
        if "time" in ds.coords:
            time_val = str(ds.time.values[-1])
        return {"values": out, "native_time": time_val, "dataset_id": dataset_id}


async def fetch_point(dataset_id: str, variables: list[str], lat: float, lon: float,
                       provider_label: str, resolution: str) -> SourceEnvelope:
    if not _credentials_configured():
        return SourceEnvelope(
            status="error",
            source=provider_label,
            updated=now_iso(),
            resolution=resolution,
            confidence="Low",
            data=None,
            error="COPERNICUS_MARINE_USERNAME/PASSWORD not configured. "
                  "Register free at data.marine.copernicus.eu and set them in .env.",
        )
    try:
        result = await asyncio.to_thread(_blocking_subset, dataset_id, variables, lat, lon)
        return SourceEnvelope(
            status="live",
            source=provider_label,
            updated=result["native_time"] or now_iso(),
            resolution=resolution,
            confidence="High",
            data=result["values"] | {"dataset_id": dataset_id, "lat": lat, "lon": lon},
        )
    except ImportError as exc:
        return SourceEnvelope(
            status="error", source=provider_label, updated=now_iso(), resolution=resolution,
            confidence="Low", data=None,
            error=f"Missing optional dependency for Copernicus Marine: {exc}. "
                  f"pip install copernicusmarine xarray netCDF4",
        )
    except Exception as exc:  # noqa: BLE001
        return SourceEnvelope(
            status="error", source=provider_label, updated=now_iso(), resolution=resolution,
            confidence="Low", data=None, error=str(exc),
        )


async def fetch_sst(lat: float, lon: float) -> SourceEnvelope:
    return await fetch_point(SST_DATASET_ID, ["analysed_sst"], lat, lon,
                              "Copernicus Marine Service (SST L4)", "0.05 deg")


async def fetch_ssh(lat: float, lon: float) -> SourceEnvelope:
    return await fetch_point(SSH_DATASET_ID, ["sla", "adt"], lat, lon,
                              "Copernicus Marine Service (SSH/SLA DUACS)", "0.25 deg")


async def fetch_currents(lat: float, lon: float) -> SourceEnvelope:
    return await fetch_point(CURRENTS_DATASET_ID, ["uo", "vo"], lat, lon,
                              "Copernicus Marine Service (Global Analysis Currents)", "1/12 deg")
