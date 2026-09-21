"""
Generic ERDDAP griddap client.

ERDDAP (https://coastwatch.pfeg.noaa.gov/erddap/) is the real, public,
key-free REST interface NOAA CoastWatch runs in front of GHRSST, OSCAR
currents, chlorophyll, and dozens of other gridded ocean products. Anyone
can query it today with no account -- which makes it the one integration
in this file we can be most confident will "just work" the moment this
runs somewhere with outbound internet.

Query shape (real, documented ERDDAP griddap REST syntax):

    GET {base_url}/griddap/{dataset_id}.json?
        {variable}[(time)][(lat_lo):(lat_hi)][(lon_lo):(lon_hi)]

We keep dataset_id / variable name / base_url all configurable via env
vars / constructor args because ERDDAP dataset IDs are renamed/retired by
NOAA occasionally -- hardcoding one is exactly the kind of brittle fake-
looking integration this project is trying to avoid. Verify the current
dataset_id at {base_url}/info/index.json before deploying.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from .envelope import SourceEnvelope, fetch_json, now_iso


class ERDDAPClient:
    def __init__(
        self,
        base_url: str,
        dataset_id: str,
        variable: str,
        provider_label: str,
        native_resolution: str,
    ):
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.variable = variable
        self.provider_label = provider_label
        self.native_resolution = native_resolution

    async def fetch_point(
        self, client: httpx.AsyncClient, lat: float, lon: float, when: datetime | None = None
    ) -> SourceEnvelope:
        when = when or datetime.now(timezone.utc)
        time_str = when.strftime("%Y-%m-%dT00:00:00Z")
        url = f"{self.base_url}/griddap/{self.dataset_id}.json"
        # ERDDAP griddap constraint syntax: var[(time)][(lat)][(lon)]
        query = f"{self.variable}[({time_str})][({lat})][({lon})]"
        try:
            # httpx's params= doesn't encode ERDDAP's bracket/paren constraint
            # syntax the way ERDDAP expects, so we build the full URL ourselves
            # (this is the standard approach every ERDDAP client, including
            # NOAA's own `erddapy`, uses).
            full_url = f"{url}?{query}"
            payload = await fetch_json(client, full_url)
            table = payload["table"]
            col_names = table["columnNames"]
            row = table["rows"][0]
            value = dict(zip(col_names, row)).get(self.variable)
            return SourceEnvelope(
                status="live",
                source=self.provider_label,
                updated=time_str,
                resolution=self.native_resolution,
                confidence="High" if value is not None else "Low",
                data={"lat": lat, "lon": lon, "variable": self.variable, "value": value, "dataset_id": self.dataset_id},
            )
        except Exception as exc:  # noqa: BLE001
            return SourceEnvelope(
                status="error",
                source=self.provider_label,
                updated=now_iso(),
                resolution=self.native_resolution,
                confidence="Low",
                data=None,
                error=str(exc),
            )

    async def fetch_grid(
        self,
        client: httpx.AsyncClient,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        when: datetime | None = None,
    ) -> SourceEnvelope:
        when = when or datetime.now(timezone.utc)
        time_str = when.strftime("%Y-%m-%dT00:00:00Z")
        query = (
            f"{self.variable}[({time_str})]"
            f"[({lat_range[0]}):({lat_range[1]})]"
            f"[({lon_range[0]}):({lon_range[1]})]"
        )
        full_url = f"{self.base_url}/griddap/{self.dataset_id}.json?{query}"
        try:
            payload = await fetch_json(client, full_url)
            return SourceEnvelope(
                status="live",
                source=self.provider_label,
                updated=time_str,
                resolution=self.native_resolution,
                confidence="High",
                data=payload["table"],
            )
        except Exception as exc:  # noqa: BLE001
            return SourceEnvelope(
                status="error",
                source=self.provider_label,
                updated=now_iso(),
                resolution=self.native_resolution,
                confidence="Low",
                data=None,
                error=str(exc),
            )


def noaa_coastwatch_sst_client() -> ERDDAPClient:
    """GHRSST L4 SST, served by NOAA CoastWatch's public ERDDAP.

    Verify current dataset_id at
    https://coastwatch.pfeg.noaa.gov/erddap/griddap/index.html?page=1&itemsPerPage=1000
    (search 'GHRSST' or 'jplMURSST') -- IDs occasionally change when NOAA
    reprocesses a product version.
    """
    return ERDDAPClient(
        base_url="https://coastwatch.pfeg.noaa.gov/erddap",
        dataset_id="jplMURSST41",
        variable="analysed_sst",
        provider_label="NOAA CoastWatch ERDDAP (GHRSST MUR SST)",
        native_resolution="0.01 deg (~1km)",
    )


def oscar_currents_client() -> ERDDAPClient:
    """OSCAR near-surface ocean currents, also mirrored on public ERDDAP.

    Verify current dataset_id at the same ERDDAP /info/index.json listing.
    """
    return ERDDAPClient(
        base_url="https://coastwatch.pfeg.noaa.gov/erddap",
        dataset_id="jplOscar_U0.33deg",
        variable="u",
        provider_label="PO.DAAC OSCAR Ocean Surface Currents",
        native_resolution="0.33 deg",
    )
