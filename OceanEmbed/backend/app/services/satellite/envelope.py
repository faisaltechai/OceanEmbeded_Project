"""
Shared envelope + HTTP helper for every live satellite/ocean data source.

Every satellite client in this package returns a `SourceEnvelope`. That
envelope is the ONE contract the rest of the app (gateway, API routes,
frontend) relies on -- it's what makes "no fake data" enforceable in code,
not just in a doc:

    status      "live" | "cached" | "demo" | "error"
    source      human-readable provider name, e.g. "NOAA CoastWatch ERDDAP"
    updated     ISO8601 UTC timestamp of when the underlying data was produced
                (NOT when we fetched it -- these can differ by hours/days)
    resolution  native spatial resolution string, e.g. "0.05 deg" / "25 km"
    confidence  "High" | "Medium" | "Low" -- provider/QC-flag derived, never
                invented (see each client's `_confidence_from_qc`)
    data        the actual payload (provider-specific shape)

Nothing downstream is allowed to synthesize a `SourceEnvelope` with
status="live" unless it actually came back from a network call that
succeeded. See docs/data-pipeline.md and docs/requirements.md for why this
matters to the SIH jury and to the "no mock API responses" requirement.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("oceanembed.satellite")

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)


@dataclass
class SourceEnvelope:
    status: str  # live | cached | demo | error
    source: str
    updated: str
    resolution: str
    confidence: str
    data: Any
    error: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    retries: int = 2,
    backoff_seconds: float = 1.5,
) -> dict:
    """GET a URL and parse JSON, with capped exponential-backoff retries.

    Raises the last exception on total failure so callers (the gateway) can
    decide the fallback path -- this file never swallows errors silently.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we log + rethrow
            last_exc = exc
            logger.warning("fetch_json failed (attempt %d/%d) for %s: %s", attempt + 1, retries + 1, url, exc)
            if attempt < retries:
                await asyncio.sleep(backoff_seconds * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


def confidence_from_qc(qc_flags: list[int] | None, good_flags: tuple[int, ...] = (0, 1)) -> str:
    """Map a list of provider QC flags to High/Medium/Low.

    This is the one place "confidence" is computed for live data so it can
    never be a hand-typed constant elsewhere. Providers differ in QC flag
    conventions (ARGO uses 1=good..4=bad, GHRSST uses 0=best..5=void); pass
    the provider's own "acceptable" flag set as `good_flags`.
    """
    if not qc_flags:
        return "Medium"
    good = sum(1 for f in qc_flags if f in good_flags)
    ratio = good / len(qc_flags)
    if ratio >= 0.9:
        return "High"
    if ratio >= 0.6:
        return "Medium"
    return "Low"
