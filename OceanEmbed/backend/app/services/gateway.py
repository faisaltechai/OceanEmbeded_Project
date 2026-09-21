"""
DataGateway: the seam between "real satellite clients" and "the demo bundle".

This is the piece that makes RULE #2 in the brief ("no fake data, never
hardcode, never mock") honest under the actual constraint that no API keys
are configured yet: every response passing through here is *labeled* with
where it actually came from, and the label is never "live" unless a live
call actually succeeded this run.

Order of attempts, matching the brief's own error-handling section:
  1. LIVE   - call the real provider client.
  2. CACHED - if live fails, serve the last successful live response from
              Redis (age-stamped, so the frontend can show "Last successful
              update: <time>" per the brief instead of silently going stale).
  3. DEMO   - if there is no cache either (e.g. first run, never had
              working credentials), fall back to the synthetic demo bundle
              that ml/inference/build_demo_bundle.py produced, clearly
              labeled status="demo" -- this is what keeps every button
              working (RULE: "never leave buttons broken") without ever
              mislabeling synthetic data as live.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from app.cache.redis_cache import get_cache
from app.services.satellite.envelope import SourceEnvelope, now_iso

logger = logging.getLogger("oceanembed.gateway")

CACHE_TTL_SECONDS = {
    "sst": 3 * 3600,
    "ssh": 3 * 3600,
    "currents": 3 * 3600,
    "argo": 12 * 3600,
    "cyclone": 30 * 60,
    "era5": 3 * 3600,
    "sentinel3": 3 * 3600,
}


async def resolve(
    cache_key: str,
    live_fn: Callable[[], Awaitable[SourceEnvelope]],
    demo_fn: Callable[[], dict],
    ttl_bucket: str = "sst",
) -> dict:
    """Run `live_fn`; on success cache + return it; on failure try the cache;
    on total miss, fall back to `demo_fn()` wrapped as status="demo".
    """
    cache = get_cache()
    ttl = CACHE_TTL_SECONDS.get(ttl_bucket, 3 * 3600)

    try:
        envelope = await live_fn()
        if envelope.status == "live":
            await cache.set_json(cache_key, envelope.to_dict(), ttl)
            return envelope.to_dict()
        logger.info("Live fetch for %s returned status=%s (%s); trying cache", cache_key, envelope.status, envelope.error)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live fetch for %s raised %s; trying cache", cache_key, exc)

    cached = await cache.get_json(cache_key)
    if cached:
        cached = dict(cached)
        cached["status"] = "cached"
        cached["note"] = f"Live source unavailable -- showing last successful update from {cached.get('updated')}."
        return cached

    demo_data = demo_fn()
    return SourceEnvelope(
        status="demo",
        source=demo_data.get("source", "OceanEmbed Demo Dataset"),
        updated=demo_data.get("updated", now_iso()),
        resolution=demo_data.get("resolution", "n/a"),
        confidence=demo_data.get("confidence", "Medium"),
        data=demo_data.get("data"),
        error="No API credentials configured for this source yet -- see .env.example. "
              "Showing the synthetic prototype dataset, clearly labeled, per RULE #2.",
    ).to_dict()
