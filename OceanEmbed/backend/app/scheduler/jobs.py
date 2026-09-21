"""
Background refresh scheduler.

Implements the brief's cadence exactly:
  - satellite data (SST/SSH/currents): every 3 hours
  - cyclone data: every 30 minutes
  - Argo metadata: every 12 hours
  - cache cleanup: daily

Each job just calls the same gateway.resolve() path the API routes use, so
a successful scheduled run pre-warms the Redis/in-memory cache the request
path will hit -- if a job fails, `gateway.resolve` already handles the
cache/demo fallback on the next real request, so a missed scheduler tick
degrades gracefully rather than breaking anything (per the brief's error-
handling rules).

Uses real APScheduler (AsyncIOScheduler) wired into FastAPI's lifespan.
"""
from __future__ import annotations

import logging

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.regions_global import REGIONS
from app.services import gateway
from app.services.satellite import argo as argo_client
from app.services.satellite.erddap import noaa_coastwatch_sst_client
from app.services.satellite.nhc_cyclones import fetch_active_storms

logger = logging.getLogger("oceanembed.scheduler")

_scheduler: AsyncIOScheduler | None = None


async def _refresh_satellite_data():
    sst_client = noaa_coastwatch_sst_client()
    async with httpx.AsyncClient() as client:
        for key, meta in REGIONS.items():
            south, west, north, east = meta["bbox"]
            lat, lon = (south + north) / 2, (west + east) / 2
            await gateway.resolve(
                cache_key=f"overview:sst:{key}",
                live_fn=lambda c=client, la=lat, lo=lon: sst_client.fetch_point(c, la, lo),
                demo_fn=lambda: {"source": "scheduler-skip", "data": None},
                ttl_bucket="sst",
            )
    logger.info("Scheduled satellite refresh completed for %d regions", len(REGIONS))


async def _refresh_cyclone_data():
    async with httpx.AsyncClient() as client:
        await gateway.resolve(
            cache_key="cyclones:active",
            live_fn=lambda: fetch_active_storms(client),
            demo_fn=lambda: {"source": "scheduler-skip", "data": None},
            ttl_bucket="cyclone",
        )
    logger.info("Scheduled cyclone refresh completed")


async def _refresh_argo_metadata():
    async with httpx.AsyncClient() as client:
        for key, meta in REGIONS.items():
            south, west, north, east = meta["bbox"]
            lat, lon = (south + north) / 2, (west + east) / 2
            await gateway.resolve(
                cache_key=f"overview:argo:{key}",
                live_fn=lambda c=client, la=lat, lo=lon: argo_client.fetch_nearby_floats(c, la, lo, radius_km=500),
                demo_fn=lambda: {"source": "scheduler-skip", "data": None},
                ttl_bucket="argo",
            )
    logger.info("Scheduled Argo metadata refresh completed for %d regions", len(REGIONS))


async def _clean_cache():
    # The Redis TTLs already expire entries; for the in-memory fallback the
    # dict is pruned lazily on access. This job is the documented seam for
    # a more aggressive daily VACUUM/consistency pass once Postgres's
    # satellite_cache audit table is in active use.
    logger.info("Daily cache cleanup tick (TTL-based eviction; no-op for in-memory backend)")


def start_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    sched = AsyncIOScheduler()
    sched.add_job(_refresh_satellite_data, IntervalTrigger(hours=3), id="refresh_satellite")
    sched.add_job(_refresh_cyclone_data, IntervalTrigger(minutes=30), id="refresh_cyclones")
    sched.add_job(_refresh_argo_metadata, IntervalTrigger(hours=12), id="refresh_argo")
    sched.add_job(_clean_cache, IntervalTrigger(hours=24), id="clean_cache")
    sched.start()
    _scheduler = sched
    logger.info("OceanEmbed background scheduler started (satellite=3h, cyclones=30m, argo=12h, cache_clean=24h)")
    return sched


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
