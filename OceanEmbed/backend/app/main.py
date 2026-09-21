"""
OceanEmbed backend - FastAPI entrypoint.

NOTE ON THIS SANDBOX: the environment this was authored in has no network
access, so live calls to Copernicus/NASA/NOAA/Argovis, a real Redis
server, and a real Postgres instance were never exercised end-to-end here.
Every client in app/services/satellite/ is written against each provider's
real, current, documented API and is structured to run unmodified with
normal outbound internet + the credentials in .env.example -- see
docs/live-integrations.md for what to verify on first real deployment.

Everything that *can* run with zero external services (the precomputed
demo bundle, the in-memory cache fallback) still works exactly as before,
so this upgrade never breaks the "every button works" requirement even
before any API key is configured.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.data_store import DataStore
from app.scheduler.jobs import start_scheduler, stop_scheduler
from app.api import (
    health, regions, ocean, events, uncertainty, argo, validation, explainability, predict,
    overview, cyclones, timeline, report, active_learning,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = DataStore(settings.bundle_path)
    scheduler = None
    if settings.enable_scheduler:
        scheduler = start_scheduler()
    yield
    if scheduler is not None:
        stop_scheduler()


app = FastAPI(
    title="OceanEmbed API",
    description="Physics-informed, uncertainty-aware subsurface ocean temperature reconstruction "
                "(SIH26066), now with a live-satellite gateway layer.",
    version="0.2.0-prototype",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Original scientific-core routers (unchanged contract).
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(regions.router, prefix="/api", tags=["regions"])
app.include_router(ocean.router, prefix="/api/ocean", tags=["ocean"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(uncertainty.router, prefix="/api/uncertainty", tags=["uncertainty"])
app.include_router(argo.router, prefix="/api/argo", tags=["argo"])
app.include_router(validation.router, prefix="/api/validation", tags=["validation"])
app.include_router(explainability.router, prefix="/api/explainability", tags=["explainability"])
app.include_router(predict.router, prefix="/api", tags=["predict"])
app.include_router(active_learning.router, prefix="/api/active-learning", tags=["active-learning"])

# New live-gateway routers, matching the brief's flat /api/* endpoint list.
app.include_router(overview.router, prefix="/api", tags=["overview"])
app.include_router(cyclones.router, prefix="/api", tags=["cyclones"])
app.include_router(timeline.router, prefix="/api", tags=["timeline"])
app.include_router(report.router, prefix="/api", tags=["report"])
