"""
Async SQLAlchemy engine + session factory.

Postgres/PostGIS is OPTIONAL for local/demo mode -- app.main only wires this
up when OCEANEMBED_ENV=production or DATABASE_URL is explicitly set, so the
demo bundle path keeps working with zero infra, per the original project's
design (see docs/architecture.md).
"""
from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://oceanembed:changeme@localhost:5432/oceanembed",
)

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)
    return _session_factory


async def get_db_session() -> AsyncSession:
    """FastAPI dependency: `session: AsyncSession = Depends(get_db_session)`."""
    factory = get_session_factory()
    async with factory() as session:
        yield session
