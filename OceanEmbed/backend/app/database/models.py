"""
SQLAlchemy ORM models for the app-state / caching layer (Postgres + PostGIS).

Scope note: the scientific core of OceanEmbed (map grids, profiles,
uncertainty) is served from the in-memory bundle / live satellite gateway,
NOT from Postgres -- Postgres here is for exactly what the brief's table
list implies: users, saved locations, a durable satellite response cache
(a persistent complement to Redis's TTL cache), model predictions worth
keeping, Argo profile metadata, cyclone events, and generated report
records. Spatial columns use PostGIS geometry so region/point queries can
use real GIST indexes instead of naive lat/lon range scans.
"""
from __future__ import annotations

import datetime as dt

from geoalchemy2 import Geometry
from sqlalchemy import (
    JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    locations: Mapped[list["Location"]] = relationship(back_populates="user")
    reports: Mapped[list["GeneratedReport"]] = relationship(back_populates="user")


class Location(Base):
    """A saved point-of-interest (e.g. 'watching this cyclone', 'my station')."""
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    label: Mapped[str] = mapped_column(String(255))
    region_key: Mapped[str] = mapped_column(String(64), index=True)
    geom: Mapped[str] = mapped_column(Geometry(geometry_type="POINT", srid=4326))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="locations")


class SatelliteCache(Base):
    """Durable (non-TTL-evicted) record of what a live source last returned,
    for audit trail + the 'Last Successful Update' UI requirement -- Redis
    remains the hot path; this is the paper trail.
    """
    __tablename__ = "satellite_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(128), index=True)
    cache_key: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(16))  # live | cached | demo | error
    resolution: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict] = mapped_column(JSON)
    fetched_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    provider_updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Prediction(Base):
    """A model output worth persisting (e.g. requested via /api/profile at a
    specific lat/lon/date), distinct from the precomputed demo bundle.
    """
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_key: Mapped[str] = mapped_column(String(64), index=True)
    geom: Mapped[str] = mapped_column(Geometry(geometry_type="POINT", srid=4326))
    valid_date: Mapped[dt.date] = mapped_column(DateTime(timezone=True))
    depths_m: Mapped[list] = mapped_column(JSON)
    temperature_c: Mapped[list] = mapped_column(JSON)
    uncertainty_c: Mapped[list] = mapped_column(JSON)
    confidence: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ArgoProfile(Base):
    __tablename__ = "argo_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform_number: Mapped[str] = mapped_column(String(32), index=True)
    cycle_number: Mapped[int] = mapped_column(Integer)
    geom: Mapped[str] = mapped_column(Geometry(geometry_type="POINT", srid=4326))
    profile_date: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    temperature_profile: Mapped[list] = mapped_column(JSON)
    salinity_profile: Mapped[list] = mapped_column(JSON)
    pressure_levels_db: Mapped[list] = mapped_column(JSON)
    position_qc: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(64), default="Argo GDAC (via Argovis)")


class CycloneEvent(Base):
    __tablename__ = "cyclone_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    storm_name: Mapped[str] = mapped_column(String(128))
    basin: Mapped[str] = mapped_column(String(64), index=True)
    category: Mapped[str] = mapped_column(String(32))
    track: Mapped[list] = mapped_column(JSON)  # list of {lat, lon, time, wind_kt, pressure_hpa}
    warm_core_regions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    subsurface_heat_reservoir_j_m2: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(64))
    last_updated: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class GeneratedReport(Base):
    __tablename__ = "generated_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    region_key: Mapped[str] = mapped_column(String(64))
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_path: Mapped[str] = mapped_column(String(512))
    sources_used: Mapped[list] = mapped_column(JSON)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="reports")
