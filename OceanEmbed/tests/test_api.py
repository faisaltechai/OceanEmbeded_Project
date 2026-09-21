"""
OceanEmbed tests - Backend API.
Requires `pip install -r backend/requirements.txt httpx` (FastAPI's TestClient
needs httpx). NOT RUNNABLE in the sandbox this project was authored in (no
network access to install FastAPI) -- written correctly and ready to run
wherever those packages are available:

    cd backend && python -m pytest ../tests/test_api.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "bay_of_bengal" in body["regions_loaded"]


def test_regions_listed():
    r = client.get("/api/regions")
    assert r.status_code == 200
    keys = [x["key"] for x in r.json()["regions"]]
    assert "bay_of_bengal" in keys and "arabian_sea" in keys


def test_map_endpoint_surface():
    r = client.get("/api/ocean/map", params={"region": "bay_of_bengal", "depth_m": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["units"] == "degC"
    assert len(body["values"]) == len(body["lats"])


def test_map_endpoint_unsupported_depth():
    r = client.get("/api/ocean/map", params={"region": "bay_of_bengal", "depth_m": 37})
    assert r.status_code == 400


def test_profile_endpoint():
    r = client.get("/api/ocean/profile", params={"region": "bay_of_bengal", "lat": 10.0, "lon": 88.0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predicted_temperature_c"]) == len(body["uncertainty_c"])
    assert "thermocline_depth_m" in body["thermocline"]


def test_validation_endpoint_has_baseline_comparison():
    r = client.get("/api/validation", params={"region": "arabian_sea"})
    body = r.json()
    assert "model" in body["overall"] and "baseline_climatology" in body["overall"]


def test_events_endpoint():
    r = client.get("/api/events", params={"region": "bay_of_bengal"})
    assert r.status_code == 200
    assert r.json()["count"] == len(r.json()["events"])


def test_argo_guidance_endpoint():
    r = client.get("/api/argo/guidance", params={"region": "arabian_sea"})
    assert r.status_code == 200
    assert len(r.json()["recommended_observation_zones"]) > 0


def test_explainability_sums_to_one():
    r = client.get("/api/explainability", params={"region": "bay_of_bengal"})
    contributions = r.json()["contributions"]
    assert abs(sum(contributions.values()) - 1.0) < 1e-3
