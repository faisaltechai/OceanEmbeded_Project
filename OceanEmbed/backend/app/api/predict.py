import os
import pickle
import sys
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

ML_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "ml")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "demo")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class PredictRequest(BaseModel):
    region: str = Field(..., description="e.g. 'bay_of_bengal' or 'arabian_sea'")
    lat: float
    lon: float
    sst: float
    sss: float
    ssh: float = 0.0
    u_current: float = 0.0
    v_current: float = 0.0
    u_wind: float = 0.0
    v_wind: float = 0.0
    n_uncertainty_passes: int = 20


@lru_cache(maxsize=8)
def _load_model(region: str):
    path = os.path.join(DATA_DIR, f"model_{region}.pkl")
    if not os.path.exists(path):
        raise HTTPException(404, f"No trained model artifact for region '{region}' at {path}. Run ml/training/train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


@router.post("/predict")
def predict(req: PredictRequest):
    """
    Runs REAL inference through the pretrained physics-informed model
    (ml/models/model.py) for an arbitrary user-supplied surface observation,
    rather than only replaying precomputed demo grid cells. Requires
    data/demo/model_<region>.pkl to exist (produced by ml/training/train.py).
    """
    net = _load_model(req.region)
    x = np.array([[req.sst, req.sss, req.ssh, req.u_current, req.v_current,
                   req.u_wind, req.v_wind, req.lat, req.lon]])
    mean, std = net.predict_with_uncertainty(x, n_passes=req.n_uncertainty_passes)
    return {
        "region": req.region, "lat": req.lat, "lon": req.lon,
        "predicted_temperature_c": mean[0].tolist(),
        "uncertainty_c": std[0].tolist(),
        "units": "degC",
        "note": "Inference via the pretrained model artifact; no retraining occurs per request.",
    }
