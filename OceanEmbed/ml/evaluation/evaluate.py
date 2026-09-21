"""
OceanEmbed - evaluate.py
Computes real validation metrics (RMSE/Bias/Correlation) for the trained
model against (a) the held-out temporal split of the synthetic truth field,
(b) the synthetic ARGO profiles, and (c) a climatological baseline model,
so the Validation Dashboard and the "OceanEmbed vs baseline" comparison in
the SIH demo are backed by numbers this file actually calculated.
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.evaluation.services import validate_depthwise, climatology_baseline, rmse, bias, correlation


def evaluate_region(net, X_train, y_train, X_test, y_test, depths):
    pred_test = net.predict(X_test)
    baseline_profile = climatology_baseline(y_train)
    baseline_pred = np.tile(baseline_profile, (X_test.shape[0], 1))

    model_depthwise = validate_depthwise(pred_test, y_test, depths)
    baseline_depthwise = validate_depthwise(baseline_pred, y_test, depths)

    overall = {
        "model": {"rmse": rmse(pred_test, y_test), "bias": bias(pred_test, y_test), "correlation": correlation(pred_test, y_test)},
        "baseline_climatology": {"rmse": rmse(baseline_pred, y_test), "bias": bias(baseline_pred, y_test), "correlation": correlation(baseline_pred, y_test)},
    }
    return {
        "overall": overall,
        "depthwise_model": model_depthwise,
        "depthwise_baseline": baseline_depthwise,
        "n_test_samples": int(X_test.shape[0]),
    }


def evaluate_against_argo(net, argo_records, meta_lookup_fn, depths):
    """
    meta_lookup_fn(date, lat_idx, lon_idx) -> surface feature row X (1, n_features)
    Compares model prediction at each synthetic-Argo location/date to that
    Argo profile -- an independent comparison per section 22/46.
    """
    preds, obs = [], []
    for rec in argo_records:
        x_row = meta_lookup_fn(rec["date"], rec["lat_idx"], rec["lon_idx"])
        if x_row is None:
            continue
        pred = net.predict(x_row.reshape(1, -1))[0]
        preds.append(pred)
        obs.append(rec["profile"])
    if not preds:
        return None
    preds, obs = np.array(preds), np.array(obs)
    return {
        "n_argo_profiles": len(preds),
        "rmse": rmse(preds, obs), "bias": bias(preds, obs), "correlation": correlation(preds, obs),
        "depthwise": validate_depthwise(preds, obs, depths),
    }
