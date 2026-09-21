"""
OceanEmbed - Active Learning: per-cell metrics for one region/date
====================================================================
Builds the raw ingredients the acquisition function (acquisition.py)
scores, reusing the exact scientific definitions already computed
elsewhere in the pipeline -- MC-Dropout uncertainty (model.py), the
anomaly convention already used by ArgoGuidanceService, ExtremeEventDetector's
z-score/MLD-collapse ingredients, and services.thermocline_metrics --
rather than reimplementing any of them.

Every array returned is a plain (n_lat, n_lon) NumPy grid for the chosen
date so it can be normalized, weighted, and serialized to JSON.
"""
from __future__ import annotations
import numpy as np

from ml.evaluation.services import thermocline_metrics, ExtremeEventDetector


def grid_thermocline_gradient(temperature_field, depths, date_index):
    """
    temperature_field: (n_dates, n_lat, n_lon, n_depths)
    Returns an (n_lat, n_lon) grid of the steepest vertical gradient
    (services.thermocline_metrics's `max_gradient_c_per_m`, always <= 0 for
    a normal profile) for the requested date. Looped per cell -- these
    grids are small (tens to a few hundred cells) -- so this calls the
    SAME definition used everywhere else in the app instead of a second,
    divergent thermocline formula.
    """
    _, n_lat, n_lon, _ = temperature_field.shape
    grad = np.zeros((n_lat, n_lon))
    for i in range(n_lat):
        for j in range(n_lon):
            therm = thermocline_metrics(depths, temperature_field[date_index, i, j, :])
            grad[i, j] = therm["max_gradient_c_per_m"]
    return grad


def grid_distance_to_nearest_observation(lats, lons, argo_records):
    """
    (n_lat, n_lon) grid of the distance (in degrees -- the demo regions are
    small enough patches that a simple lat/lon Euclidean distance is an
    adequate proxy, not a claim of great-circle accuracy) from each cell to
    the NEAREST synthetic-Argo observation on record, across all demo
    dates. A cell with an existing observation sitting on it gets 0.
    """
    n_lat, n_lon = len(lats), len(lons)
    if not argo_records:
        return np.full((n_lat, n_lon), np.nan)
    LAT, LON = np.meshgrid(np.asarray(lats, dtype=float), np.asarray(lons, dtype=float), indexing="ij")
    dist = np.full((n_lat, n_lon), np.inf)
    for rec in argo_records:
        ol, on = lats[rec["lat_idx"]], lons[rec["lon_idx"]]
        d = np.sqrt((LAT - ol) ** 2 + (LON - on) ** 2)
        dist = np.minimum(dist, d)
    return dist


def compute_grid_metrics(temperature_field, uncertainty_field, mixed_layer_depth_field,
                          depths, lats, lons, dates, date_index, observation_density,
                          argo_records, event_detector: ExtremeEventDetector):
    """
    Returns a dict of (n_lat, n_lon) grids for ONE date, all built from
    quantities the pipeline already computes elsewhere -- nothing here is
    a new/independent scientific method:

      uncertainty_score            mean-over-depth MC-Dropout std (model.py)
      prediction_variance          uncertainty_score ** 2
      temperature_anomaly          (temp - per-cell climatology) / climatology
                                    std, averaged over depth (same convention
                                    ArgoGuidanceService already scores on)
      thermocline_gradient         services.thermocline_metrics, per cell
      event_risk                   ExtremeEventDetector's z-score / MLD-collapse
                                    ingredients, continuous 0-1
      observation_density          passed through unchanged
                                    (ml/inference/inference.py: observation_density_grid)
      distance_to_observation_deg  nearest synthetic-Argo hit, degrees
    """
    clim_mean = temperature_field.mean(axis=0)
    clim_std = temperature_field.std(axis=0) + 1e-6
    uncertainty_score = uncertainty_field[date_index].mean(axis=-1)
    temperature_anomaly = ((temperature_field[date_index] - clim_mean) / clim_std).mean(axis=-1)

    return {
        "uncertainty_score": uncertainty_score,
        "prediction_variance": uncertainty_score ** 2,
        "temperature_anomaly": temperature_anomaly,
        "thermocline_gradient": grid_thermocline_gradient(temperature_field, depths, date_index),
        "event_risk": event_detector.event_risk_grid(temperature_field, mixed_layer_depth_field, date_index),
        "observation_density": np.asarray(observation_density, dtype=float),
        "distance_to_observation_deg": grid_distance_to_nearest_observation(lats, lons, argo_records),
    }
