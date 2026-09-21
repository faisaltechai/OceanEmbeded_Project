"""
OceanEmbed - inference.py
Runs the trained model over the FULL grid (not just held-out rows) to
produce the daily 3D subsurface temperature field + uncertainty that the
map/profile/API layers need. Pretrained-model-then-cached-inference
pattern per section 38 (no retraining on every dashboard open).
"""
from __future__ import annotations
import numpy as np


def predict_full_grid(net, fields, n_passes=20):
    """
    fields: dict from data_pipeline.generate_raw_fields (already QC'd)
    Returns:
      temperature: (n_dates, n_lat, n_lon, n_depths)
      uncertainty: (n_dates, n_lat, n_lon, n_depths)
    """
    n_t, n_lat, n_lon = fields["sst"].shape
    n_d = len(fields["depths"])
    LAT, LON = np.meshgrid(fields["lats"], fields["lons"], indexing="ij")

    temperature = np.zeros((n_t, n_lat, n_lon, n_d))
    uncertainty = np.zeros((n_t, n_lat, n_lon, n_d))

    for ti in range(n_t):
        X_t = np.stack([
            fields["sst"][ti], fields["sss"][ti], fields["ssh"][ti],
            fields["u_current"][ti], fields["v_current"][ti],
            fields["u_wind"][ti], fields["v_wind"][ti], LAT, LON,
        ], axis=-1).reshape(-1, 9)
        mean, std = net.predict_with_uncertainty(X_t, n_passes=n_passes)
        temperature[ti] = mean.reshape(n_lat, n_lon, n_d)
        uncertainty[ti] = std.reshape(n_lat, n_lon, n_d)

    return temperature, uncertainty


def confidence_category(std_value, low_thresh=0.35, high_thresh=0.8):
    if std_value <= low_thresh:
        return "High"
    elif std_value <= high_thresh:
        return "Medium"
    return "Low"


def observation_density_grid(argo_records, n_lat, n_lon):
    """Counts synthetic-Argo hits per grid cell across all dates -> proxy for real obs density."""
    density = np.zeros((n_lat, n_lon))
    for rec in argo_records:
        density[rec["lat_idx"], rec["lon_idx"]] += 1
    return density
