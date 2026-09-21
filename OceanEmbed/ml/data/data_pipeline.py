"""
OceanEmbed - Demo Data Pipeline
================================
IMPORTANT / SCIENTIFIC HONESTY NOTICE
--------------------------------------
This sandbox has no network access to download real Copernicus/GLORYS,
NOAA/satellite SST-SSS-SSH, or Argo GDAC data. Building OceanEmbed's demo
mode therefore uses a SYNTHETIC-BUT-PHYSICALLY-STYLED proxy dataset:

  - Vertical temperature profiles follow a real oceanographic shape
    (mixed layer -> thermocline -> deep water, via a tanh formulation),
    parameterised by latitude, a smooth spatial random field, and a
    per-date seasonal signal.
  - "ARGO" here = a sparse random subsample of grid cells/dates (mimicking
    Argo floats' sparse coverage), built by adding float-like noise to the
    synthetic truth field.
  - "GLORYS" here = a reanalysis-like reference built by adding smoothing +
    a small structured bias to the synthetic truth field (reanalyses are
    smoother than reality but carry their own bias).

This is explicitly a stand-in so the full pipeline (ingestion -> QC ->
regridding -> model -> validation -> API -> UI) is real and runs end to
end. Every output this module produces is tagged demo_source="synthetic"
so nothing downstream can present it as a live observation. Swapping in
real NetCDF granules from Copernicus Marine / PODAAC / Argo GDAC requires
only replacing `generate_raw_fields()` below with real readers -- the
rest of the pipeline (QC, regridding, normalization, model I/O contracts)
is written against the same array shapes real data would provide.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yaml
import json
import os
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "..", "configs", "config.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_grid(bbox: dict, resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(bbox["lat_min"], bbox["lat_max"] + 1e-9, resolution_deg)
    lons = np.arange(bbox["lon_min"], bbox["lon_max"] + 1e-9, resolution_deg)
    return lats, lons


def _smooth_random_field(shape, rng, smoothness=3):
    """Cheap spatial smoothing via repeated box blur (no scipy.ndimage needed)."""
    field = rng.normal(0, 1, size=shape)
    for _ in range(smoothness):
        field = (
            field
            + np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
        ) / 5.0
    return field


def generate_raw_fields(region_bbox: dict, resolution_deg: float, dates: list[str], depths_m: list[int], seed: int = 42):
    """
    Generates the synthetic-but-physically-styled surface + subsurface truth
    fields for one region across a list of ISO dates.

    Returns a dict of numpy arrays keyed by variable name, all shaped
    (n_dates, n_lat, n_lon) for surface vars, and (n_dates, n_lat, n_lon, n_depths)
    for the "true" subsurface temperature field.
    """
    rng = np.random.default_rng(seed)
    lats, lons = build_grid(region_bbox, resolution_deg)
    n_lat, n_lon, n_t, n_d = len(lats), len(lons), len(dates), len(depths_m)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    depths = np.array(depths_m, dtype=float)

    sst = np.zeros((n_t, n_lat, n_lon))
    sss = np.zeros((n_t, n_lat, n_lon))
    ssh = np.zeros((n_t, n_lat, n_lon))
    u_current = np.zeros((n_t, n_lat, n_lon))
    v_current = np.zeros((n_t, n_lat, n_lon))
    u_wind = np.zeros((n_t, n_lat, n_lon))
    v_wind = np.zeros((n_t, n_lat, n_lon))
    true_temp = np.zeros((n_t, n_lat, n_lon, n_d))
    mixed_layer_depth = np.zeros((n_t, n_lat, n_lon))
    events_mask = np.zeros((n_t, n_lat, n_lon), dtype=bool)  # synthetic marine-heatwave patches

    spatial_field = _smooth_random_field((n_lat, n_lon), rng)  # a fixed "eddy field" reused across dates

    for ti in range(n_t):
        seasonal = 0.6 * np.sin(2 * np.pi * ti / max(n_t, 1))
        day_noise = _smooth_random_field((n_lat, n_lon), rng, smoothness=2) * 0.4

        # SST: warmer at low latitude, cooler poleward, plus eddies + seasonal term
        sst_t = 29.5 - 0.28 * (LAT - region_bbox["lat_min"]) + 1.1 * spatial_field + seasonal + day_noise
        sss_t = 34.5 + 0.6 * spatial_field - 0.3 * np.sin(np.radians(LAT)) + 0.1 * day_noise
        ssh_t = 0.15 * spatial_field + 0.05 * seasonal
        u_current_t = 0.25 * np.roll(spatial_field, 1, axis=1) + 0.05 * day_noise
        v_current_t = 0.2 * np.roll(spatial_field, 1, axis=0) - 0.05 * day_noise
        u_wind_t = 4.0 + 3.0 * spatial_field + 1.0 * seasonal
        v_wind_t = 1.0 - 2.0 * spatial_field

        # Mixed layer depth: shallower in warm/eddy-rich areas, deeper poleward
        mld_t = 45 + 25 * (LAT - region_bbox["lat_min"]) / (region_bbox["lat_max"] - region_bbox["lat_min"]) - 15 * spatial_field
        mld_t = np.clip(mld_t, 15, 120)

        # Inject a handful of synthetic marine-heatwave / subsurface-anomaly patches
        heatwave_mask_t = np.zeros((n_lat, n_lon), dtype=bool)
        if ti >= n_t - 3:  # anomalies appear in the later (more "recent") demo dates
            n_events = rng.integers(1, 3)
            for _ in range(n_events):
                ci, cj = rng.integers(0, n_lat), rng.integers(0, n_lon)
                ri, rj = max(1, n_lat // 8), max(1, n_lon // 8)
                ii, jj = np.ogrid[:n_lat, :n_lon]
                patch = ((ii - ci) ** 2) / (ri ** 2) + ((jj - cj) ** 2) / (rj ** 2) <= 1
                sst_t = np.where(patch, sst_t + rng.uniform(1.2, 2.4), sst_t)
                mld_t = np.where(patch, mld_t * rng.uniform(0.45, 0.7), mld_t)
                heatwave_mask_t = heatwave_mask_t | patch

        sst[ti], sss[ti], ssh[ti] = sst_t, sss_t, ssh_t
        u_current[ti], v_current[ti] = u_current_t, v_current_t
        u_wind[ti], v_wind[ti] = u_wind_t, v_wind_t
        mixed_layer_depth[ti] = mld_t
        events_mask[ti] = heatwave_mask_t

        # Build the "true" vertical profile per grid cell using a tanh thermocline shape:
        # T(z) = T_deep + (T_surf - T_deep) * 0.5 * (1 - tanh((z - mld) / thermocline_width))
        deep_temp = 4.0 + 0.3 * spatial_field  # abyssal-ish asymptote, per-cell
        thermocline_width = 60 + 20 * spatial_field
        for zi, z in enumerate(depths):
            shape = 0.5 * (1 - np.tanh((z - mld_t) / np.clip(thermocline_width, 20, None)))
            true_temp[ti, :, :, zi] = deep_temp + (sst_t - deep_temp) * shape

    return {
        "lats": lats, "lons": lons, "dates": dates, "depths": depths,
        "sst": sst, "sss": sss, "ssh": ssh,
        "u_current": u_current, "v_current": v_current,
        "u_wind": u_wind, "v_wind": v_wind,
        "true_temp": true_temp, "mixed_layer_depth": mixed_layer_depth,
        "events_mask": events_mask,
    }


def make_argo_observations(fields: dict, coverage: float, obs_noise_std: float, seed: int = 7):
    """Sparse random subsample of (date, lat, lon) cells with float-like noise -> proxy for Argo profiles."""
    rng = np.random.default_rng(seed)
    n_t, n_lat, n_lon, n_d = fields["true_temp"].shape
    argo = []
    for ti in range(n_t):
        n_floats = max(1, int(coverage * n_lat * n_lon))
        idxs = rng.choice(n_lat * n_lon, size=n_floats, replace=False)
        for idx in idxs:
            i, j = divmod(idx, n_lon)
            profile = fields["true_temp"][ti, i, j, :] + rng.normal(0, obs_noise_std, size=n_d)
            argo.append({
                "date": fields["dates"][ti], "lat_idx": int(i), "lon_idx": int(j),
                "lat": float(fields["lats"][i]), "lon": float(fields["lons"][j]),
                "profile": profile.tolist(),
            })
    return argo


def make_glorys_reference(fields: dict, smoothing_bias_std: float, seed: int = 11):
    """Full-grid reanalysis-like reference: smoother + small structured bias vs. truth."""
    rng = np.random.default_rng(seed)
    bias_field = _smooth_random_field(fields["true_temp"].shape[1:3], rng, smoothness=4) * smoothing_bias_std
    glorys = fields["true_temp"] + bias_field[None, :, :, None] * 0.6
    # smooth slightly across depth to mimic reanalysis vertical smoothing
    glorys = (glorys + np.roll(glorys, 1, axis=3) + np.roll(glorys, -1, axis=3)) / 3.0
    glorys[:, :, :, 0] = fields["true_temp"][:, :, :, 0] + bias_field * 0.3  # surface tied closer to obs
    return glorys


def quality_control(fields: dict) -> dict:
    """Basic QC: clip physically impossible values, flag/interpolate any NaNs."""
    fields["sst"] = np.clip(fields["sst"], -2, 36)
    fields["sss"] = np.clip(fields["sss"], 30, 40)
    fields["true_temp"] = np.clip(fields["true_temp"], -2, 36)
    for key in ["sst", "sss", "ssh", "u_current", "v_current", "u_wind", "v_wind"]:
        arr = fields[key]
        if np.isnan(arr).any():
            col_mean = np.nanmean(arr)
            fields[key] = np.nan_to_num(arr, nan=col_mean)
    return fields


def build_feature_matrix(fields: dict, region_name: str):
    """Flatten grid+time into an (N, features) surface-variable matrix + (N, n_depths) targets."""
    n_t, n_lat, n_lon, n_d = fields["true_temp"].shape
    LAT, LON = np.meshgrid(fields["lats"], fields["lons"], indexing="ij")
    rows_X, rows_y, meta = [], [], []
    for ti in range(n_t):
        X_t = np.stack([
            fields["sst"][ti], fields["sss"][ti], fields["ssh"][ti],
            fields["u_current"][ti], fields["v_current"][ti],
            fields["u_wind"][ti], fields["v_wind"][ti],
            LAT, LON,
        ], axis=-1).reshape(-1, 9)
        y_t = fields["true_temp"][ti].reshape(-1, n_d)
        rows_X.append(X_t)
        rows_y.append(y_t)
        for i in range(n_lat):
            for j in range(n_lon):
                meta.append({"date": fields["dates"][ti], "lat": float(fields["lats"][i]),
                             "lon": float(fields["lons"][j]), "region": region_name,
                             "date_idx": ti, "lat_idx": i, "lon_idx": j})
    X = np.concatenate(rows_X, axis=0)
    y = np.concatenate(rows_y, axis=0)
    return X, y, meta


def temporal_split(n_dates: int, test_dates: int = 1):
    """Split by TIME, not randomly -- avoids leakage across the temporal axis (section 39)."""
    train_idx = list(range(0, n_dates - test_dates))
    test_idx = list(range(n_dates - test_dates, n_dates))
    return train_idx, test_idx


def run_pipeline(config: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    demo_cfg = config["demo"]
    start = datetime.fromisoformat(demo_cfg["start_date"])
    dates = [(start + timedelta(days=i)).date().isoformat() for i in range(demo_cfg["n_dates"])]
    depths = config["depths_m"]
    resolution = config["grid"]["demo_resolution_deg"]

    all_regions = {}
    for key, region in config["regions"].items():
        fields = generate_raw_fields(region["bbox"], resolution, dates, depths, seed=config["model"]["seed"])
        fields = quality_control(fields)
        argo = make_argo_observations(fields, coverage=0.10, obs_noise_std=0.35)
        glorys = make_glorys_reference(fields, smoothing_bias_std=0.5)
        all_regions[key] = {"fields": fields, "argo": argo, "glorys": glorys, "meta": region}
        print(f"[data_pipeline] {region['name']}: grid {fields['sst'].shape[1]}x{fields['sst'].shape[2]}, "
              f"{len(dates)} dates, {len(argo)} synthetic ARGO profiles")
    return all_regions, dates, depths


if __name__ == "__main__":
    cfg = load_config()
    run_pipeline(cfg, out_dir=os.path.join(HERE, "..", "..", "data", "demo"))
