"""
OceanEmbed - build_demo_bundle.py
=================================
The single script that runs the ENTIRE prototype pipeline end-to-end and
writes real, computed JSON outputs (never fabricated numbers) that the
backend API and the demo frontend both read:

  data/demo/backend_bundle.json   <- everything the FastAPI endpoints need
  data/demo/frontend_demo.json    <- a compact slice for the in-chat demo UI

Run: python ml/inference/build_demo_bundle.py
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.data.data_pipeline import load_config, run_pipeline, build_feature_matrix, temporal_split
from ml.models.model import PhysicsInformedOceanNet
from ml.evaluation.evaluate import evaluate_region, evaluate_against_argo
from ml.evaluation.services import (
    ExtremeEventDetector, ArgoGuidanceService, thermocline_metrics, ocean_heat_content,
)
from ml.inference.inference import predict_full_grid, confidence_category, observation_density_grid
from ml.active_learning import build_active_learning_bundle

FEATURE_NAMES = ["sst", "sss", "ssh", "u_current", "v_current", "u_wind", "v_wind", "lat", "lon"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "data", "demo")


def process_region(region_key, region_data, dates, depths, config):
    fields = region_data["fields"]
    X, y, meta = build_feature_matrix(fields, region_key)
    n_dates = len(dates)
    train_date_idx, test_date_idx = temporal_split(n_dates, test_dates=1)
    train_mask = np.array([m["date_idx"] in train_date_idx for m in meta])
    test_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model_cfg = config["model"]
    net = PhysicsInformedOceanNet(
        n_features=X.shape[1], n_depths=y.shape[1],
        hidden_dim=model_cfg["hidden_dim"], embedding_dim=model_cfg["embedding_dim"],
        dropout_rate=config["uncertainty"]["dropout_rate"], seed=model_cfg["seed"],
    )
    t0 = time.time()
    net.train(X_train, y_train, sst_col_idx=0, physics_cfg=config["physics_loss"],
              epochs=model_cfg["epochs"], lr=model_cfg["learning_rate"],
              batch_size=model_cfg["batch_size"], verbose=False)
    print(f"[{region_key}] trained in {time.time()-t0:.1f}s")

    metrics = evaluate_region(net, X_train, y_train, X_test, y_test, depths)

    # Meta lookup for ARGO comparison
    meta_by_key = {(m["date"], m["lat_idx"], m["lon_idx"]): i for i, m in enumerate(meta)}
    def lookup(date, lat_idx, lon_idx):
        key = (date, lat_idx, lon_idx)
        if key not in meta_by_key:
            return None
        return X[meta_by_key[key]]
    argo_metrics = evaluate_against_argo(net, region_data["argo"], lookup, depths)

    # Full-grid inference (cached "pretrained model -> inference -> cached output" pattern)
    temperature, uncertainty = predict_full_grid(net, fields, n_passes=config["uncertainty"]["n_forward_passes"])

    # Explainability (feature ablation) on a sample of grid cells from the most recent date
    sample_idx = np.random.default_rng(0).choice(X.shape[0], size=min(300, X.shape[0]), replace=False)
    explainability = net.feature_ablation_importance(X[sample_idx], FEATURE_NAMES)

    # Extreme events
    detector = ExtremeEventDetector(std_threshold=config["events"]["subsurface_heat_anomaly_std"])
    events = detector.detect(temperature, fields["mixed_layer_depth"], depths, fields["lats"], fields["lons"], dates)

    # Smart ARGO guidance (scored on the most recent date)
    n_lat, n_lon = fields["sst"].shape[1], fields["sst"].shape[2]
    density = observation_density_grid(region_data["argo"], n_lat, n_lon)
    last_t = -1
    clim_mean = temperature.mean(axis=0)
    clim_std = temperature.std(axis=0) + 1e-6
    anomaly_last = ((temperature[last_t] - clim_mean) / clim_std).mean(axis=-1)  # avg anomaly across depths
    uncertainty_last = uncertainty[last_t].mean(axis=-1)  # avg uncertainty across depths

    guidance = ArgoGuidanceService()
    scores, components = guidance.score(uncertainty_last, density, anomaly_last)
    top_n = 8
    flat_idx = np.argsort(scores.ravel())[::-1][:top_n]
    guidance_points = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, scores.shape)
        reason = guidance.explain(components["uncertainty_component"][i, j],
                                   components["sparsity_component"][i, j],
                                   components["anomaly_component"][i, j])
        guidance_points.append({
            "lat": float(fields["lats"][i]), "lon": float(fields["lons"][j]),
            "priority_score": float(scores[i, j]), "reason": reason,
        })

    # Active Learning & Adaptive Sampling (extends ArgoGuidanceService above with
    # thermocline-gradient / event-risk components, spatial diversity, and
    # platform + depth + re-sampling-cadence recommendations -- see
    # ml/active_learning/ and docs/active-learning.md).
    active_learning = build_active_learning_bundle(
        temperature, uncertainty, fields["mixed_layer_depth"],
        depths, fields["lats"], fields["lons"], dates, last_t, density,
        region_data["argo"], config.get("active_learning", {}),
    )

    # Thermocline + heat content for a handful of representative cells (last date)
    # Prefer sample cells that actually have a synthetic-Argo hit on the last date so the
    # demo's "OceanEmbed vs ARGO vs GLORYS" comparison has real Argo overlays to show.
    argo_last_date_cells = [(a["lat_idx"], a["lon_idx"]) for a in region_data["argo"] if a["date"] == dates[last_t]]
    default_cells = [(n_lat // 4, n_lon // 4), (n_lat // 2, n_lon // 2), (3 * n_lat // 4, 3 * n_lon // 4)]
    sample_cells = (argo_last_date_cells[:2] + default_cells)[:3]
    argo_by_cell_date = {(a["lat_idx"], a["lon_idx"], a["date"]): a for a in region_data["argo"]}
    glorys = region_data["glorys"]
    profile_samples = []
    for (i, j) in sample_cells:
        profile = temperature[last_t, i, j, :]
        unc = uncertainty[last_t, i, j, :]
        therm = thermocline_metrics(depths, profile)
        ohc = ocean_heat_content(depths, profile, depth_range=(0, 300))
        glorys_profile = glorys[last_t, i, j, :]
        argo_rec = argo_by_cell_date.get((i, j, dates[last_t]))
        # fall back to nearest earlier date with an Argo hit at this cell, if any
        if argo_rec is None:
            for d in reversed(dates[:-1]):
                argo_rec = argo_by_cell_date.get((i, j, d))
                if argo_rec is not None:
                    break
        profile_samples.append({
            "lat": float(fields["lats"][i]), "lon": float(fields["lons"][j]),
            "date": dates[last_t],
            "predicted_temperature_c": profile.tolist(),
            "uncertainty_c": unc.tolist(),
            "confidence": [confidence_category(s) for s in unc],
            "thermocline": therm, "ocean_heat_content": ohc,
            "glorys_reference_c": glorys_profile.round(2).tolist(),
            "argo_observed_c": (argo_rec["profile"] if argo_rec else None),
            "argo_observed_date": (argo_rec["date"] if argo_rec else None),
        })

    return {
        "region": region_key,
        "region_name": region_data["meta"]["name"],
        "bbox": region_data["meta"]["bbox"],
        "lats": fields["lats"].tolist(), "lons": fields["lons"].tolist(),
        "dates": dates, "depths_m": [float(d) for d in depths],
        "metrics": metrics,
        "argo_validation": argo_metrics,
        "explainability": explainability,
        "events": events,
        "argo_guidance": guidance_points,
        "active_learning": active_learning,
        "profile_samples": profile_samples,
        "n_argo_profiles_total": len(region_data["argo"]),
        # store compact grids for a couple of dates only (surface + one depth slice + uncertainty)
        "map_layers": {
            "date_used": dates[last_t],
            "sst": fields["sst"][last_t].round(2).tolist(),
            "temperature_by_depth": {
                str(int(depths[di])): temperature[last_t, :, :, di].round(2).tolist()
                for di in range(0, len(depths), 3)  # every 3rd depth level to keep payload small
            },
            "uncertainty_surface_mean": uncertainty[last_t].mean(axis=-1).round(3).tolist(),
        },
    }


def main():
    cfg = load_config()
    all_regions, dates, depths = run_pipeline(cfg, out_dir=OUT_DIR)
    bundle = {"dataset_label": cfg["demo"]["dataset_label"], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "depths_m": [float(d) for d in depths], "regions": {}}
    for key, region_data in all_regions.items():
        bundle["regions"][key] = process_region(key, region_data, dates, depths, cfg)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "backend_bundle.json"), "w") as f:
        json.dump(bundle, f)
    print(f"[build_demo_bundle] wrote {os.path.join(OUT_DIR, 'backend_bundle.json')}")

    # Build a compact frontend slice (drop the big per-cell map arrays where not needed for the demo UI grid render;
    # keep them but this is already coarse-grid so it stays small).
    frontend_bundle = {
        "dataset_label": bundle["dataset_label"],
        "depths_m": bundle["depths_m"],
        "regions": {},
    }
    for key, r in bundle["regions"].items():
        frontend_bundle["regions"][key] = {
            "region_name": r["region_name"], "bbox": r["bbox"],
            "lats": r["lats"], "lons": r["lons"], "dates": r["dates"],
            "metrics": r["metrics"]["overall"],
            "depthwise_model": r["metrics"]["depthwise_model"],
            "depthwise_baseline": r["metrics"]["depthwise_baseline"],
            "argo_validation": r["argo_validation"],
            "explainability": r["explainability"],
            "events": r["events"],
            "argo_guidance": r["argo_guidance"],
            "active_learning": r["active_learning"],
            "profile_samples": r["profile_samples"],
            "n_argo_profiles_total": r["n_argo_profiles_total"],
            "map_layers": r["map_layers"],
        }
    with open(os.path.join(OUT_DIR, "frontend_demo.json"), "w") as f:
        json.dump(frontend_bundle, f)
    print(f"[build_demo_bundle] wrote {os.path.join(OUT_DIR, 'frontend_demo.json')}")

    fsize = os.path.getsize(os.path.join(OUT_DIR, "frontend_demo.json"))
    print(f"[build_demo_bundle] frontend_demo.json size = {fsize/1024:.1f} KB")


if __name__ == "__main__":
    main()
