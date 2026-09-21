"""
OceanEmbed - train.py
Trains the physics-informed ocean embedding model per region on the demo
dataset, using a TEMPORAL split (last date held out) to avoid the
train/test leakage across time that section 39 warns against.
"""
from __future__ import annotations
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.data.data_pipeline import load_config, run_pipeline, build_feature_matrix, temporal_split
from ml.models.model import PhysicsInformedOceanNet

FEATURE_NAMES = ["sst", "sss", "ssh", "u_current", "v_current", "u_wind", "v_wind", "lat", "lon"]
SST_COL_IDX = 0

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(HERE, "..", "..", "data", "demo")


def train_region(region_key: str, region_data: dict, dates: list[str], depths: list[float], config: dict):
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
    print(f"[train] region={region_key}  train_n={X_train.shape[0]}  test_n={X_test.shape[0]}")
    history = net.train(
        X_train, y_train, sst_col_idx=SST_COL_IDX, physics_cfg=config["physics_loss"],
        epochs=model_cfg["epochs"], lr=model_cfg["learning_rate"], batch_size=model_cfg["batch_size"],
    )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, f"model_{region_key}.pkl"), "wb") as f:
        pickle.dump(net, f)
    with open(os.path.join(ARTIFACT_DIR, f"train_history_{region_key}.json"), "w") as f:
        json.dump({"loss_curve": history}, f)

    return net, (X_train, y_train, X_test, y_test, meta, train_mask, test_mask)


if __name__ == "__main__":
    cfg = load_config()
    all_regions, dates, depths = run_pipeline(cfg, out_dir=ARTIFACT_DIR)
    results = {}
    for key, region_data in all_regions.items():
        results[key] = train_region(key, region_data, dates, depths, cfg)
    print("[train] done ->", list(results.keys()))
