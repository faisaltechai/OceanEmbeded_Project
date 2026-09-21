"""
OceanEmbed tests - ML core.
Run with `pytest tests/test_ml_pipeline.py` (if pytest is installed) or
`python3 tests/test_ml_pipeline.py` (plain runner, no pytest dependency --
this sandbox has no network access to install pytest).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ml.data.data_pipeline import load_config, generate_raw_fields, build_feature_matrix, temporal_split, quality_control
from ml.models.model import PhysicsInformedOceanNet
from ml.evaluation.services import rmse, bias, correlation, thermocline_metrics, ocean_heat_content, ExtremeEventDetector, validate_depthwise


def _small_fields():
    cfg = load_config()
    bbox = cfg["regions"]["bay_of_bengal"]["bbox"]
    dates = ["2026-01-10", "2026-01-11", "2026-01-12"]
    depths = cfg["depths_m"]
    fields = generate_raw_fields(bbox, resolution_deg=2.0, dates=dates, depths_m=depths, seed=1)
    return quality_control(fields), depths


def test_data_pipeline_shapes():
    fields, depths = _small_fields()
    n_t, n_lat, n_lon = fields["sst"].shape
    assert fields["true_temp"].shape == (n_t, n_lat, n_lon, len(depths))
    assert n_t == 3 and n_lat > 0 and n_lon > 0
    assert not np.isnan(fields["sst"]).any()
    assert fields["sst"].min() >= -2 and fields["sst"].max() <= 36


def test_feature_matrix_and_temporal_split():
    fields, depths = _small_fields()
    X, y, meta = build_feature_matrix(fields, "bay_of_bengal")
    assert X.shape[1] == 9
    assert y.shape[1] == len(depths)
    assert X.shape[0] == y.shape[0] == len(meta)
    train_idx, test_idx = temporal_split(n_dates=3, test_dates=1)
    assert train_idx == [0, 1] and test_idx == [2]
    assert set(train_idx).isdisjoint(set(test_idx))


def test_model_forward_shapes():
    net = PhysicsInformedOceanNet(n_features=9, n_depths=16, hidden_dim=8, embedding_dim=4, seed=0)
    X = np.random.default_rng(0).normal(size=(20, 9))
    y = np.random.default_rng(1).normal(size=(20, 16))
    net.fit_scalers(X, y)
    pred = net.predict(X)
    assert pred.shape == (20, 16)
    mean, std = net.predict_with_uncertainty(X, n_passes=5)
    assert mean.shape == (20, 16) and std.shape == (20, 16)
    assert (std >= 0).all()


def test_model_training_reduces_loss():
    fields, depths = _small_fields()
    X, y, meta = build_feature_matrix(fields, "bay_of_bengal")
    cfg = load_config()
    net = PhysicsInformedOceanNet(n_features=X.shape[1], n_depths=y.shape[1],
                                   hidden_dim=16, embedding_dim=8, seed=0)
    history = net.train(X, y, sst_col_idx=0, physics_cfg=cfg["physics_loss"],
                         epochs=50, lr=0.005, batch_size=64, verbose=False)
    assert len(history) == 50
    assert history[-1] < history[0], "training loss should decrease"
    assert np.isfinite(history[-1]), "loss must not diverge to NaN/inf"


def test_physics_penalty_discourages_depth_increase():
    """A profile that INCREASES with depth should get a physics_grad pushing it down,
    while a smoothly decreasing profile should get near-zero physics_grad."""
    net = PhysicsInformedOceanNet(n_features=9, n_depths=4, seed=0)
    increasing = np.array([[10.0, 12.0, 14.0, 16.0]])  # unphysical: warms with depth
    decreasing = np.array([[20.0, 15.0, 10.0, 5.0]])    # physically typical
    cfg = {"lambda_surface_consistency": 0.0, "lambda_physics": 1.0, "lambda_smoothness": 0.0, "lambda_regularization": 0.0}
    g_inc = net._physics_grad(increasing, sst_true=np.array([10.0]), cfg=cfg)
    g_dec = net._physics_grad(decreasing, sst_true=np.array([20.0]), cfg=cfg)
    assert np.abs(g_inc).sum() > np.abs(g_dec).sum()


def test_validation_metrics_sanity():
    true = np.array([10.0, 12.0, 9.0, 11.0])
    perfect = true.copy()
    noisy = true + np.array([1.0, -1.0, 1.0, -1.0])
    assert rmse(perfect, true) == 0.0
    assert rmse(noisy, true) > 0.0
    assert abs(bias(perfect, true)) < 1e-9
    assert correlation(perfect, true) > 0.999


def test_thermocline_and_heat_content():
    depths = [0, 50, 100, 200, 400, 1000]
    profile = [28, 27, 20, 10, 6, 4]  # steepest drop between 50m and 100m
    therm = thermocline_metrics(depths, profile)
    assert 50 <= therm["thermocline_depth_m"] <= 100
    ohc = ocean_heat_content(depths, profile, depth_range=(0, 200))
    assert ohc["heat_content_j_per_m2"] > 0  # warm profile above freezing -> positive OHC vs 0degC ref


def test_extreme_event_detector_flags_injected_anomaly():
    rng = np.random.default_rng(0)
    n_t, n_lat, n_lon, n_d = 5, 4, 4, 3
    base = 20 + rng.normal(0, 0.05, size=(n_t, n_lat, n_lon, n_d))  # low-noise baseline
    base[-1, 1, 1, :] += 5.0  # inject an obvious anomaly on the last date at (1,1)
    mld = np.full((n_t, n_lat, n_lon), 50.0)
    depths = [0, 50, 100]
    lats, lons = np.arange(n_lat, dtype=float), np.arange(n_lon, dtype=float)
    dates = [f"2026-01-{10+i}" for i in range(n_t)]
    detector = ExtremeEventDetector(std_threshold=1.5, min_abs_anomaly_c=0.8)
    events = detector.detect(base, mld, depths, lats, lons, dates)
    hit = [e for e in events if e["lat"] == 1.0 and e["lon"] == 1.0 and e["date"] == dates[-1]]
    assert len(hit) >= 1, "the obviously-injected anomaly must be detected"


def test_depthwise_validation_length_matches_depths():
    depths = [0, 50, 100, 200]
    pred = np.random.default_rng(0).normal(20, 2, size=(10, 4))
    true = pred + np.random.default_rng(1).normal(0, 0.5, size=(10, 4))
    out = validate_depthwise(pred, true, depths)
    assert len(out) == len(depths)
    assert all(set(row.keys()) == {"depth_m", "rmse", "bias", "correlation"} for row in out)


if __name__ == "__main__":
    # plain runner (no pytest dependency available in this sandbox)
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
