"""
OceanEmbed tests - Active Learning & Adaptive Sampling (ml/active_learning/).

Run with `pytest tests/test_active_learning.py` (if pytest is installed) or
`python3 tests/test_active_learning.py` (plain runner, no pytest dependency --
matches every other test file in this repo).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.evaluation.services import ExtremeEventDetector
from ml.active_learning.acquisition import AcquisitionFunction, _minmax01
from ml.active_learning.sampling_optimizer import (
    select_diverse_locations, recommend_platform, recommend_depths,
)
from ml.active_learning.adaptive_frequency import recommend_frequency
from ml.active_learning.uncertainty_map import compute_grid_metrics
from ml.active_learning import build_active_learning_bundle


# --------------------------------------------------------------------- helpers
def _synthetic_fields(n_t=4, n_lat=6, n_lon=7, depths=(0, 50, 100, 300, 800), seed=2):
    rng = np.random.default_rng(seed)
    n_d = len(depths)
    temperature = 25 - np.cumsum(rng.uniform(1.0, 4.0, size=(n_t, n_lat, n_lon, n_d)), axis=-1)
    uncertainty = rng.uniform(0.05, 0.6, size=(n_t, n_lat, n_lon, n_d))
    mld = rng.uniform(20, 100, size=(n_t, n_lat, n_lon))
    lats = np.linspace(5, 15, n_lat)
    lons = np.linspace(80, 90, n_lon)
    dates = [f"2026-03-{10 + i:02d}" for i in range(n_t)]
    density = rng.integers(0, 6, size=(n_lat, n_lon)).astype(float)
    argo_records = [{"lat_idx": int(i), "lon_idx": int(j), "date": dates[0]}
                     for i in range(0, n_lat, 2) for j in range(0, n_lon, 2)]
    return temperature, uncertainty, mld, list(depths), lats, lons, dates, density, argo_records


# --------------------------------------------------------------------- acquisition
def test_minmax01_bounds_and_nan_safe():
    a = np.array([1.0, 5.0, 3.0, np.nan])
    out = _minmax01(a)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[-1] == 0.0  # NaN input -> 0, never propagated


def test_acquisition_weights_always_renormalize_to_one():
    fn = AcquisitionFunction({"uncertainty": 5.0})  # deliberately unnormalized override
    assert abs(sum(fn.weights.values()) - 1.0) < 1e-9


def test_acquisition_score_ranks_by_uncertainty_when_other_factors_equal():
    metrics = {
        "uncertainty_score": np.array([[0.1, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.1]]),
        "temperature_anomaly": np.zeros((3, 3)),
        "thermocline_gradient": np.zeros((3, 3)),
        "event_risk": np.zeros((3, 3)),
        "observation_density": np.ones((3, 3)),
    }
    fn = AcquisitionFunction()
    acquisition, components = fn.score(metrics)
    assert acquisition.shape == (3, 3)
    assert acquisition.min() >= 0.0 and acquisition.max() <= 1.0 + 1e-9
    assert np.unravel_index(np.argmax(acquisition), acquisition.shape) == (1, 1)


def test_acquisition_explain_reflects_crossed_components_only():
    comps = {"uncertainty": 0.9, "anomaly": 0.1, "thermocline_gradient": 0.05,
             "event_risk": 0.0, "observation_gap": 0.65}
    reason = AcquisitionFunction.explain(comps, threshold=0.6)
    assert "high prediction uncertainty" in reason
    assert "sparse nearby observations" in reason
    assert "anomaly" not in reason and "thermocline" not in reason and "event" not in reason


def test_acquisition_explain_falls_back_when_nothing_crosses_threshold():
    comps = {"uncertainty": 0.2, "anomaly": 0.1, "thermocline_gradient": 0.1,
             "event_risk": 0.0, "observation_gap": 0.3}
    assert AcquisitionFunction.explain(comps) == "moderate combined score across all factors"


# --------------------------------------------------------------------- spatial diversity
def test_select_diverse_locations_respects_min_distance_and_ranking():
    lats = np.arange(10, dtype=float)
    lons = np.arange(10, dtype=float)
    acquisition = np.zeros((10, 10))
    acquisition[1, 1] = 0.9
    acquisition[1, 2] = 0.85  # adjacent to (1,1) -- must be skipped at min_distance=2.0
    acquisition[5, 5] = 0.7
    acquisition[8, 8] = 0.6

    selected = select_diverse_locations(acquisition, lats, lons, top_n=3, min_distance_deg=2.0)

    assert selected[0] == (1, 1), "highest-scoring cell must be selected first"
    assert (1, 2) not in selected, "too close to an already-selected cell"
    assert set(selected) == {(1, 1), (5, 5), (8, 8)}
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            ia, ja = selected[a]
            ib, jb = selected[b]
            dist = np.hypot(lats[ia] - lats[ib], lons[ja] - lons[jb])
            assert dist >= 2.0 - 1e-9


def test_select_diverse_locations_returns_fewer_when_grid_too_small():
    lats, lons = np.array([0.0, 0.1]), np.array([0.0, 0.1])  # everything within min_distance
    acquisition = np.array([[0.9, 0.5], [0.3, 0.1]])
    selected = select_diverse_locations(acquisition, lats, lons, top_n=4, min_distance_deg=5.0)
    assert len(selected) == 1  # only the single highest-scoring cell fits


# --------------------------------------------------------------------- platform / depths
_RULES = {"deep_focus_m": 300.0, "gradient_threshold_c_per_m": 0.05, "event_risk_threshold": 0.7}


def test_recommend_platform_auv_for_high_event_risk():
    profile = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    assert recommend_platform(profile, [0, 50, 100, 300, 500], 0.01, event_risk=0.9, rules=_RULES) == "AUV"


def test_recommend_platform_glider_for_steep_gradient():
    profile = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    assert recommend_platform(profile, [0, 50, 100, 300, 500], -0.2, event_risk=0.1, rules=_RULES) == "Glider"


def test_recommend_platform_bgc_argo_for_deep_concentrated_uncertainty():
    profile = np.array([0.05, 0.05, 0.05, 0.5, 0.5])  # last two depths (>=300) are the deep ones
    assert recommend_platform(profile, [0, 50, 100, 300, 500], 0.01, event_risk=0.1, rules=_RULES) == "BGC-Argo"


def test_recommend_platform_defaults_to_argo():
    profile = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # uniform -> deep is not > shallow
    assert recommend_platform(profile, [0, 50, 100, 300, 500], 0.01, event_risk=0.1, rules=_RULES) == "Argo"


def test_recommend_depths_uses_only_standard_levels():
    depths = [0, 25, 50, 75, 100, 200, 500, 900, 1000]
    out = recommend_depths(depths, thermocline_depth_m=50, n_deep_anchors=2)
    assert out == [0, 50, 900, 1000]
    assert set(out).issubset(set(depths))


# --------------------------------------------------------------------- adaptive frequency
def test_recommend_frequency_bands():
    low = recommend_frequency(0.1, is_extreme_event=False, low_thresh=0.35, high_thresh=0.8)
    med = recommend_frequency(0.5, is_extreme_event=False, low_thresh=0.35, high_thresh=0.8)
    high = recommend_frequency(0.9, is_extreme_event=False, low_thresh=0.35, high_thresh=0.8)
    extreme = recommend_frequency(0.05, is_extreme_event=True, low_thresh=0.35, high_thresh=0.8)

    assert low["priority_level"] == "normal" and low["recommended_sampling_interval_days"] == 10
    assert med["priority_level"] == "increased" and med["recommended_sampling_interval_days"] == 5
    assert high["priority_level"] == "intensive" and high["recommended_sampling_interval_days"] == 2
    assert extreme["priority_level"] == "priority" and extreme["recommended_sampling_interval_days"] == 1
    # interval strictly decreases as priority rises
    intervals = [low["recommended_sampling_interval_days"], med["recommended_sampling_interval_days"],
                 high["recommended_sampling_interval_days"], extreme["recommended_sampling_interval_days"]]
    assert intervals == sorted(intervals, reverse=True)


# --------------------------------------------------------------------- event risk (extends ExtremeEventDetector)
def test_event_risk_grid_flags_injected_anomaly_highest():
    rng = np.random.default_rng(0)
    n_t, n_lat, n_lon, n_d = 5, 4, 4, 3
    base = 20 + rng.normal(0, 0.05, size=(n_t, n_lat, n_lon, n_d))
    base[-1, 1, 1, :] += 5.0  # obvious injected anomaly on the last date at (1,1)
    mld = np.full((n_t, n_lat, n_lon), 50.0)
    detector = ExtremeEventDetector(std_threshold=1.5, min_abs_anomaly_c=0.8)

    risk = detector.event_risk_grid(base, mld, date_index=-1)

    assert risk.shape == (n_lat, n_lon)
    assert (risk >= 0).all() and (risk <= 1 + 1e-9).all()
    assert np.unravel_index(np.argmax(risk), risk.shape) == (1, 1)


# --------------------------------------------------------------------- grid metrics + full bundle
def test_compute_grid_metrics_shapes_and_consistency():
    temperature, uncertainty, mld, depths, lats, lons, dates, density, argo_records = _synthetic_fields()
    detector = ExtremeEventDetector()

    metrics = compute_grid_metrics(temperature, uncertainty, mld, depths, lats, lons, dates,
                                    -1, density, argo_records, detector)

    expected_keys = {"uncertainty_score", "prediction_variance", "temperature_anomaly",
                      "thermocline_gradient", "event_risk", "observation_density",
                      "distance_to_observation_deg"}
    assert set(metrics.keys()) == expected_keys
    n_lat, n_lon = len(lats), len(lons)
    for k, v in metrics.items():
        assert v.shape == (n_lat, n_lon), f"{k} has shape {v.shape}, expected {(n_lat, n_lon)}"
    assert np.allclose(metrics["prediction_variance"], metrics["uncertainty_score"] ** 2)
    assert metrics["distance_to_observation_deg"][0, 0] == 0.0  # an observation sits exactly there


def test_build_active_learning_bundle_end_to_end():
    temperature, uncertainty, mld, depths, lats, lons, dates, density, argo_records = _synthetic_fields()
    config = {
        "spatial_diversity": {"min_distance_deg": 2.0, "default_top_n": 5},
        "frequency_thresholds": {"low_max_c": 0.2, "medium_max_c": 0.4},
        "platform_rules": _RULES,
    }

    bundle = build_active_learning_bundle(temperature, uncertainty, mld, depths, lats, lons, dates,
                                           -1, density, argo_records, config)

    assert bundle["status"] == "model-derived"
    recs = bundle["recommendations"]
    assert 0 < len(recs) <= 5

    scores = [r["priority_score"] for r in recs]
    assert scores == sorted(scores, reverse=True), "recommendations must be rank-ordered"

    for i, r in enumerate(recs, start=1):
        assert r["rank"] == i
        assert r["recommended_platform"] in {"AUV", "Glider", "Argo", "BGC-Argo"}
        assert set(r["recommended_depths_m"]).issubset(set(float(d) for d in depths))
        assert r["adaptive_sampling"]["priority_level"] in {"normal", "increased", "intensive", "priority"}
        assert 0.0 <= r["priority_score"] <= 1.0 + 1e-9

    # spatial diversity actually enforced among the returned recommendations
    for a in range(len(recs)):
        for b in range(a + 1, len(recs)):
            dist = np.hypot(recs[a]["latitude"] - recs[b]["latitude"], recs[a]["longitude"] - recs[b]["longitude"])
            assert dist >= 2.0 - 1e-9

    assert len(bundle["grid"]["acquisition_score"]) == len(lats)
    assert len(bundle["grid"]["acquisition_score"][0]) == len(lons)


def test_build_active_learning_bundle_recommendation_count_matches_requested_top_n():
    temperature, uncertainty, mld, depths, lats, lons, dates, density, argo_records = _synthetic_fields(
        n_lat=10, n_lon=10)
    small_top_n = {"spatial_diversity": {"min_distance_deg": 0.5, "default_top_n": 3}}
    bundle = build_active_learning_bundle(temperature, uncertainty, mld, depths, lats, lons, dates,
                                           -1, density, argo_records, small_top_n)
    assert len(bundle["recommendations"]) == 3


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
