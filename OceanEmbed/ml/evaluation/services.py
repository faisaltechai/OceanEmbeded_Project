"""
OceanEmbed - Core scientific services (pure NumPy, no web framework deps).
These are imported directly by evaluate.py/inference.py AND by the FastAPI
service layer in backend/app/services/*, so the scientific logic is
defined exactly once.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------- validation
def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def bias(pred, true):
    return float(np.mean(np.asarray(pred) - np.asarray(true)))


def correlation(pred, true):
    pred, true = np.asarray(pred).ravel(), np.asarray(true).ravel()
    if pred.std() < 1e-9 or true.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(pred, true)[0, 1])


def validate_depthwise(pred_profiles: np.ndarray, true_profiles: np.ndarray, depths):
    """pred/true_profiles: (N, n_depths). Returns per-depth RMSE/Bias/Corr."""
    out = []
    for d_idx, depth in enumerate(depths):
        p, t = pred_profiles[:, d_idx], true_profiles[:, d_idx]
        out.append({"depth_m": float(depth), "rmse": rmse(p, t), "bias": bias(p, t), "correlation": correlation(p, t)})
    return out


def climatology_baseline(train_profiles: np.ndarray) -> np.ndarray:
    """Simplest defensible baseline: per-depth mean profile from the training period (section 23)."""
    return train_profiles.mean(axis=0)


# --------------------------------------------------------------------- thermocline / heat content
def thermocline_metrics(depths, profile):
    """
    Thermocline depth = depth of the maximum negative vertical temperature gradient
    (steepest drop). Thermocline temperature = profile value there. Gradient = that
    steepest dT/dz. This is a standard, documented, simple definition (not a fabricated one).
    """
    depths = np.asarray(depths, dtype=float)
    profile = np.asarray(profile, dtype=float)
    dT = np.diff(profile)
    dz = np.diff(depths)
    gradient = dT / dz
    steepest_idx = int(np.argmin(gradient))  # most negative = steepest drop
    return {
        "thermocline_depth_m": float((depths[steepest_idx] + depths[steepest_idx + 1]) / 2),
        "thermocline_temperature_c": float((profile[steepest_idx] + profile[steepest_idx + 1]) / 2),
        "max_gradient_c_per_m": float(gradient[steepest_idx]),
    }


def ocean_heat_content(depths, profile, depth_range=(0, 300), rho=1025.0, cp=4000.0, t_ref=0.0):
    """
    OHC proxy = rho * cp * integral( T(z) - t_ref , dz ) over the chosen depth range.
    This is a standard heat-content formulation (units: J/m^2) applied to our
    reconstructed profile -- documented, not presented as an operational/validated
    scientific product (section 10).
    """
    depths = np.asarray(depths, dtype=float)
    profile = np.asarray(profile, dtype=float)
    mask = (depths >= depth_range[0]) & (depths <= depth_range[1])
    z, T = depths[mask], profile[mask]
    if len(z) < 2:
        return {"heat_content_j_per_m2": None, "depth_range_m": list(depth_range)}
    integral = np.trapezoid(T - t_ref, z) if hasattr(np, "trapezoid") else np.trapz(T - t_ref, z)
    ohc = rho * cp * integral
    return {"heat_content_j_per_m2": float(ohc), "depth_range_m": list(depth_range)}


# --------------------------------------------------------------------- extreme event detector
class ExtremeEventDetector:
    """
    Transparent, threshold-based detector (prototype methodology, per section 7/43).
    Flags a grid cell/date as an event when its temperature anomaly relative to the
    per-cell climatology (mean over all demo dates) exceeds `std_threshold` standard
    deviations at any depth, OR when the mixed-layer depth collapses sharply
    (a proxy for rapid thermocline shoaling associated with subsurface heat events).
    """

    def __init__(self, std_threshold: float = 1.5, mld_collapse_ratio: float = 0.6,
                 min_abs_anomaly_c: float = 0.8, max_events_per_region: int = 25):
        self.std_threshold = std_threshold
        self.mld_collapse_ratio = mld_collapse_ratio
        self.min_abs_anomaly_c = min_abs_anomaly_c
        self.max_events_per_region = max_events_per_region

    @staticmethod
    def _anomaly_fields(temp_field: np.ndarray, mld_field: np.ndarray):
        """Shared climatology ingredients used by both detect() and
        event_risk_grid() -- one definition, not two divergent ones."""
        clim_mean = temp_field.mean(axis=0)  # (n_lat, n_lon, n_depths)
        clim_std = temp_field.std(axis=0) + 1e-6
        mld_clim = mld_field.mean(axis=0)
        return clim_mean, clim_std, mld_clim

    def event_risk_grid(self, temp_field: np.ndarray, mld_field: np.ndarray, date_index: int):
        """
        Continuous 0-1 "event risk" for every grid cell on ONE date, built
        from the exact same z-score / MLD-collapse ingredients detect() uses
        to flag discrete events -- for the Active Learning acquisition
        function (ml/active_learning/), which needs a continuous score
        rather than a binary flag. Not a second detector: same thresholds,
        same fields, just not binarized/top-N-capped.
        """
        clim_mean, clim_std, mld_clim = self._anomaly_fields(temp_field, mld_field)
        abs_anomaly = temp_field[date_index] - clim_mean
        z_anomaly = np.abs(abs_anomaly / clim_std)
        worst_z = z_anomaly.max(axis=-1)
        temp_component = worst_z / self.std_threshold  # 1.0 == right at detect()'s threshold

        mld_ratio = mld_field[date_index] / (mld_clim + 1e-6)
        mld_component = np.clip((self.mld_collapse_ratio - mld_ratio) / self.mld_collapse_ratio, 0, None)

        raw = np.maximum(temp_component, mld_component)
        return ArgoGuidanceService._minmax(raw)

    def detect(self, temp_field: np.ndarray, mld_field: np.ndarray, depths, lats, lons, dates):
        """
        temp_field: (n_dates, n_lat, n_lon, n_depths)
        mld_field:  (n_dates, n_lat, n_lon)

        A cell/date is flagged only when BOTH a z-score condition (relative to
        the short demo-period climatology) AND an absolute-degree condition are
        met -- with only 6 demo dates, z-score alone is statistically noisy, so
        requiring a minimum real temperature departure (min_abs_anomaly_c) avoids
        flagging normal day-to-day variability as an "event". Results are then
        capped to the top N most severe per region so the Extreme Events panel
        stays readable in a live demo (documented here, not silently dropped).
        """
        clim_mean, clim_std, mld_clim = self._anomaly_fields(temp_field, mld_field)

        candidates = []
        n_t, n_lat, n_lon, n_d = temp_field.shape
        for ti in range(n_t):
            abs_anomaly = temp_field[ti] - clim_mean               # (n_lat, n_lon, n_depths), degrees C
            z_anomaly = abs_anomaly / clim_std
            worst_depth_idx = np.argmax(np.abs(z_anomaly), axis=-1)
            worst_z = np.take_along_axis(z_anomaly, worst_depth_idx[..., None], axis=-1)[..., 0]
            worst_abs_c = np.take_along_axis(abs_anomaly, worst_depth_idx[..., None], axis=-1)[..., 0]
            mld_ratio = mld_field[ti] / (mld_clim + 1e-6)

            temp_flag = (np.abs(worst_z) >= self.std_threshold) & (np.abs(worst_abs_c) >= self.min_abs_anomaly_c)
            mld_flag = mld_ratio <= self.mld_collapse_ratio
            flagged = temp_flag | mld_flag
            ii_list, jj_list = np.where(flagged)
            for i, j in zip(ii_list, jj_list):
                d_idx = int(worst_depth_idx[i, j])
                reason = []
                if temp_flag[i, j]:
                    reason.append(f"temperature anomaly {worst_abs_c[i, j]:+.2f}C ({worst_z[i, j]:.1f} sigma) at {depths[d_idx]:.0f}m")
                if mld_flag[i, j]:
                    reason.append(f"mixed-layer depth collapsed to {mld_ratio[i, j]*100:.0f}% of climatology")
                severity_score = max(abs(worst_z[i, j]), (1 - mld_ratio[i, j]) * 3 if mld_flag[i, j] else 0)
                severity = "High" if severity_score >= self.std_threshold * 1.3 else "Medium"
                candidates.append({
                    "type": "Subsurface Heat Anomaly" if worst_abs_c[i, j] > 0 else "Subsurface Cold Anomaly",
                    "severity": severity, "_severity_score": float(severity_score),
                    "date": dates[ti], "lat": float(lats[i]), "lon": float(lons[j]),
                    "depth_m": float(depths[d_idx]), "anomaly_std": float(worst_z[i, j]),
                    "anomaly_c": float(worst_abs_c[i, j]),
                    "reason": "; ".join(reason),
                    "detection_method": "prototype threshold-based (std-dev anomaly + abs-degree floor + MLD collapse ratio)",
                })

        candidates.sort(key=lambda e: e["_severity_score"], reverse=True)
        top = candidates[: self.max_events_per_region]
        for e in top:
            e.pop("_severity_score", None)
        return top


# --------------------------------------------------------------------- ARGO guidance
class ArgoGuidanceService:
    """
    Scores each grid cell for how valuable an additional real observation there
    would be, using ONLY quantities the prototype actually computes:
      - prediction uncertainty (from MC-Dropout)
      - local Argo observation density (from the synthetic sparse Argo set)
      - anomaly strength (from ExtremeEventDetector-style z-score)
    priority_score = normalized weighted sum of the three (weights configurable).
    """

    def __init__(self, w_uncertainty=0.5, w_density=0.3, w_anomaly=0.2):
        self.w_uncertainty, self.w_density, self.w_anomaly = w_uncertainty, w_density, w_anomaly

    @staticmethod
    def _minmax(a):
        a = np.asarray(a, dtype=float)
        lo, hi = np.nanmin(a), np.nanmax(a)
        return (a - lo) / (hi - lo + 1e-9)

    def score(self, uncertainty_field, observation_density_field, anomaly_field):
        u = self._minmax(uncertainty_field)
        sparsity = 1 - self._minmax(observation_density_field)  # low density -> high score
        a = self._minmax(np.abs(anomaly_field))
        score = self.w_uncertainty * u + self.w_density * sparsity + self.w_anomaly * a
        return score, {"uncertainty_component": u, "sparsity_component": sparsity, "anomaly_component": a}

    @staticmethod
    def explain(u_val, sparsity_val, a_val):
        reasons = []
        if u_val > 0.6:
            reasons.append("high model uncertainty")
        if sparsity_val > 0.6:
            reasons.append("sparse existing Argo coverage")
        if a_val > 0.6:
            reasons.append("strong local anomaly")
        return "; ".join(reasons) if reasons else "moderate combined score"
