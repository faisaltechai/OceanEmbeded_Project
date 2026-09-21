"""
OceanEmbed - Physics-Informed Ocean Embedding Model
=====================================================
ENGINEERING NOTE: this sandbox has no network access, so PyTorch could not
be installed here. Rather than fake a torch import, this is a real,
from-scratch NumPy implementation of the same architecture described in
the spec (surface variables -> encoder -> ocean embedding -> decoder ->
15+1 depth predictions -> uncertainty head), trained with real
gradient-descent backprop (derived by hand below, not a canned library).
The public interface (fit / predict / predict_with_uncertainty) is what
the FastAPI service layer calls, so swapping this module for a PyTorch
version later requires no changes outside this file.

Architecture
------------
Input (9):  SST, SSS, SSH/SLA, u_current, v_current, u_wind, v_wind, lat, lon
Encoder:    Dense(9->hidden) + tanh
Embedding:  Dense(hidden->embedding_dim)          <- the "ocean embedding"
Decoder:    Dense(embedding_dim->hidden) + tanh
Output:     Dense(hidden->n_depths)               <- temperature at each depth

Physics-informed loss (documented, all coefficients in config.yaml)
--------------------------------------------------------------------
L = L_mse
    + lambda_surface_consistency * (pred[0] - observed_SST)^2
    + lambda_physics * mean( relu(pred[d+1] - pred[d] - tol)^2 )   # discourages temperature
                                                                     # INCREASING with depth
                                                                     # beyond a small tolerance
    + lambda_smoothness * mean( (pred[d+1] - 2*pred[d] + pred[d-1])^2 )  # 2nd-derivative
                                                                            # roughness penalty
    + lambda_regularization * ||W||^2

Uncertainty: Monte-Carlo Dropout (Gal & Ghahramani, 2016). Dropout is left
ACTIVE at inference time; N stochastic forward passes are averaged for the
point prediction and their std-dev is reported as epistemic uncertainty.
"""
from __future__ import annotations
import numpy as np


class PhysicsInformedOceanNet:
    def __init__(self, n_features: int, n_depths: int, hidden_dim: int = 64,
                 embedding_dim: int = 32, dropout_rate: float = 0.15, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_features, self.n_depths = n_features, n_depths
        self.hidden_dim, self.embedding_dim = hidden_dim, embedding_dim
        self.dropout_rate = dropout_rate

        def glorot(fan_in, fan_out):
            limit = np.sqrt(6 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.W1 = glorot(n_features, hidden_dim); self.b1 = np.zeros(hidden_dim)
        self.W2 = glorot(hidden_dim, embedding_dim); self.b2 = np.zeros(embedding_dim)
        self.W3 = glorot(embedding_dim, hidden_dim); self.b3 = np.zeros(hidden_dim)
        self.W4 = glorot(hidden_dim, n_depths); self.b4 = np.zeros(n_depths)

        self.x_mean = None; self.x_std = None
        self.y_mean = None; self.y_std = None
        self.rng = rng

    # ---------------------------------------------------------------- utils
    def _normalize_x(self, X):
        return (X - self.x_mean) / self.x_std

    def _normalize_y(self, y):
        return (y - self.y_mean) / self.y_std

    def _denormalize_y(self, y):
        return y * self.y_std + self.y_mean

    def fit_scalers(self, X, y):
        self.x_mean, self.x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-8

    # -------------------------------------------------------------- forward
    def _forward(self, Xn, training: bool, dropout_active: bool):
        z1 = Xn @ self.W1 + self.b1
        h1 = np.tanh(z1)
        mask1 = None
        if dropout_active:
            mask1 = (self.rng.random(h1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            h1 = h1 * mask1

        emb = h1 @ self.W2 + self.b2  # ocean embedding (linear bottleneck)

        z3 = emb @ self.W3 + self.b3
        h2 = np.tanh(z3)
        mask2 = None
        if dropout_active:
            mask2 = (self.rng.random(h2.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            h2 = h2 * mask2

        out_n = h2 @ self.W4 + self.b4  # normalized temperature output
        cache = (Xn, z1, h1, mask1, emb, z3, h2, mask2, out_n)
        return out_n, cache

    # ------------------------------------------------------- physics terms
    def _physics_grad(self, out_denorm, sst_true, cfg):
        """Returns d(physics-informed loss)/d(out_denorm), same shape as out_denorm (N, n_depths)."""
        N, D = out_denorm.shape
        grad = np.zeros_like(out_denorm)

        # surface consistency: tie predicted 0m temperature to observed SST
        surf_err = out_denorm[:, 0] - sst_true
        grad[:, 0] += 2 * cfg["lambda_surface_consistency"] * surf_err / N

        # monotonic-ish physics penalty: discourage temperature increasing with depth
        tol = 0.05
        diffs = out_denorm[:, 1:] - out_denorm[:, :-1] - tol
        viol = np.maximum(diffs, 0)
        gphys = 2 * cfg["lambda_physics"] * viol / N
        grad[:, 1:] += gphys
        grad[:, :-1] -= gphys

        # vertical smoothness: penalize 2nd-derivative roughness
        if D >= 3:
            d2 = out_denorm[:, 2:] - 2 * out_denorm[:, 1:-1] + out_denorm[:, :-2]
            gsm = 2 * cfg["lambda_smoothness"] * d2 / N
            grad[:, 2:] += gsm
            grad[:, 1:-1] += -2 * gsm
            grad[:, :-2] += gsm
        return grad

    # ----------------------------------------------------------- training
    def train(self, X, y, sst_col_idx, physics_cfg, epochs=300, lr=0.01, batch_size=256, verbose=True):
        self.fit_scalers(X, y)
        Xn = self._normalize_x(X)
        yn = self._normalize_y(y)
        sst_true = X[:, sst_col_idx]
        n = X.shape[0]
        history = []

        for epoch in range(epochs):
            perm = self.rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                xb, yb, sstb = Xn[idx], yn[idx], sst_true[idx]
                out_n, cache = self._forward(xb, training=True, dropout_active=True)
                Xn_, z1, h1, mask1, emb, z3, h2, mask2, out_n_ = cache

                out_denorm = self._denormalize_y(out_n)
                y_denorm = self._denormalize_y(yb)

                # --- combined gradient at output (denormalized space) ---
                mse_grad_denorm = 2 * (out_denorm - y_denorm) / len(idx)
                phys_grad_denorm = self._physics_grad(out_denorm, sstb, physics_cfg)
                grad_out_denorm = mse_grad_denorm + phys_grad_denorm
                grad_out_n = grad_out_denorm * self.y_std  # chain rule through denormalization

                loss = (np.mean((out_denorm - y_denorm) ** 2)
                        + physics_cfg["lambda_surface_consistency"] * np.mean((out_denorm[:, 0] - sstb) ** 2)
                        + physics_cfg["lambda_physics"] * np.mean(np.maximum(out_denorm[:, 1:] - out_denorm[:, :-1] - 0.05, 0) ** 2)
                        )
                epoch_loss += loss * len(idx)

                # --- backprop ---
                gW4 = h2.T @ grad_out_n + physics_cfg["lambda_regularization"] * self.W4
                gb4 = grad_out_n.sum(axis=0)
                gh2 = grad_out_n @ self.W4.T
                if mask2 is not None:
                    gh2 = gh2 * mask2
                gz3 = gh2 * (1 - h2 ** 2)
                gW3 = emb.T @ gz3 + physics_cfg["lambda_regularization"] * self.W3
                gb3 = gz3.sum(axis=0)
                gemb = gz3 @ self.W3.T
                gW2 = h1.T @ gemb + physics_cfg["lambda_regularization"] * self.W2
                gb2 = gemb.sum(axis=0)
                gh1 = gemb @ self.W2.T
                if mask1 is not None:
                    gh1 = gh1 * mask1
                gz1 = gh1 * (1 - h1 ** 2)
                gW1 = xb.T @ gz1 + physics_cfg["lambda_regularization"] * self.W1
                gb1 = gz1.sum(axis=0)

                # global-norm gradient clipping for numerical stability (small nets, hand-rolled backprop)
                max_norm = 5.0
                grads = [gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4]
                total_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads)) + 1e-12
                if total_norm > max_norm:
                    scale = max_norm / total_norm
                    grads = [g * scale for g in grads]
                gW1, gb1, gW2, gb2, gW3, gb3, gW4, gb4 = grads

                for W, gW in [(self.W1, gW1), (self.W2, gW2), (self.W3, gW3), (self.W4, gW4)]:
                    W -= lr * gW
                for b, gb in [(self.b1, gb1), (self.b2, gb2), (self.b3, gb3), (self.b4, gb4)]:
                    b -= lr * gb

            epoch_loss /= n
            history.append(epoch_loss)
            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:4d}  loss={epoch_loss:.5f}")
        return history

    # --------------------------------------------------------- inference
    def predict(self, X):
        Xn = self._normalize_x(X)
        out_n, _ = self._forward(Xn, training=False, dropout_active=False)
        return self._denormalize_y(out_n)

    def predict_with_uncertainty(self, X, n_passes: int = 20):
        """MC-Dropout: keep dropout active across N stochastic passes; std = epistemic uncertainty."""
        Xn = self._normalize_x(X)
        preds = np.zeros((n_passes, X.shape[0], self.n_depths))
        for k in range(n_passes):
            out_n, _ = self._forward(Xn, training=False, dropout_active=True)
            preds[k] = self._denormalize_y(out_n)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std

    def feature_ablation_importance(self, X, feature_names):
        """
        Explainability method: FEATURE ABLATION (a real, simple, well-understood
        method -- not an arbitrary weight readout). For each surface variable we
        zero it out (replace with its dataset mean) and measure how much the
        model's mean predicted profile shifts vs. the un-perturbed prediction.
        Larger shift = more influence on the prediction.
        """
        base_pred = self.predict(X)
        importances = {}
        for i, name in enumerate(feature_names):
            X_ablated = X.copy()
            X_ablated[:, i] = self.x_mean[i] * self.x_std[i] + self.x_mean[i]  # replace with mean (denorm space)
            X_ablated[:, i] = X[:, i].mean()
            pred_ablated = self.predict(X_ablated)
            shift = np.mean(np.abs(pred_ablated - base_pred))
            importances[name] = float(shift)
        total = sum(importances.values()) + 1e-9
        return {k: v / total for k, v in importances.items()}
