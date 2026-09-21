# Uncertainty Estimation

**Method:** Monte-Carlo Dropout (Gal & Ghahramani, 2016). Dropout (`dropout_rate` in
config, default 0.15) is applied to both hidden layers during training AND left ACTIVE
at inference. `predict_with_uncertainty()` runs `n_forward_passes` (default 20)
stochastic forward passes per grid cell; the mean is the point prediction, the
standard deviation across passes is the reported epistemic uncertainty.

**Confidence categories** (`ml/inference/inference.py: confidence_category`):
- `std <= 0.35°C` → **High** confidence
- `0.35–0.8°C` → **Medium**
- `> 0.8°C` → **Low**

These thresholds are prototype defaults tuned to this synthetic dataset's noise scale —
document and re-tune against real validation residuals before any operational use.

**Honest limitation:** MC-Dropout captures model (epistemic) uncertainty, not
observation/instrument uncertainty, and has not been calibrated against real held-out
Argo residuals (only synthetic ones here).
