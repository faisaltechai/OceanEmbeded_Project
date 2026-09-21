# Explainable AI

**Method: feature ablation** (`ml/models/model.py: feature_ablation_importance`) — a
real, well-understood, model-agnostic explainability method. For each of the 9 input
features (SST, SSS, SSH, u/v current, u/v wind, lat, lon), the feature is replaced by
its dataset mean and the model re-predicts; importance = mean absolute shift in the
predicted 16-depth profile versus the un-perturbed prediction, normalized so all
feature contributions sum to 1.

This is explicitly **not** SHAP, integrated gradients, or attention weights — those
require either a differentiable-framework integration (SHAP/IG) or an attention
mechanism the current architecture doesn't have. Ablation was chosen because it is
correct to implement with the from-scratch NumPy model available in this sandbox and
is honestly labeled as such, per the brief's explicit instruction to "never label
arbitrary weights as explainability."

Typical result on the demo dataset: SST, SSS, and wind stress dominate; current
components contribute almost nothing — which is an honest reflection of how the
synthetic generator constructs its surface fields (currents were given a weak,
mostly-decorative role there), not a claim about real ocean physics.
