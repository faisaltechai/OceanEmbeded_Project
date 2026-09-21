# Physics-Informed Learning

Implemented in `ml/models/model.py`, applied **during training** (not just inference),
gradients derived and coded by hand (`_physics_grad`). Three constraints, each with a
configurable weight in `ml/configs/config.yaml: physics_loss`:

1. **Surface consistency** (`lambda_surface_consistency`): penalizes disagreement between
   the model's predicted 0 m temperature and the observed SST for that cell.
2. **Vertical monotonicity / physics penalty** (`lambda_physics`): penalizes the predicted
   profile for *increasing* with depth beyond a small tolerance (0.05°C) — real ocean
   profiles are overwhelmingly non-increasing with depth away from the mixed layer.
3. **Vertical smoothness** (`lambda_smoothness`): penalizes large 2nd-derivative
   roughness across depth, discouraging unphysical oscillation.

What this does **not** claim: it is not a full primitive-equation ocean model, it does
not enforce mass/heat conservation, and it has not been validated against real ARGO
data (the ARGO in this build is synthetic). It is a documented, working, lightweight
regularization scheme — exactly the kind of "physics-informed AI" a hackathon
prototype can honestly claim.
