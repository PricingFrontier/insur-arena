"""InsurArena — a TabArena-style benchmark of tabular modelling techniques on insurance datasets.

Compares modelling techniques (GLM, GBM, ...) across insurance-pricing datasets using
actuarially-correct metrics (Poisson/Gamma/Tweedie deviance, exposure-weighted Gini,
calibration). One fitted model per split — no benchmark-level bagging / blending / stacking.
See SPEC.md.
"""

__version__ = "0.1.0"
