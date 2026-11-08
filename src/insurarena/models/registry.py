"""Model registry — name -> adapter factory."""
from __future__ import annotations

from insurarena.models.base import ModelAdapter
from insurarena.models.catboost_model import CatBoost
from insurarena.models.ebm import EBM
from insurarena.models.gbm import LightGBM
from insurarena.models.glm import GLM
from insurarena.models.rustystats_model import RustyStatsGLM
from insurarena.models.tabdpt_model import TabDPT
from insurarena.models.tabicl_model import TabICL
from insurarena.models.tabpfn3_model import TabPFN3
from insurarena.models.tboost_model import TBoost
from insurarena.models.xgb import XGBoost

_REGISTRY: dict[str, type[ModelAdapter]] = {
    "glm": GLM,
    "rustystats": RustyStatsGLM,
    "lightgbm": LightGBM,
    "xgboost": XGBoost,
    "catboost": CatBoost,
    "ebm": EBM,
    "t-boost": TBoost,   # depth-3 oblivious GBM, interpretable (functional-ANOVA); insurance-native
    # foundation-model black-box comparators (optional deps; exposure-agnostic)
    "tabdpt": TabDPT,     # regressor + classifier, CPU, no auth
    "tabicl": TabICL,     # classifier only, CPU, no auth
    "tabpfn3": TabPFN3,   # license-gated weights: one-time acceptance + TABPFN_TOKEN or a cached
                          # ~/.cache/tabpfn/auth_token; GPU recommended
}


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, hp: dict | None = None) -> ModelAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {list_models()}")
    return _REGISTRY[name](hp=hp)


def model_class(name: str) -> type[ModelAdapter]:
    return _REGISTRY[name]
