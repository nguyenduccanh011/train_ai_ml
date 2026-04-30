from __future__ import annotations

from typing import Any


def _build_lightgbm(**kwargs: Any):
    from src.components.models.lightgbm_classifier import LightGBMEntryModel

    return LightGBMEntryModel(**kwargs)


def _build_xgboost(**kwargs: Any):
    from src.components.models.xgboost_classifier import XGBoostEntryModel

    return XGBoostEntryModel(**kwargs)


def _build_catboost(**kwargs: Any):
    from src.components.models.catboost_classifier import CatBoostEntryModel

    return CatBoostEntryModel(**kwargs)


def _build_random_forest(**kwargs: Any):
    from src.components.models.random_forest import RandomForestEntryModel

    kwargs.pop("device", None)
    return RandomForestEntryModel(**kwargs)


def _build_gru(**kwargs: Any):
    from src.components.models.gru_seq import GRUEntryModel

    return GRUEntryModel(**kwargs)


_BUILDERS: dict[str, Any] = {
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
    "random_forest": _build_random_forest,
    "gru": _build_gru,
}


def get_model(name: str, **kwargs: Any):
    """Instantiate an EntryModel by name with optional hyperparameter overrides."""
    if name not in _BUILDERS:
        raise KeyError(f"Unknown entry model: {name!r}. Available: {list_models()}")
    return _BUILDERS[name](**kwargs)


def list_models() -> list[str]:
    return list(_BUILDERS.keys())
