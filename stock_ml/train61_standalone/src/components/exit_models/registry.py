from __future__ import annotations

from typing import Any

from src.components.exit_models.base import ExitModel
from src.components.exit_models.ml_exit import MLExitModel
from src.components.exit_models.no_exit_model import NullExitModel


def _build_null(**kwargs: Any) -> ExitModel:
    return NullExitModel()


def _build_lightgbm(**kwargs: Any) -> ExitModel:
    return MLExitModel("lightgbm", **kwargs)


def _build_xgboost(**kwargs: Any) -> ExitModel:
    return MLExitModel("xgboost", **kwargs)


def _build_catboost(**kwargs: Any) -> ExitModel:
    return MLExitModel("catboost", **kwargs)


_BUILDERS: dict[str, Any] = {
    "null": _build_null,
    "lightgbm": _build_lightgbm,
    "xgboost": _build_xgboost,
    "catboost": _build_catboost,
}


def get_exit_model(name: str, **kwargs: Any) -> ExitModel:
    if name not in _BUILDERS:
        raise KeyError(f"Unknown exit model: {name!r}. Available: {list_exit_models()}")
    return _BUILDERS[name](**kwargs)


def list_exit_models() -> list[str]:
    return list(_BUILDERS.keys())
