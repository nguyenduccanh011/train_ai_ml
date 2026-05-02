from __future__ import annotations

from typing import Any

import numpy as np

from src.components.models.registry import get_model


class MLExitModel:
    def __init__(self, model_type: str = "lightgbm", **kwargs: Any) -> None:
        self.model_type = model_type
        self.kwargs = kwargs
        self.name = model_type
        self.model = get_model(model_type, **kwargs)

    def fit(self, X_train: np.ndarray, y_exit_train: np.ndarray) -> None:
        self.model.fit(X_train, y_exit_train.astype(int))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X)).astype(int).clip(0, 1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if not hasattr(self.model, "predict_proba"):
            return None
        return self.model.predict_proba(X)
