from __future__ import annotations

import numpy as np


class NullExitModel:
    name = "null"

    def fit(self, X_train: np.ndarray, y_exit_train: np.ndarray) -> None:
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None
