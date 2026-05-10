from __future__ import annotations

import numpy as np

try:
    from catboost import CatBoostClassifier

    HAS_CAT = True
except ImportError:
    HAS_CAT = False


class CatBoostEntryModel:
    """CatBoost classifier wrapping CatBoostClassifier to EntryModel Protocol."""

    name = "catboost"

    def __init__(
        self,
        iterations: int = 300,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        device: str = "cpu",
        **extra,
    ) -> None:
        if not HAS_CAT:
            raise ImportError("catboost is required for CatBoostEntryModel")
        params: dict = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_state,
            verbose=0,
            **extra,
        )
        if device in ("gpu", "cuda"):
            params["task_type"] = "GPU"
            params["devices"] = "0"
        else:
            params["task_type"] = "CPU"
        self._model = CatBoostClassifier(**params)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> list[int]:
        return [int(c) for c in self._model.classes_]
