from __future__ import annotations

import numpy as np

try:
    from lightgbm import LGBMClassifier

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


class LightGBMEntryModel:
    """LightGBM classifier wrapping sklearn Pipeline interface to EntryModel Protocol."""

    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        device: str = "cpu",
        **extra,
    ) -> None:
        if not HAS_LGB:
            raise ImportError("lightgbm is required for LightGBMEntryModel")
        params: dict = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbose=-1,
            **extra,
        )
        if device in ("gpu",):
            params["device"] = "gpu"
        else:
            params["device"] = "cpu"
            params["n_jobs"] = n_jobs
        self._model = LGBMClassifier(**params)
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._model.fit(X_train, y_train)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> list[int]:
        return list(self._model.classes_)
