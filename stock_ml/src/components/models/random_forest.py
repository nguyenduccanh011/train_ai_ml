from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class RandomForestEntryModel:
    """Random Forest classifier with RobustScaler to EntryModel Protocol."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 12,
        min_samples_leaf: int = 20,
        random_state: int = 42,
        n_jobs: int = -1,
        **extra,
    ) -> None:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
            **extra,
        )
        self._pipeline = Pipeline([("scaler", RobustScaler()), ("model", clf)])
        self._fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._pipeline.fit(X_train, y_train)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return self._pipeline.predict_proba(X)

    @property
    def classes_(self) -> list[int]:
        return list(self._pipeline.named_steps["model"].classes_)
