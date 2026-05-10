from __future__ import annotations

import numpy as np

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class XGBoostEntryModel:
    """XGBoost classifier wrapping XGBClassifier to EntryModel Protocol."""

    name = "xgboost"

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
        if not HAS_XGB:
            raise ImportError("xgboost is required for XGBoostEntryModel")
        params: dict = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric="mlogloss",
            **extra,
        )
        if device in ("gpu", "cuda"):
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        else:
            params["device"] = "cpu"
            params["n_jobs"] = n_jobs
        self._model = XGBClassifier(**params)
        self._fitted = False
        self._class_map: dict[int, int] = {}
        self._inv_map: dict[int, int] = {}

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        # XGBoost requires contiguous 0-based labels
        unique = sorted(set(int(v) for v in y_train))
        self._class_map = {c: i for i, c in enumerate(unique)}
        self._inv_map = {i: c for c, i in self._class_map.items()}
        y_mapped = np.array([self._class_map[int(v)] for v in y_train])
        self._model.fit(X_train, y_mapped)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X)
        return np.array([self._inv_map[int(v)] for v in raw])

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> list[int]:
        return [self._inv_map[i] for i in range(len(self._inv_map))]
