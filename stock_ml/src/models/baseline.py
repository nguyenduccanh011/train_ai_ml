"""Baseline classifier: LightGBM if available, else sklearn HistGradientBoosting.

The model accepts integer labels {-1, 0, 1}. We map them to {0, 1, 2} internally
because some classifiers don't accept negative labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LABEL_FWD = {-1.0: 0, 0.0: 1, 1.0: 2}
_LABEL_BACK = {0: -1, 1: 0, 2: 1}


def _build_classifier(seed: int):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=50,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            random_state=seed,
        )


@dataclass
class BaselineModel:
    seed: int = 42

    def __post_init__(self) -> None:
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaselineModel:
        y_mapped = np.array([_LABEL_FWD[float(v)] for v in y], dtype=np.int32)
        self._clf = _build_classifier(self.seed)
        self._clf.fit(X, y_mapped)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        raw = self._clf.predict(X)
        return np.array([_LABEL_BACK[int(v)] for v in raw], dtype=np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None or not hasattr(self._clf, "predict_proba"):
            raise RuntimeError("model not fitted or proba not supported")
        return self._clf.predict_proba(X)
