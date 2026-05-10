from __future__ import annotations

from typing import Literal

import numpy as np


class EnsembleEntryModel:
    """Voting ensemble over multiple EntryModel instances.

    Skeleton implementation — combines predictions via majority vote (hard) or
    probability averaging (soft). Actual models injected at construction time.
    """

    name = "ensemble"

    def __init__(
        self,
        models: list,  # list[EntryModel]
        voting: Literal["hard", "soft"] = "soft",
    ) -> None:
        if not models:
            raise ValueError("EnsembleEntryModel requires at least one model")
        self.models = models
        self.voting = voting
        self._classes: list[int] | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        for m in self.models:
            m.fit(X_train, y_train)
        self._classes = self.models[0].classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.voting == "soft":
            proba = self.predict_proba(X)
            if proba is not None:
                idx = np.argmax(proba, axis=1)
                return np.array(self.classes_)[idx]
        # hard voting: majority
        preds = np.stack([m.predict(X) for m in self.models], axis=1)
        result = []
        for row in preds:
            vals, counts = np.unique(row, return_counts=True)
            result.append(vals[np.argmax(counts)])
        return np.array(result)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        probas = [m.predict_proba(X) for m in self.models]
        valid = [p for p in probas if p is not None]
        if not valid:
            return None
        return np.mean(valid, axis=0)

    @property
    def classes_(self) -> list[int]:
        if self._classes is None:
            raise RuntimeError("EnsembleEntryModel has not been fitted yet")
        return self._classes
