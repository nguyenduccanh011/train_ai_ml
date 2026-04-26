from __future__ import annotations

from typing import Protocol

import numpy as np


class EntryModel(Protocol):
    name: str

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit model on training data."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return canonical signals (-1, 0, 1)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return probabilities when available."""

    @property
    def classes_(self) -> list[int]:
        """Return model classes."""
