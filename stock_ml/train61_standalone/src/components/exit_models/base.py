from __future__ import annotations

from typing import Protocol

import numpy as np


class ExitModel(Protocol):
    name: str

    def fit(self, X_train: np.ndarray, y_exit_train: np.ndarray) -> None:
        """Fit model on binary exit labels."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary signals (0 or 1)."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return probabilities when available."""
