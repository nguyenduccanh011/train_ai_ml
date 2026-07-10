"""Regime models for market state detection.

Stub implementations ready for extension with real regime logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NaiveRegimeModel:
    """Stub regime model — always returns neutral regime (1).

    Placeholder for real regime detection logic (volatility, trending, mean-reversion).
    Always returns regime=1 (neutral), allowing full signal pass-through.

    Use this as a template:
    - fit() trains regime classifier on regime labels
    - predict() returns regime codes {0, 1, 2, ...}
    - Regimes can be used to filter entry signals or scale position size
    """

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> NaiveRegimeModel:
        """No-op stub. Real implementation would train classifier.

        Args:
            X: feature matrix
            y: regime labels {0, 1, 2, ...}

        Returns:
            self
        """
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return neutral regime for all samples.

        Args:
            X: feature matrix of shape (n_samples, n_features)

        Returns:
            regime codes, all set to 1 (neutral)
        """
        if X.shape[0] == 0:
            return np.array([], dtype=np.int8)
        return np.ones(X.shape[0], dtype=np.int8)
