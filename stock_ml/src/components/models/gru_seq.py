from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from src.models.sequence import HAS_TORCH, GRUClassifier
except ImportError:
    HAS_TORCH = False
    GRUClassifier = None  # type: ignore[assignment,misc]


class GRUEntryModel:
    """GRU sequence classifier (V37d) wrapped to EntryModel Protocol.

    Wraps the existing GRUClassifier + RobustScaler (matches legacy pipeline).
    Assumes flat 2D input — windows are built internally by GRUClassifier.
    """

    name = "gru"

    def __init__(
        self,
        window: int = 20,
        hidden: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 8,
        batch_size: int = 512,
        lr: float = 1e-3,
        n_classes: int = 3,
        device: str = "auto",
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("torch is required for GRUEntryModel")
        clf = GRUClassifier(
            window=window,
            hidden=hidden,
            n_layers=n_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            n_classes=n_classes,
            device=device,
            random_state=random_state,
            verbose=verbose,
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
