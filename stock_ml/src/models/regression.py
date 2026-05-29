"""Regression models for forward-return prediction.

These models predict float returns (not binary classes).
Used with ForwardReturnRegressionTarget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Self


class RegressionModelProtocol:
    """Regression model interface — predicts float returns ∈ ℝ."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train on float targets."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict float returns."""
        ...


@dataclass
class LGBMRegressionModel:
    """LightGBM regression model for forward returns."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._reg = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LGBMRegressionModel:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise RuntimeError("lightgbm not installed: pip install lightgbm") from e

        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        merged = {**defaults, **self.params}
        self._reg = LGBMRegressor(**merged, random_state=0)
        self._reg.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._reg is None:
            raise RuntimeError("model not fitted")
        return self._reg.predict(X).astype(np.float32)


@dataclass
class XGBRegressionModel:
    """XGBoost regression model for forward returns."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._reg = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBRegressionModel:
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise RuntimeError("xgboost not installed: pip install xgboost") from e

        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        merged = {**defaults, **self.params}
        self._reg = XGBRegressor(**merged, random_state=0, verbosity=0)
        self._reg.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._reg is None:
            raise RuntimeError("model not fitted")
        return self._reg.predict(X).astype(np.float32)


@dataclass
class RandomForestRegressionModel:
    """Random Forest regression model for forward returns."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._reg = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressionModel:
        from sklearn.ensemble import RandomForestRegressor

        defaults = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        }
        merged = {**defaults, **self.params}
        self._reg = RandomForestRegressor(**merged, random_state=0, n_jobs=-1)
        self._reg.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._reg is None:
            raise RuntimeError("model not fitted")
        return self._reg.predict(X).astype(np.float32)
