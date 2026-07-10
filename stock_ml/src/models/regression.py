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
    seed: int = 42

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
            # REPRODUCIBILITY: with feature/bagging subsampling, LightGBM's default
            # multi-threaded histogram build has float accumulation-order non-determinism,
            # so the SAME config+seed scored ~±5 composite pts run-to-run (the leaderboard's
            # whole recent climb sat inside that band — project_t1837_AB_refuted). deterministic
            # + force_col_wise give stable results across num_threads (LightGBM docs require one
            # of force_*_wise when deterministic). User params can still override.
            "deterministic": True,
            "force_col_wise": True,
        }
        # Remove None values from params to avoid "multiple values for keyword" errors
        clean_params = {k: v for k, v in self.params.items() if v is not None}
        merged = {**defaults, **clean_params}
        self._reg = LGBMRegressor(**merged, random_state=self.seed)
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
    seed: int = 42

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
        # Remove None values from params to avoid "multiple values for keyword" errors
        clean_params = {k: v for k, v in self.params.items() if v is not None}
        merged = {**defaults, **clean_params}
        self._reg = XGBRegressor(**merged, random_state=self.seed, verbosity=0)
        self._reg.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._reg is None:
            raise RuntimeError("model not fitted")
        return self._reg.predict(X).astype(np.float32)


@dataclass
class MLPRegressionModel:
    """Multi-layer perceptron (sklearn) regression for forward returns — a DIFFERENT-character
    learner than the tree family (lgbm/xgb/rf). NN needs feature scaling, so the regressor is
    wrapped in a StandardScaler pipeline. Tree hyperparams (num_leaves/min_data_in_leaf/...) and
    monotone constraints have no MLP analog and are ignored (only MLP-native keys are honored).
    Deterministic via random_state=seed (the early-stopping validation split is seeded)."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._reg = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPRegressionModel:
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        defaults = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "alpha": 1e-3,            # L2 regularization
            "learning_rate_init": 1e-3,
            "max_iter": 300,
            "early_stopping": True,
            "n_iter_no_change": 15,
            "validation_fraction": 0.15,
        }
        mlp_keys = {
            "hidden_layer_sizes", "activation", "alpha", "learning_rate_init", "max_iter",
            "early_stopping", "n_iter_no_change", "validation_fraction", "batch_size",
            "solver", "learning_rate", "momentum", "beta_1", "beta_2",
        }
        clean = {k: v for k, v in self.params.items() if v is not None and k in mlp_keys}
        merged = {**defaults, **clean}
        if isinstance(merged.get("hidden_layer_sizes"), list):
            merged["hidden_layer_sizes"] = tuple(merged["hidden_layer_sizes"])
        mlp = MLPRegressor(random_state=self.seed, **merged)
        self._reg = make_pipeline(StandardScaler(), mlp)
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
    seed: int = 42

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
        # Remove None values from params to avoid "multiple values for keyword" errors
        clean_params = {k: v for k, v in self.params.items() if v is not None}
        merged = {**defaults, **clean_params}
        self._reg = RandomForestRegressor(**merged, random_state=self.seed, n_jobs=-1)
        self._reg.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._reg is None:
            raise RuntimeError("model not fitted")
        return self._reg.predict(X).astype(np.float32)
