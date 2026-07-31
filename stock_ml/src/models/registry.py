"""Model registry — factories for entry/exit models with support for multiple algorithms.

Supported types: lightgbm, xgboost, random_forest, mlp, lstm, rule
Each trains on binary labels {0, 1}.

For regression (forward return prediction), use regression module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from stock_ml.src.models.regression import (
    LGBMRegressionModel,
    MLPRegressionModel,
    RandomForestRegressionModel,
    XGBRegressionModel,
)
from stock_ml.src.models.rules import RuleModel

if TYPE_CHECKING:
    from typing import Self


class EntryModelProtocol(Protocol):
    """Entry model interface — predicts buy signals {0, 1}."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train on binary labels (1=buy, 0=not buy)."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels {0, 1}."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities [P(class=0), P(class=1)]. Optional."""
        ...


class ExitModelProtocol(Protocol):
    """Exit model interface — predicts sell signals {0, 1}."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Train on binary labels (1=sell, 0=not sell)."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels {0, 1}."""
        ...


@dataclass
class LGBMEntryModel:
    """LightGBM entry model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LGBMEntryModel:
        try:
            from lightgbm import LGBMClassifier
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
            # REPRODUCIBILITY (see regression.py): subsampling + default multi-threaded LightGBM
            # is non-deterministic (same config+seed varied ~±5 composite pts run-to-run). These
            # two flags give stable results across num_threads. User params can override.
            "deterministic": True,
            "force_col_wise": True,
        }
        merged = {**defaults, **self.params}
        class_weight = merged.pop("class_weight", None)
        self._clf = LGBMClassifier(**merged, random_state=self.seed)
        sample_weight = None
        if class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            freq = dict(zip(classes, counts))
            total = len(y)
            n_cls = len(classes)
            w = {c: total / (n_cls * cnt) for c, cnt in freq.items()}
            sample_weight = np.array([w[yi] for yi in y])
        elif isinstance(class_weight, dict):
            sample_weight = np.array([class_weight.get(yi, 1.0) for yi in y])
        self._clf.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict_proba(X)


@dataclass
class LGBMExitModel:
    """LightGBM exit model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LGBMExitModel:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise RuntimeError("lightgbm not installed") from e
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            # REPRODUCIBILITY (see regression.py): subsampling + default multi-threaded LightGBM
            # is non-deterministic (same config+seed varied ~±5 composite pts run-to-run). These
            # two flags give stable results across num_threads. User params can override.
            "deterministic": True,
            "force_col_wise": True,
        }
        merged = {**defaults, **self.params}
        self._clf = LGBMClassifier(**merged, random_state=self.seed)
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)


@dataclass
class XGBEntryModel:
    """XGBoost entry model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBEntryModel:
        try:
            from xgboost import XGBClassifier
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
        self._clf = XGBClassifier(**merged, random_state=self.seed, verbosity=0)
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict_proba(X)


@dataclass
class XGBExitModel:
    """XGBoost exit model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBExitModel:
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise RuntimeError("xgboost not installed") from e
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        merged = {**defaults, **self.params}
        self._clf = XGBClassifier(**merged, random_state=self.seed, verbosity=0)
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)


@dataclass
class RandomForestEntryModel:
    """Random Forest entry model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestEntryModel:
        from sklearn.ensemble import RandomForestClassifier

        defaults = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        }
        merged = {**defaults, **self.params}
        self._clf = RandomForestClassifier(**merged, random_state=self.seed, n_jobs=-1)
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict_proba(X)


@dataclass
class RandomForestExitModel:
    """Random Forest exit model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestExitModel:
        from sklearn.ensemble import RandomForestClassifier

        defaults = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
        }
        merged = {**defaults, **self.params}
        self._clf = RandomForestClassifier(**merged, random_state=self.seed, n_jobs=-1)
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)


@dataclass
class MLPEntryModel:
    """MLP (sklearn) entry model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPEntryModel:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        defaults = {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 1000,
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
        }
        merged = {**defaults, **self.params}
        # Drop params MLPClassifier doesn't accept (e.g. class_weight, injected by callers for
        # tree models; MLP handles imbalance via the data/early_stopping, not class_weight).
        for k in (
            "class_weight",
            "num_leaves",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "bagging_freq",
            "n_estimators",
            "deterministic",
            "force_col_wise",
            "lambda_l1",
            "lambda_l2",
            "learning_rate",
            "verbose",
        ):
            merged.pop(k, None)
        # MLP needs standardized inputs (features span ranks 0-1, returns, ratios); without a
        # scaler it barely converges. Pipeline exposes classes_/predict_proba (delegated).
        self._clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(**merged, random_state=self.seed, early_stopping=True),
        )
        # NOTE: do NOT balance classes for the MLP — its UNBALANCED ultra-selectivity (entering
        # only the clearest setups) is the feature, giving pf 8-13 / mdd ~0.05; sample_weight
        # balancing collapsed it to pf 2.0 / mdd 0.3 (v31). Train on the natural distribution.
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict_proba(X)


@dataclass
class MLPExitModel:
    """MLP (sklearn) exit model wrapper."""

    params: dict = None
    seed: int = 42

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPExitModel:
        from sklearn.neural_network import MLPClassifier

        defaults = {
            "hidden_layer_sizes": (100, 50),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 1000,
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
        }
        merged = {**defaults, **self.params}
        self._clf = MLPClassifier(**merged, random_state=self.seed, early_stopping=True)
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("model not fitted")
        return self._clf.predict(X).astype(np.int8)


@dataclass
class LSTMEntryModel:
    """LSTM entry model wrapper (keras/tensorflow)."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSTMEntryModel:
        try:
            import keras.layers  # noqa: F401
            import keras.models  # noqa: F401
        except ImportError as e:
            raise RuntimeError("LSTM requires keras/tensorflow: pip install tensorflow") from e
        raise NotImplementedError("LSTM entry model not yet implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LSTM not yet implemented")


@dataclass
class LSTMExitModel:
    """LSTM exit model wrapper (keras/tensorflow)."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LSTMExitModel:
        raise NotImplementedError("LSTM exit model not yet implemented")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("LSTM not yet implemented")


_ENTRY_REGISTRY: dict[str, type] = {
    "lightgbm": LGBMEntryModel,
    "xgboost": XGBEntryModel,
    "random_forest": RandomForestEntryModel,
    "mlp": MLPEntryModel,
    "lstm": LSTMEntryModel,
    "rule": RuleModel,
}

_EXIT_REGISTRY: dict[str, type] = {
    "lightgbm": LGBMExitModel,
    "xgboost": XGBExitModel,
    "random_forest": RandomForestExitModel,
    "mlp": MLPExitModel,
    "lstm": LSTMExitModel,
    "rule": RuleModel,
}


def build_entry_model(
    model_type: str, params: dict | None = None, seed: int = 42
) -> EntryModelProtocol:
    """Build an entry model by type.

    Args:
        model_type: one of {lightgbm, xgboost, random_forest, mlp, lstm, rule}
        params: dict of model-specific hyperparameters
        seed: random seed for reproducibility

    Returns:
        Fitted or unfitted model instance implementing EntryModelProtocol

    Raises:
        KeyError: if model_type not in registry
    """
    if model_type not in _ENTRY_REGISTRY:
        raise KeyError(
            f"Unknown entry model type: {model_type}. Available: {sorted(_ENTRY_REGISTRY.keys())}"
        )
    model_class = _ENTRY_REGISTRY[model_type]
    return model_class(params=params or {}, seed=seed)


def build_exit_model(
    model_type: str, params: dict | None = None, seed: int = 42
) -> ExitModelProtocol:
    """Build an exit model by type.

    Args:
        model_type: one of {lightgbm, xgboost, random_forest, mlp, lstm, rule}
        params: dict of model-specific hyperparameters
        seed: random seed for reproducibility

    Returns:
        Fitted or unfitted model instance implementing ExitModelProtocol

    Raises:
        KeyError: if model_type not in registry
    """
    if model_type not in _EXIT_REGISTRY:
        raise KeyError(
            f"Unknown exit model type: {model_type}. Available: {sorted(_EXIT_REGISTRY.keys())}"
        )
    model_class = _EXIT_REGISTRY[model_type]
    return model_class(params=params or {}, seed=seed)


_REGRESSION_REGISTRY: dict[str, type] = {
    "lightgbm": LGBMRegressionModel,
    "xgboost": XGBRegressionModel,
    "random_forest": RandomForestRegressionModel,
    "mlp": MLPRegressionModel,
}


def build_regression_model(model_type: str, params: dict | None = None, seed: int = 42):
    """Build a regression model for forward-return prediction.

    Args:
        model_type: one of {lightgbm, xgboost, random_forest}
        params: dict of model-specific hyperparameters
        seed: random seed for reproducibility

    Returns:
        Regression model instance

    Raises:
        KeyError: if model_type not in registry
    """
    if model_type not in _REGRESSION_REGISTRY:
        raise KeyError(
            f"Unknown regression model type: {model_type}. Available: {sorted(_REGRESSION_REGISTRY.keys())}"
        )
    model_class = _REGRESSION_REGISTRY[model_type]
    return model_class(params=params or {}, seed=seed)


_GPU_DETECTED = None


def detect_device(device: str = "auto") -> str:
    global _GPU_DETECTED
    if device != "auto":
        return device.lower()
    if _GPU_DETECTED is None:
        try:
            import lightgbm as lgb

            _GPU_DETECTED = lgb.basic.device_type() == "gpu"
        except Exception:
            _GPU_DETECTED = False
    return "gpu" if _GPU_DETECTED else "cpu"
