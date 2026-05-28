"""Model registry — factories for entry/exit models with support for multiple algorithms.

Supported types: lightgbm, xgboost, random_forest, mlp, lstm, rule
Each trains on binary labels {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

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
        }
        merged = {**defaults, **self.params}
        self._clf = LGBMClassifier(**merged, random_state=0)
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
class LGBMExitModel:
    """LightGBM exit model wrapper."""

    params: dict = None

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
        }
        merged = {**defaults, **self.params}
        self._clf = LGBMClassifier(**merged, random_state=0)
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
        self._clf = XGBClassifier(**merged, random_state=0, verbosity=0)
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
        self._clf = XGBClassifier(**merged, random_state=0, verbosity=0)
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
        self._clf = RandomForestClassifier(**merged, random_state=0, n_jobs=-1)
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
        self._clf = RandomForestClassifier(**merged, random_state=0, n_jobs=-1)
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

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPEntryModel:
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
        self._clf = MLPClassifier(**merged, random_state=0, early_stopping=True)
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
        self._clf = MLPClassifier(**merged, random_state=0, early_stopping=True)
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


@dataclass
class RuleEntryModel:
    """Rule-based entry model — threshold on features."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> RuleEntryModel:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Rule entry model not yet implemented")


@dataclass
class RuleExitModel:
    """Rule-based exit model — threshold on features."""

    params: dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> RuleExitModel:
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Rule exit model not yet implemented")


_ENTRY_REGISTRY: dict[str, type] = {
    "lightgbm": LGBMEntryModel,
    "xgboost": XGBEntryModel,
    "random_forest": RandomForestEntryModel,
    "mlp": MLPEntryModel,
    "lstm": LSTMEntryModel,
    "rule": RuleEntryModel,
}

_EXIT_REGISTRY: dict[str, type] = {
    "lightgbm": LGBMExitModel,
    "xgboost": XGBExitModel,
    "random_forest": RandomForestExitModel,
    "mlp": MLPExitModel,
    "lstm": LSTMExitModel,
    "rule": RuleExitModel,
}


def build_entry_model(model_type: str, params: dict | None = None) -> EntryModelProtocol:
    """Build an entry model by type.

    Args:
        model_type: one of {lightgbm, xgboost, random_forest, mlp, lstm, rule}
        params: dict of model-specific hyperparameters

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
    return model_class(params=params or {})


def build_exit_model(model_type: str, params: dict | None = None) -> ExitModelProtocol:
    """Build an exit model by type.

    Args:
        model_type: one of {lightgbm, xgboost, random_forest, mlp, lstm, rule}
        params: dict of model-specific hyperparameters

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
    return model_class(params=params or {})
