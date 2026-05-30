"""Rule-based entry/exit models for technical analysis conditions.

Supports AND/OR condition chains on features without ML training.
Example condition set for entry:
  - macd_line > 0
  - sma_20_ratio > 0
  - close > open
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class RuleModel:
    """Rule-based model — evaluates AND/OR condition chains on feature values.

    No training required (fit is a no-op). Conditions are evaluated at predict time
    against a feature matrix, using column name → index mapping.

    Condition format in params:
        {
            "conditions": [
                {"feature": "macd_line", "op": ">", "value": 0},
                {"feature": "sma_20_ratio", "op": ">", "value": 0},
                {"feature": "close_to_open", "op": ">", "value": 0},
            ],
            "logic": "AND",  # "AND" or "OR"
            "score_feature": "macd_line",  # which feature to use as score
            "feature_cols": [...],  # list of feature column names (injected at build time)
        }
    """

    params: dict = None
    seed: int = 42  # unused for rule models, kept for protocol compatibility

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        self._feature_cols = self.params.get("feature_cols", [])
        self._conditions = self.params.get("conditions", [])
        self._logic = self.params.get("logic", "AND").upper()
        self._score_feature = self.params.get("score_feature", "macd_line")

    def fit(self, X: np.ndarray, y: np.ndarray) -> RuleModel:
        """No-op for rule models. Conditions are static, not learned.

        Args:
            X: ignored (not used for rule models)
            y: ignored (not used for rule models)

        Returns:
            self
        """
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate condition chain against feature matrix.

        Args:
            X: feature matrix of shape (n_samples, n_features)

        Returns:
            binary predictions {0, 1} where 1 = conditions met
        """
        if X.shape[0] == 0:
            return np.array([], dtype=np.int8)

        if not self._conditions:
            return np.ones(X.shape[0], dtype=np.int8)

        if not self._feature_cols:
            raise ValueError(
                "RuleModel requires feature_cols mapping. "
                "Ensure params['feature_cols'] is set at build time."
            )

        # Build column index mapping
        col_index = {col: idx for idx, col in enumerate(self._feature_cols)}

        # Evaluate each condition
        condition_results = []
        for cond in self._conditions:
            feature = cond.get("feature")
            op = cond.get("op", ">")
            value = cond.get("value", 0)

            if feature not in col_index:
                raise ValueError(
                    f"Feature '{feature}' not found in feature_cols. "
                    f"Available: {self._feature_cols}"
                )

            col_idx = col_index[feature]
            col_values = X[:, col_idx]

            # Apply operator
            if op == ">":
                result = col_values > value
            elif op == ">=":
                result = col_values >= value
            elif op == "<":
                result = col_values < value
            elif op == "<=":
                result = col_values <= value
            elif op == "==":
                result = col_values == value
            elif op == "!=":
                result = col_values != value
            else:
                raise ValueError(f"Unknown operator: {op}")

            condition_results.append(result)

        # Combine conditions with logic
        if not condition_results:
            return np.ones(X.shape[0], dtype=np.int8)

        if self._logic == "AND":
            combined = np.all(condition_results, axis=0)
        elif self._logic == "OR":
            combined = np.any(condition_results, axis=0)
        else:
            raise ValueError(f"Unknown logic: {self._logic}")

        return combined.astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return pseudo-probability based on score feature.

        For rule models, we return [1-score, score] where score comes from
        a configured feature (e.g., macd_line), normalized to [0, 1].

        Args:
            X: feature matrix of shape (n_samples, n_features)

        Returns:
            probabilities array of shape (n_samples, 2) where column 1 = confidence
        """
        if X.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if not self._feature_cols:
            raise ValueError("RuleModel requires feature_cols for predict_proba")

        col_index = {col: idx for idx, col in enumerate(self._feature_cols)}
        if self._score_feature not in col_index:
            raise ValueError(f"Score feature '{self._score_feature}' not found in feature_cols")

        col_idx = col_index[self._score_feature]
        scores = X[:, col_idx]

        # Normalize to [0, 1] range (simple min-max on batch)
        score_min = scores.min()
        score_max = scores.max()
        if score_max > score_min:
            normalized = (scores - score_min) / (score_max - score_min)
        else:
            normalized = np.ones_like(scores) * 0.5

        proba = np.column_stack([1.0 - normalized, normalized]).astype(np.float32)
        return proba
