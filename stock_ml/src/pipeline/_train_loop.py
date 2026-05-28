"""Shared walk-forward train+predict loop.

Used by both:
- src.pipeline.trainer.build_prediction_cache (new ExperimentConfig path)
- src.pipeline.build_predictions._build_predictions (legacy run_pipeline.py path)

Keeping the loop in one place prevents drift between the two pipelines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.signal_adapter import canonicalize_predictions


def _finite_matrix(values: Any) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def train_predict_walk_forward(
    *,
    df: pd.DataFrame,
    splitter: Any,
    symbols: list[str],
    feature_cols: list[str],
    target_cfg: dict[str, Any],
    entry_model_factory,
    exit_model_factory=None,
    has_exit: bool = False,
) -> list[dict[str, Any]]:
    """Train one entry model (and optional exit model) per fold and emit per-symbol predictions.

    Args:
        df: Pooled DataFrame with target/feature columns and a 'symbol' column.
        splitter: WalkForwardSplitter providing .split(df) → (window, train_df, test_df) tuples.
        symbols: Whitelist of symbols to emit predictions for.
        feature_cols: Feature column names to feed the model.
        target_cfg: Target config dict — passed to canonicalize_predictions.
        entry_model_factory: Zero-arg callable returning a fresh entry model per fold.
        exit_model_factory: Zero-arg callable returning a fresh exit model per fold (or None).
        has_exit: True if df has 'target_sell' and exit_model_factory should be used.

    Returns:
        list of dicts with keys: symbol, y_pred, y_pred_exit, y_proba, classes, returns,
        sym_test_df, feature_cols.
    """
    results: list[dict[str, Any]] = []

    for _window, train_df, test_df in splitter.split(df):
        train_label_cols = ["target"] + (["target_sell"] if has_exit else [])
        train_fit_df = train_df.dropna(subset=train_label_cols)
        if train_fit_df.empty:
            continue
        model = entry_model_factory()
        X_train = _finite_matrix(train_fit_df[feature_cols].values)
        y_train = train_fit_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        sell_model = None
        if has_exit and exit_model_factory is not None:
            sell_model = exit_model_factory()
            sell_model.fit(X_train, train_fit_df["target_sell"].values.astype(int))

        for sym in test_df["symbol"].unique():
            if sym not in symbols:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = _finite_matrix(sym_test[feature_cols].values)
            y_pred_raw = model.predict(X_sym)
            y_pred = canonicalize_predictions(y_pred_raw, target_cfg)
            rets = sym_test["return_1d"].values

            y_proba = None
            classes = None
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_sym)
                    final_est = model.steps[-1][1] if hasattr(model, "steps") else model
                    classes = list(final_est.classes_)
            except Exception:
                y_proba = None

            results.append(
                {
                    "symbol": sym,
                    "y_pred": y_pred,
                    "y_pred_exit": (
                        sell_model.predict(X_sym).astype(int) if sell_model is not None else None
                    ),
                    "y_proba": y_proba,
                    "classes": classes,
                    "returns": rets,
                    "sym_test_df": sym_test,
                    "feature_cols": feature_cols,
                }
            )

    return results
