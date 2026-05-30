"""Unified signal generation — shared by backtest and live_sim.

Ensures consistent signal logic across all code paths (backtest, live_sim, hyperparameter search).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def generate_signals_from_predictions(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    signal_threshold: float = 0.0,
    is_regression: bool = True,
    entry_model: Any = None,
    exit_model: Any = None,
    X_test: np.ndarray | None = None,
) -> pd.DataFrame:
    """Generate signals from model predictions.

    Unified signal generation for regression and classification approaches.

    **Regression mode (RECOMMENDED):**
    - Input: predicted returns (float array)
    - Signal rule:
      * if pred_return > +threshold → buy (1)
      * if pred_return < -threshold → sell (-1)
      * else → hold (0)
    - Score: predicted return

    **Classification mode (Legacy):**
    - Input: entry_model predictions, optional exit_model
    - Entry signal if entry_pred == 1, exit signal if exit_pred == 1
    - Score: entry_proba or predict_proba[:, 1]

    Args:
        predictions: shape (n,) array of predicted values
                    - Regression: predicted returns (float)
                    - Classification: not used (models used directly)
        test_df: DataFrame with [symbol, date, ...] and any feat cols
        signal_threshold: threshold for signal generation (regression only)
        is_regression: if True, use regression mode; else classification
        entry_model: fitted entry model (classification mode only)
        exit_model: fitted exit model (classification mode only)
        X_test: feature matrix for test set (classification mode only)

    Returns:
        DataFrame with columns [symbol, date, signal, score]
        where signal ∈ {-1, 0, 1} and score is float
    """
    test_use = test_df.copy()

    if is_regression:
        signals = np.where(
            predictions > signal_threshold,
            1,
            np.where(predictions < -signal_threshold, -1, 0),
        )
        scores = predictions.astype(np.float32)

    else:
        if entry_model is None or X_test is None:
            raise ValueError("Classification mode requires entry_model and X_test (classification)")

        entry_pred = entry_model.predict(X_test)
        entry_proba = (
            entry_model.predict_proba(X_test)[:, 1]
            if hasattr(entry_model, "predict_proba")
            else np.zeros(len(entry_pred))
        )

        signals = []
        for idx in range(len(entry_pred)):
            sig = 0
            if entry_pred[idx] == 1:
                sig = 1
            elif exit_model is not None:
                exit_pred = exit_model.predict(X_test[idx : idx + 1])
                if exit_pred[0] == 1:
                    sig = -1
            signals.append(sig)

        signals = np.array(signals, dtype=np.int8)
        scores = entry_proba

    test_use["signal"] = signals
    test_use["score"] = scores.astype(np.float32)
    return test_use[["symbol", "date", "signal", "score"]]


def generate_signals_dict(
    models: dict[str, tuple[np.ndarray, float]],
    symbols: list[str],
    signal_threshold: float = 0.0,
) -> dict[str, int]:
    """Generate signal dict {symbol: signal} from pre-computed predictions.

    For live trading: predictions are pre-computed at T-1 from history features.

    Args:
        models: dict mapping symbol → (predicted_return, model_confidence)
        symbols: list of symbols in universe
        signal_threshold: threshold for signal generation

    Returns:
        dict[symbol] → signal ∈ {-1, 0, 1}
    """
    signals = {}
    for sym in symbols:
        if sym not in models:
            signals[sym] = 0
            continue

        pred_return, confidence = models[sym]
        if np.isnan(pred_return) or np.isnan(confidence):
            signals[sym] = 0
            continue

        if pred_return > signal_threshold:
            signals[sym] = 1
        elif pred_return < -signal_threshold:
            signals[sym] = -1
        else:
            signals[sym] = 0

    return signals


def generate_signals_from_technical_rules(
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate signals from technical indicator rules (rule-based, no ML).

    Rules:
    - BUY (1): macd_line > 0 AND sma_20_ratio > 0 AND C > O
    - SELL (-1): macd_line < 0 AND sma_20_ratio < 0 AND C < O
    - HOLD (0): otherwise

    Args:
        test_df: DataFrame with [symbol, date, macd_line, sma_20_ratio, open, close]

    Returns:
        DataFrame with columns [symbol, date, signal, score]
    """
    test_use = test_df.copy()

    required_cols = ["macd_line", "sma_20_ratio", "open", "close"]
    missing = [c for c in required_cols if c not in test_use.columns]
    if missing:
        raise ValueError(f"Missing required columns for technical rules: {missing}")

    buy_signal = (
        (test_use["macd_line"] > 0)
        & (test_use["sma_20_ratio"] > 0)
        & (test_use["close"] > test_use["open"])
    )
    sell_signal = (
        (test_use["macd_line"] < 0)
        & (test_use["sma_20_ratio"] < 0)
        & (test_use["close"] < test_use["open"])
    )

    signals = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    scores = test_use["macd_line"].astype(np.float32)

    test_use["signal"] = signals
    test_use["score"] = scores
    return test_use[["symbol", "date", "signal", "score"]]


def generate_signals_from_features(
    model: Any,
    history_feat: pd.DataFrame,
    symbols: list[str],
    feature_cols: list[str],
    signal_threshold: float = 0.0,
    filter_fn: Callable[[dict[str, int], pd.DataFrame, pd.Timestamp], dict[str, int]] | None = None,
) -> dict[str, int]:
    """Generate signal dict from model predictions on live features (T-1 time).

    For live trading at T-1, generates signals for execution at T.
    - Extracts last row of features per symbol
    - Calls model.predict() on latest features
    - Applies signal threshold (regression logic)
    - Applies optional filters (e.g., min_volume)

    Args:
        model: fitted model with .predict(X) method
        history_feat: DataFrame with [symbol, date, *feature_cols]
                     max(date) must be T-1 (no lookahead)
        symbols: list of symbols in universe
        feature_cols: list of feature column names to use
        signal_threshold: threshold for signal generation (regression)
        filter_fn: optional filter function that takes (signals_dict, history_feat, date)
                  and returns filtered signals_dict

    Returns:
        dict[symbol] → signal ∈ {-1, 0, 1}

    Raises:
        ValueError: if any symbol missing data at T-1
        ValueError: if features have NaN values
    """
    if history_feat.empty:
        return {sym: 0 for sym in symbols}

    max_date = pd.to_datetime(history_feat["date"]).max()
    eval_date = max_date

    raw_signals = {}
    for sym in symbols:
        sym_feat = history_feat[history_feat["symbol"] == sym]
        if sym_feat.empty:
            raw_signals[sym] = 0
            continue

        last_row = sym_feat.iloc[-1]
        last_row_date = pd.to_datetime(last_row["date"]).normalize().date()
        eval_date_norm = eval_date.normalize().date()
        if last_row_date != eval_date_norm:
            raise ValueError(
                f"symbol {sym} missing data at {eval_date_norm}: last row is {last_row['date']}"
            )

        feat_cols_present = [c for c in feature_cols if c in sym_feat.columns]
        if not feat_cols_present:
            raise ValueError(f"symbol {sym} has no feature columns")

        if sym_feat[feat_cols_present].iloc[-1].isna().any():
            raise ValueError(
                f"symbol {sym} has NaN features at {eval_date_norm}: {feat_cols_present}"
            )

        X = sym_feat[feat_cols_present].iloc[-1:].to_numpy(dtype=np.float32)
        pred_return = float(model.predict(X)[0])

        if pred_return > signal_threshold:
            raw_signals[sym] = 1
        elif pred_return < -signal_threshold:
            raw_signals[sym] = -1
        else:
            raw_signals[sym] = 0

    filtered = raw_signals
    if filter_fn is not None:
        filtered = filter_fn(raw_signals, history_feat, eval_date)

    return filtered
