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
    signal_mode: str = "entry_first",
    direction: str = "long",
    entry_threshold: float | None = None,
    exit_threshold: float | None = None,
) -> pd.DataFrame:
    """Generate signals from model predictions.

    Unified signal generation for regression and classification approaches with direction support.

    **Regression mode (RECOMMENDED):**
    - Input: predicted returns (float array)
    - Signal rule (hysteresis band):
      * if score > entry_threshold → buy (1)
      * if score < exit_threshold  → sell/exit (-1)
      * else → hold (0)
    - When entry_threshold/exit_threshold are not given they default to
      (+signal_threshold, -signal_threshold), reproducing the legacy symmetric rule.
      Setting exit_threshold < entry_threshold creates a neutral dead-band (hysteresis)
      that the downstream position state machine uses to avoid whipsaw.
    - Score: predicted return (continuous alpha, kept verbatim)

    **Classification mode (Legacy):**
    - Input: entry_model predictions, optional exit_model
    - Entry signal if entry_pred == 1, exit signal if exit_pred == 1
    - Score: entry_proba or predict_proba[:, 1]

    **Direction:**
    - long (default): signals unchanged (1=buy, -1=sell)
    - short: signals flipped (1=short entry, -1=cover exit)

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
        signal_mode: DEPRECATED (no-op in regression; only affects the legacy
                    classification branch). Hysteresis thresholds replace it.
        direction: long (default) | short (flips signals)
        entry_threshold: score above which a long signal (+1) is emitted.
                        Defaults to +signal_threshold.
        exit_threshold: score below which an exit/short signal (-1) is emitted.
                        Defaults to -signal_threshold.

    Returns:
        DataFrame with columns [symbol, date, signal, score]
        where signal ∈ {-1, 0, 1} and score is float
    """
    test_use = test_df.copy()

    if is_regression:
        hi = entry_threshold if entry_threshold is not None else signal_threshold
        lo = exit_threshold if exit_threshold is not None else -signal_threshold
        signals = np.where(
            predictions > hi,
            1,
            np.where(predictions < lo, -1, 0),
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
            entry_is_1 = entry_pred[idx] == 1
            exit_is_1 = exit_model is not None and exit_model.predict(X_test[idx : idx + 1])[0] == 1

            if signal_mode == "entry_first":
                if entry_is_1:
                    sig = 1
                elif exit_is_1:
                    sig = -1
            elif signal_mode == "exit_first":
                if exit_is_1:
                    sig = -1
                elif entry_is_1:
                    sig = 1
            elif signal_mode == "independent":
                if exit_is_1:
                    sig = -1
                elif entry_is_1:
                    sig = 1

            signals.append(sig)

        signals = np.array(signals, dtype=np.int8)
        scores = entry_proba

    # Apply direction flip: short direction inverts signals
    if direction == "short":
        signals = signals * -1

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
    entry_rules: dict | None = None,
    exit_rules: dict | None = None,
    score_feature: str = "macd_line",
) -> pd.DataFrame:
    """Generate signals from technical indicator rules (rule-based, no ML).

    Supports configurable entry/exit rules from YAML config.

    Args:
        test_df: DataFrame with feature columns (symbol, date, OHLCV, features)
        entry_rules: dict mapping feature -> operator (e.g., {"macd_hist": ">0", "sma_20_ratio": ">0"})
        exit_rules: dict mapping feature -> operator (e.g., {"macd_hist": "<0", "sma_20_ratio": "<0"})
        score_feature: feature to use for score column (default: macd_line)

    Returns:
        DataFrame with columns [symbol, date, signal, score]
    """
    test_use = test_df.copy()

    # Default rules if not provided (backward compat: v1 hardcoded behavior)
    if entry_rules is None:
        entry_rules = {
            "macd_line": ">0",
            "sma_20_ratio": ">0",
            "close_to_open": ">0",
        }
    if exit_rules is None:
        exit_rules = {
            "macd_line": "<0",
            "sma_20_ratio": "<0",
            "close_to_open": "<0",
        }

    # Parse rules and check columns exist
    all_features = set(entry_rules.keys()) | set(exit_rules.keys())
    missing = [f for f in all_features if f not in test_use.columns]
    if missing:
        raise ValueError(f"Missing required features for technical rules: {missing}")

    # Evaluate entry signal: AND of all entry conditions
    buy_signal = np.ones(len(test_use), dtype=bool)
    for feature, condition in entry_rules.items():
        buy_signal &= _evaluate_condition(test_use[feature], condition)

    # Evaluate exit signal: AND of all exit conditions
    sell_signal = np.ones(len(test_use), dtype=bool)
    for feature, condition in exit_rules.items():
        sell_signal &= _evaluate_condition(test_use[feature], condition)

    signals = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
    scores = (
        test_use[score_feature].astype(np.float32)
        if score_feature in test_use.columns
        else np.zeros(len(test_use), dtype=np.float32)
    )

    test_use["signal"] = signals
    test_use["score"] = scores
    return test_use[["symbol", "date", "signal", "score"]]


def generate_slot_signals(
    X_test: pd.DataFrame,
    ml_predictions: np.ndarray | None = None,
    rule_conditions: list[dict] | None = None,
    rule_logic: str = "AND",
    signal_threshold: float = 0.0,
    direction: str = "long",
    entry_threshold: float | None = None,
    exit_threshold: float | None = None,
) -> np.ndarray:
    """Generate unified signals for a slot (entry/exit/regime/size).

    Dual-component signal generation: ML and Rule work together based on config.
    Mode is INFERRED from null/non-null inputs (no explicit mode parameter).

    **Inference logic:**
    - ml_predictions is not None, rule_conditions is None → ml_only
    - ml_predictions is None, rule_conditions is not None → rule_only
    - both not None → hybrid (rule filters ML via AND)
    - both None → return zeros (no signal generation)

    **Hybrid mode semantics:**
    - Generate ML signals from predictions using threshold
    - Evaluate rule conditions → boolean mask per row
    - Final signal = ML signal if rule passes, else 0 (veto)
    - This gives "stricter" entry (both must pass) vs loose (either passes)

    Args:
        X_test: DataFrame with features for rule evaluation
        ml_predictions: shape (n,) array of predicted returns (regression)
                       None if no ML component
        rule_conditions: list[dict] for rule evaluation, each dict is {feature: operator}
                        None if no rule component
        rule_logic: "AND" (all conditions must pass) or "OR" (any condition passes)
        signal_threshold: threshold for ML regression signals
        direction: "long" (unchanged) or "short" (flips signals)
        entry_threshold: score above which a long signal is emitted (default +signal_threshold)
        exit_threshold: score below which an exit/short signal is emitted (default -signal_threshold)

    Returns:
        shape (n,) int8 array with values {-1, 0, 1}
        -1 = sell, 0 = hold, 1 = buy
    """
    n = len(X_test)
    hi = entry_threshold if entry_threshold is not None else signal_threshold
    lo = exit_threshold if exit_threshold is not None else -signal_threshold

    # Mode 1: ML only
    if ml_predictions is not None and rule_conditions is None:
        signals = np.where(
            ml_predictions > hi,
            1,
            np.where(ml_predictions < lo, -1, 0),
        )

    # Mode 2: Rule only
    elif ml_predictions is None and rule_conditions is not None:
        rule_mask = _evaluate_rule_conditions(X_test, rule_conditions, rule_logic)
        signals = np.where(rule_mask, 1, 0).astype(np.int8)

    # Mode 3: Hybrid (ML + Rule)
    elif ml_predictions is not None and rule_conditions is not None:
        ml_signals = np.where(
            ml_predictions > hi,
            1,
            np.where(ml_predictions < lo, -1, 0),
        )
        rule_mask = _evaluate_rule_conditions(X_test, rule_conditions, rule_logic)
        signals = np.where(rule_mask, ml_signals, 0).astype(np.int8)

    # Mode 4: Neither (no signal)
    else:
        signals = np.zeros(n, dtype=np.int8)

    # Apply direction flip: short direction inverts signals
    if direction == "short":
        signals = signals * -1

    return signals.astype(np.int8)


def _evaluate_rule_conditions(
    X_test: pd.DataFrame,
    conditions: list[dict],
    logic: str = "AND",
) -> np.ndarray:
    """Evaluate rule conditions across all rows.

    Args:
        X_test: DataFrame with feature columns
        conditions: list of condition dicts, each {feature: condition_str}
                   e.g., [{"macd_line": ">0"}, {"volume_sma": ">0.8"}]
        logic: "AND" (all must pass) or "OR" (any must pass)

    Returns:
        boolean array (True = conditions pass, False = conditions fail)
    """
    if not conditions:
        return np.ones(len(X_test), dtype=bool)

    masks = []
    for cond_dict in conditions:
        for feature, op_str in cond_dict.items():
            if feature not in X_test.columns:
                raise ValueError(f"Feature '{feature}' not found in X_test")
            mask = _evaluate_condition(X_test[feature], op_str)
            masks.append(mask)

    if logic.upper() == "AND":
        result = np.ones(len(X_test), dtype=bool)
        for mask in masks:
            result &= mask
        return result
    elif logic.upper() == "OR":
        result = np.zeros(len(X_test), dtype=bool)
        for mask in masks:
            result |= mask
        return result
    else:
        raise ValueError(f"Unknown logic: {logic}. Expected 'AND' or 'OR'.")


def _evaluate_condition(values: pd.Series, condition: str) -> np.ndarray:
    """Evaluate a condition string like '>0', '> 0', '<0' against a series.

    Args:
        values: pd.Series of numeric values
        condition: string like '>0', '> 0', '>=1.0', '<0', etc. (spaces optional)

    Returns:
        boolean array
    """
    cond = condition.replace(" ", "").strip()  # remove whitespace
    if cond.startswith(">="):
        threshold = float(cond[2:])
        return values >= threshold
    elif cond.startswith(">"):
        threshold = float(cond[1:])
        return values > threshold
    elif cond.startswith("<="):
        threshold = float(cond[2:])
        return values <= threshold
    elif cond.startswith("<"):
        threshold = float(cond[1:])
        return values < threshold
    elif cond.startswith("=="):
        threshold = float(cond[2:])
        return values == threshold
    elif cond.startswith("!="):
        threshold = float(cond[2:])
        return values != threshold
    else:
        raise ValueError(f"Unknown condition format: {condition}")


def generate_signals_from_features(
    model: Any,
    history_feat: pd.DataFrame,
    symbols: list[str],
    feature_cols: list[str],
    signal_threshold: float = 0.0,
    filter_fn: Callable[[dict[str, int], pd.DataFrame, pd.Timestamp], dict[str, int]] | None = None,
    entry_threshold: float | None = None,
    exit_threshold: float | None = None,
    return_scores: bool = False,
) -> dict[str, int] | tuple[dict[str, int], dict[str, float]]:
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
    hi = entry_threshold if entry_threshold is not None else signal_threshold
    lo = exit_threshold if exit_threshold is not None else -signal_threshold

    if history_feat.empty:
        empty = {sym: 0 for sym in symbols}
        return (empty, {sym: 0.0 for sym in symbols}) if return_scores else empty

    max_date = pd.to_datetime(history_feat["date"]).max()
    eval_date = max_date

    raw_signals = {}
    scores: dict[str, float] = {}
    for sym in symbols:
        sym_feat = history_feat[history_feat["symbol"] == sym]
        if sym_feat.empty:
            raw_signals[sym] = 0
            scores[sym] = 0.0
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
        scores[sym] = pred_return

        if pred_return > hi:
            raw_signals[sym] = 1
        elif pred_return < lo:
            raw_signals[sym] = -1
        else:
            raw_signals[sym] = 0

    filtered = raw_signals
    if filter_fn is not None:
        filtered = filter_fn(raw_signals, history_feat, eval_date)

    return (filtered, scores) if return_scores else filtered
