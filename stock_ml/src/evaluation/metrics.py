"""
Evaluation metrics for stock prediction models.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef, balanced_accuracy_score,
)


def compute_metrics(y_true, y_pred, average: str = "weighted") -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def compute_trading_metrics(
    y_true, y_pred, returns: pd.Series, target_classes: Dict[int, str] = None
) -> Dict[str, float]:
    """
    Compute trading-oriented metrics.
    Assumes: 1=UPTREND (buy), 0=SIDEWAYS (hold), -1=DOWNTREND (sell/avoid)
    """
    target_classes = target_classes or {1: "UPTREND", 0: "SIDEWAYS", -1: "DOWNTREND"}

    pred_up = (np.array(y_pred) == 1)
    pred_down = (np.array(y_pred) == -1)
    actual_up = (np.array(y_true) == 1)

    # When model says BUY, what's the actual return?
    buy_returns = returns[pred_up] if pred_up.any() else pd.Series([0])
    avoid_returns = returns[pred_down] if pred_down.any() else pd.Series([0])

    # Hit rate: when we predict UP, how often is it actually UP?
    hit_rate = actual_up[pred_up].mean() if pred_up.any() else 0

    # Avg return when buying vs avoiding
    avg_buy_return = buy_returns.mean() if len(buy_returns) > 0 else 0
    avg_avoid_return = avoid_returns.mean() if len(avoid_returns) > 0 else 0

    return {
        "hit_rate_buy": float(hit_rate),
        "avg_return_when_buy": float(avg_buy_return),
        "avg_return_when_avoid": float(avg_avoid_return),
        "buy_signal_ratio": float(pred_up.mean()),
        "n_buy_signals": int(pred_up.sum()),
        "n_total": len(y_pred),
    }


def format_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format experiment results into a comparison table."""
    df = pd.DataFrame(results)

    # Sort by primary metric
    sort_col = "f1_macro" if "f1_macro" in df.columns else "f1"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    return df


def print_leaderboard(results_df: pd.DataFrame, top_n: int = 10):
    """Print a formatted leaderboard of model results."""
    display_cols = [
        "model", "feature_set", "window",
        "accuracy", "balanced_accuracy", "f1_macro", "mcc",
        "hit_rate_buy", "avg_return_when_buy",
    ]
    cols = [c for c in display_cols if c in results_df.columns]
    top = results_df.head(top_n)[cols]

    print("\n" + "=" * 80)
    print("🏆 MODEL LEADERBOARD")
    print("=" * 80)
    print(top.to_string(index=False, float_format="%.4f"))
    print("=" * 80)
