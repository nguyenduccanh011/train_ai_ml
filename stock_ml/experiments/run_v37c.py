"""V37c — Per-symbol probability threshold tuning.

Insight: argmax 3-class collapses signals — bank stocks (low vol) hiếm khi
có P(buy) > P(neutral) → ML không fire. V37c hạ threshold cho slow movers,
nâng cho momentum (chống fomo).

V37c = V34 engine + per-profile proba threshold (override argmax).
Thresholds đọc từ models.yaml::v37c.proba_thresholds.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


def backtest_v37c(y_pred, returns, df_test, feature_cols, **kwargs):
    """V37c uses V34 engine; threshold transformation happens in run_pipeline._apply_proba_thresholds."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)
