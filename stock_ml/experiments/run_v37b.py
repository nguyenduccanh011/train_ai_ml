"""V37b — Multi-target ML (buy + sell heads).

Insight: 1 model 3-class hiếm fire sell label vì class imbalance. Tách:
  - head_buy:  current early_wave target (P(target==1))
  - head_sell: binary forward-drawdown >= loss_threshold trong forward_window

Engine giữ V34 logic. Sell signal được overlay vào y_pred (force -1) bởi
run_pipeline._run_backtest_from_cache khi item['y_pred_sell'] tồn tại.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


def backtest_v37b(y_pred, returns, df_test, feature_cols, **kwargs):
    """V37b uses V34 engine; sell-signal overlay handled in run_pipeline."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)
