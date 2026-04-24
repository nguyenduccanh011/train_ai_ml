"""V37d — GRU sequence model replaces LightGBM.

Engine identical to V34 — breakthrough attempt comes from RNN capturing HA
streak dynamics (3-5 consecutive candles, body/shadow evolution) that
tabular GBT treats as independent bars.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


def backtest_v37d(y_pred, returns, df_test, feature_cols, **kwargs):
    """V37d uses V34 engine; ML swap happens in run_pipeline._build_predictions."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)
