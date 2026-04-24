"""V38d2 - Anti-fomo nhe hon + co-pilot strict hon.

Thay doi tu V38d:
  - fomo threshold 0.08 (vs 0.06) — chi block FOMO that su nang
  - copilot min_hold 6 (vs 4), min_profit 0.05 (vs 0.03), keep ratio 0.7 (vs 0.6)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38D2_FLAGS = dict(
    v38d_fomo_filter=True,
    v38d_fomo_ret5d_thresh=0.08,
    v38d_fomo_dist_thresh=0.08,
    v38d_copilot_exit=True,
    v38d_copilot_min_hold=6,
    v38d_copilot_min_profit=0.05,
)


def backtest_v38d2(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38D2_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
