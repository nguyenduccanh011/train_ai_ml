"""V38b2 - V38b version nhe hon (less aggressive).

Thay doi tu V38b:
  - HAP trigger 0.04 (vs 0.03), floor -0.03 (vs -0.02) — bot nhay 1 nac
  - Stall-exit: hold>=10 (vs 6), max_profit<0.015 (vs 0.02), pnl<-0.025 (vs -0.02)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38B2_FLAGS = dict(
    v32_hap_pre_trigger=0.04,
    v32_hap_pre_floor=-0.03,
    v38b_stall_exit=True,
    v38b_stall_min_hold=10,
    v38b_stall_max_profit=0.015,
    v38b_stall_pnl_thresh=-0.025,
)


def backtest_v38b2(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38B2_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
