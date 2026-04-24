"""V38e - combine V38b2 (HAP nhay + stall strict) + V38d2 co-pilot (tat anti-fomo filter).

V38d2 anti-fomo filter lam giam total pnl, chi giu co-pilot exit.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38E_FLAGS = dict(
    # V38b2 flags
    v32_hap_pre_trigger=0.04,
    v32_hap_pre_floor=-0.03,
    v38b_stall_exit=True,
    v38b_stall_min_hold=10,
    v38b_stall_max_profit=0.015,
    v38b_stall_pnl_thresh=-0.025,
    # V38d co-pilot exit only (no fomo filter)
    v38d_fomo_filter=False,
    v38d_copilot_exit=True,
    v38d_copilot_min_hold=6,
    v38d_copilot_min_profit=0.05,
)


def backtest_v38e(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38E_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
