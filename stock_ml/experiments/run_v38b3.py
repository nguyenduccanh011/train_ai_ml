"""V38b3 - V38b2 nhung stall-exit chi cho balanced profile (nhom du lieu thua).

Balanced profile chiem 16/21 symbol thua.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32
from src.backtest.defaults import SYMBOL_PROFILES


V38B3_BASE_FLAGS = dict(
    v32_hap_pre_trigger=0.04,
    v32_hap_pre_floor=-0.03,
)
V38B3_BALANCED_ONLY = dict(
    v38b_stall_exit=True,
    v38b_stall_min_hold=10,
    v38b_stall_max_profit=0.015,
    v38b_stall_pnl_thresh=-0.025,
)
V38B3_TARGET_PROFILES = {"balanced"}


def backtest_v38b3(y_pred, returns, df_test, feature_cols, **kwargs):
    sym = "?"
    if "symbol" in df_test.columns and len(df_test) > 0:
        sym = str(df_test["symbol"].iloc[0])
    profile = SYMBOL_PROFILES.get(sym, "balanced")

    merged = {**kwargs, **V38B3_BASE_FLAGS}
    if profile in V38B3_TARGET_PROFILES:
        merged.update(V38B3_BALANCED_ONLY)
    else:
        # Force off cho profile khac
        merged["v38b_stall_exit"] = False
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
