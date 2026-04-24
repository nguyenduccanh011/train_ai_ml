"""V38 combos - thu cac to hop V38b/c/d."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32
from experiments.run_v38b import V38B_FLAGS
from experiments.run_v38c import V38C_FLAGS
from experiments.run_v38d import V38D_FLAGS


def backtest_v38bc(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38B_FLAGS, **V38C_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v38bd(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38B_FLAGS, **V38D_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v38cd(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38C_FLAGS, **V38D_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)


def backtest_v38bcd(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38B_FLAGS, **V38C_FLAGS, **V38D_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
