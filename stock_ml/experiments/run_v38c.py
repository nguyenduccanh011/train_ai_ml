"""V38c - HA-driven exit.

Insight tu phan tich V37a:
  - HA features dung tot o entry (leading_v4) nhung ENGINE khong dung HA o exit
  - Exit chu yeu signal (135/203 thua nang) + hap_preempt (60/203)
  - HA bearish_reversal_signal va late_wave + body_shrinking la tin hieu phan phoi som

V38c changes:
  - Them exit rule: hold>=3 + cur_ret<0 + (ha_bearish_reversal | ha_late_wave+body_shrinking) -> exit
  - Bien HA tu entry-only thanh entry+exit
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38C_FLAGS = dict(
    v38c_ha_exit=True,
    v38c_ha_min_hold=3,
    v38c_ha_pnl_thresh=0.0,  # exit khi cur_ret < 0
)


def backtest_v38c(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38C_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
