"""V38b - HAP nhay hon + stall-exit.

Insight tu phan tich V37a:
  - HAP exits hien tai: WR=0%, mean -8.94% (87 trades) -> trigger 5%/-5% qua tre
  - 60/203 trade thua >=8% co exit_reason='v32_hap_preempt'
  - Trades hold >= 10 ngay nhung max_profit < 2% thuong ket thuc -3% den -6%

V38b changes (vs V34/V37a):
  - HAP trigger 0.05 -> 0.03, floor -0.05 -> -0.02 (nhay gap 1.5x)
  - Them stall_exit: hold>=6 ngay + max_profit<2% + cur_ret<-2% -> exit
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38B_FLAGS = dict(
    # HAP nhay hon
    v32_hap_pre_trigger=0.03,
    v32_hap_pre_floor=-0.02,
    # Stall exit
    v38b_stall_exit=True,
    v38b_stall_min_hold=6,
    v38b_stall_max_profit=0.02,
    v38b_stall_pnl_thresh=-0.02,
)


def backtest_v38b(y_pred, returns, df_test, feature_cols, **kwargs):
    # V38B_FLAGS phai ghi de kwargs (vi pipeline pass yaml params qua kwargs)
    merged = {**kwargs, **V38B_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
