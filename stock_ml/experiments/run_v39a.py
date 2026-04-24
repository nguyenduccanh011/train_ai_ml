"""V39a — Signal Exit Min Hold = 35 ngay.

Root cause fix: early_wave fw=8d target tao bias exit o 21-30d (WR=12%, -2671%).
  Signal exit winners: avg hold=52d. Signal exit losers: avg hold=26d.
  -> Block signal exit truoc 35d de cho trade thoi gian du de phat trien.

V39a = V37a engine + v39a_signal_exit_min_hold=35
  - Signal exit chi duoc khoat tru sau 35 ngay hold
  - Tat ca risk exits khac (hard_cap, HAP, stop_loss...) van hoat dong binh thuong
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


V39A_FLAGS = dict(
    v39a_signal_exit_min_hold=35,
)


def backtest_v39a(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39A_FLAGS, **kwargs}
    for k, v in V39A_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
