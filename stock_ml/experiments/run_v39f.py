"""V39f — V39b + V39a2: HAP reform + rule confirm signal exit.

Combo cua 2 variant tot nhat:
  V39b: HAP trigger 8% + min_hold=15 (giam HAP 87->32, save 460% loss)
  V39a2: signal exit chi khi MACD<0 AND Close<MA20 (reduce false exits, Sharpe=0.272)

Ky vong: vuot V37a (score 420.7) khi giai quyet ca 2 diem yeu:
  1. HAP cat dau song (87 trades, 100% loss, -778%)
  2. Signal exit sai khi rule van bullish
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


V39F_FLAGS = dict(
    v39b_hap_trigger=0.08,
    v39b_hap_min_hold=15,
    v39a_rule_confirm_exit=True,
)


def backtest_v39f(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39F_FLAGS, **kwargs}
    for k, v in V39F_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
