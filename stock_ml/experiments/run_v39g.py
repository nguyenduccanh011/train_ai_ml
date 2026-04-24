"""V39g — Selective rule confirm: chi defer signal exit khi max_profit cao.

Insight tu phan tich V39a2 better/worse cases:
  - Better cases: avg hold tang 75d, ma co max_profit cao (>150%)
  - Worse cases: hold them 5-10d, ma co max_profit trung binh (100-155%)

-> Chi defer signal exit khi max_profit >= 150% (trade da co big upside tiềm năng)
   Voi max_profit nho (<150%), signal exit co the dung (giam risk, tranh keep losing)

V39g = V37a + rule_confirm_exit chi khi max_profit_pct >= 150%
  - Tranh giu lai cac trade stall (<5% max_profit sau 21 ngay)
  - Chi giu lai cac trade da chung to tiem nang lon (max_profit >= 150%)
  - Ket hop V39b HAP reform de tranh cat dau song
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


V39G_FLAGS = dict(
    v39b_hap_trigger=0.08,
    v39b_hap_min_hold=15,
    v39a_rule_confirm_exit=True,
    v39g_rule_confirm_min_maxprofit=1.50,  # chi defer khi max_profit >= 150%
)


def backtest_v39g(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39G_FLAGS, **kwargs}
    for k, v in V39G_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
