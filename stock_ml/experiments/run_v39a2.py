"""V39a2 — Signal Exit Defer + Rule Confirm (khong min_hold cung).

Thay vi block signal exit theo so ngay (V39a), V39a2 require rule confirm:
  signal exit chi kich hoat khi CA HAI dieu kien rule bearish:
    - MACD_hist < 0
    - Close < MA20

Khi chi co ML signal ma rule van bullish -> defer exit.
Van su dung V37a base (per-profile dispatch).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


V39A2_FLAGS = dict(
    v39a_rule_confirm_exit=True,
)


def backtest_v39a2(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39A2_FLAGS, **kwargs}
    for k, v in V39A2_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
