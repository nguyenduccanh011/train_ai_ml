"""V38d - Anti-fomo entry filter + Rule co-pilot exit.

Insight tu phan tich V37a:
  - 58% trades thua nang vao luc entry_ret_5d>5% hoac dist_sma20>5% (fomo entry)
  - 77% trades thua nang co rule thoat truoc voi gia tot hon
  - Rule signal (close>MA20 + MACD>0 + close>open) la nhip co-pilot tu nhien

V38d changes:
  - Block entry neu entry_ret_5d > 6% hoac dist_sma20 > 6% (tru breakout/vshape)
  - Rule co-pilot exit: hold>=4 + co profit>=3% + (close<MA20 hoac MACD<=0) + cur_ret<60%peak -> exit
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38D_FLAGS = dict(
    v38d_fomo_filter=True,
    v38d_fomo_ret5d_thresh=0.06,
    v38d_fomo_dist_thresh=0.06,
    v38d_copilot_exit=True,
    v38d_copilot_min_hold=4,
    v38d_copilot_min_profit=0.03,
)


def backtest_v38d(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38D_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
