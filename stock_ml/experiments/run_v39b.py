"""V39b — HAP Reform: raise trigger 4%->8% + min_hold=15.

Root cause: HAP preempt n=87, 100% thua, avg=-8.94%, avg max_profit=190%.
  HAP dang cat "dau song": entry dung, gia tang nhe 4%, HAP pre-trigger,
  roi giam 3% ve floor -> exit. Sau do co phieu bung len +190% trung binh.

V39b = V37a engine + HAP trigger 8% + HAP chi active sau 15 ngay hold.
  - v39b_hap_trigger=0.08: can tang len 8% moi trigger HAP
  - v39b_hap_min_hold=15: HAP khong active trong 15 ngay dau (tranh cat song moi bat dau)
  - Giu v32_hap_pre_floor=-0.07 de van co bao ve khi da co big profit
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


V39B_FLAGS = dict(
    v39b_hap_trigger=0.08,
    v39b_hap_min_hold=15,
)


def backtest_v39b(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39B_FLAGS, **kwargs}
    for k, v in V39B_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
