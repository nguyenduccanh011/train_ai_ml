"""V39e — Combination: V39a + V39b (signal exit min hold + HAP reform).

Giai quyet dong thoi 2 van de lon nhat cua V37a:
  1. Signal exit qua som o 21-30d (bucket WR=12%) -> V39a min_hold=35
  2. HAP preempt cat dau song (87 trades 100% loss) -> V39b hap trigger 8% + min_hold=15

Ky vong: dot pha vi giai quyet ca entry hold qua ngan va HAP pre-empt dau song.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


V39E_FLAGS = dict(
    v39a_signal_exit_min_hold=35,
    v39b_hap_trigger=0.08,
    v39b_hap_min_hold=15,
)


def backtest_v39e(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39E_FLAGS, **kwargs}
    for k, v in V39E_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
