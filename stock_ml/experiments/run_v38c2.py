"""V38c2 - HA-driven exit chi khi confirm 2 dau hieu hoac hold du dai.

Thay doi tu V38c:
  - min_hold 5 (vs 3) — tranh exit qua som
  - require ca ha_bearish_reversal AND late_wave (cho V38c2 strict version)
  - hoac chi 1 trong 2 nhung pnl <= -2% (cho confirm)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v32_final import backtest_v32


V38C2_FLAGS = dict(
    v38c_ha_exit=True,
    v38c_ha_min_hold=5,
    v38c_ha_pnl_thresh=-0.02,  # exit chi khi loss da >= 2%
)


def backtest_v38c2(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**kwargs, **V38C2_FLAGS}
    return backtest_v32(y_pred, returns, df_test, feature_cols, **merged)
