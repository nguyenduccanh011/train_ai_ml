"""V39d — Per-symbol Rule-Exit Hybrid cho 21 ma thua Rule.

Root cause: 21 ma stable-trend (PVS, AAS, BSR, PVD, KBC, GAS, PLX, BCM, SBT, BID...)
  V37a thua Rule do signal exit sai. Rule (MACD+MA20) bat trend 30-60d tot hon ML fw=8d.

V39d = V39e + v39d_rule_exit_symbols cho top-loss symbols:
  - Voi cac ma trong set: signal exit chi khi Close<MA20 OR MACD_hist<=0
  - Giu toan bo risk exits (hard_cap, HAP, stop_loss) binh thuong
  - Build tren V39e (da co V39a+V39b) de co nen vung

Top 12 ma V37a thua Rule nhieu nhat (diff < -25%):
  PVS(-142), AAS(-117), BSR(-110), PVD(-104), KBC(-90), AAV(-85),
  GAS(-79), FRT(-76), BCM(-75), PLX(-70), SBT(-67), BID(-47)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_v37a import backtest_v37a


# Top 12 symbols V37a thua Rule nhieu nhat, uu tien rule-exit logic
V39D_RULE_EXIT_SYMBOLS = {
    "PVS", "AAS", "BSR", "PVD", "KBC", "AAV",
    "GAS", "FRT", "BCM", "PLX", "SBT", "BID",
}

V39D_FLAGS = dict(
    v39a_signal_exit_min_hold=35,
    v39b_hap_trigger=0.08,
    v39b_hap_min_hold=15,
    v39d_rule_exit_symbols=V39D_RULE_EXIT_SYMBOLS,
)


def backtest_v39d(y_pred, returns, df_test, feature_cols, **kwargs):
    merged = {**V39D_FLAGS, **kwargs}
    for k, v in V39D_FLAGS.items():
        merged[k] = v
    return backtest_v37a(y_pred, returns, df_test, feature_cols, **merged)
