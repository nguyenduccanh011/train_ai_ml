"""
V34 FINAL — Heikin-Ashi features (leading_v4) + tighter gain target retrain.

Ablation kết quả:
  Phase 1 (leading_v4 vs leading_v3, V33 engine):
    V34A: leading_v4 + V29_target(g5,l4)    comp=433  (+3)
    V34B: leading_v4 + target_shorter(fwd6) comp=438  (+8)
    V34C: leading_v4 + target_tight(g6,l4)  comp=448  (+18) ★
    V34D: V32engine + leading_v4            comp=433  (+3)

  Phase 2 mini-sweep (quanh V34C):
    fwd8_g6_l3                              comp=450  (+20) ★ BEST
    fwd8_g6_l4 (V34C)                       comp=448  (+18)
    fwd8_g7_l4                              comp=440  (+10)
    fwd9_g6_l4                              comp=436  (+6)

  Phase 3 engine patches (on leading_v4 + best_target):
    V32 engine (no RPF)                     comp=450  (+20) — RPF không cần thiết
    V33 engine (với RPF)                    comp=450  (+20) — tương đương
    RPF loose/tight variants                comp=450  (+20) — HA model tự handle

  Decision: V34 = V32 engine + leading_v4 + target(fwd8, g6%, l3%)
  comp=450 (+17 vs V33=433, +20 vs V33_baseline=430)

Why leading_v4 (HA features) works:
  - HA streak/shadow features giúp model phân biệt "đầu sóng" vs "cuối sóng/fomo":
    * ha_green_streak: 1-2 nến đầu vs 5+ nến cuối sóng
    * ha_upper_shadow_ratio/growing: tín hiệu phân phối sớm
    * ha_no_lower_shadow: uptrend chắc, không râu dưới
    * ha_body_shrinking: đà giảm dần → cuối sóng
    * ha_streak_position: 0=đầu, 1=đỉnh sóng
  - 18 HA features thêm, tất cả 0% NaN sau bug fix (index alignment)
  - Tighter gain_threshold (6% vs 5%): label quality cải thiện
    * Model học các trades với signal rõ ràng hơn
    * Loss threshold nới (3%) giúp model không cắt lệnh tốt

Bug fixed trong quá trình:
  - pd.Series(numpy_array) trong grouped DataFrame tạo index mismatch → 68% NaN
  - Fix: pd.Series(array, index=df.index)
  - ha_bullish_reversal_signal 100% zero vì red_streak=0 khi color_switch=1
  - Fix: dùng prev_red_streak = red_streak.shift(1)

V34 = V32 engine (= V31 + hap_preempt(5%,-5%) + weak_oversold_exit(dist=-9%))
    + leading_v4 retrain (leading_v3 + 18 Heikin-Ashi features)
    + target(type=early_wave, fwd=8, short=8, long=20, gain=6%, loss=3%)

Headline vs V33 (comp~433):
  comp         433 → 450   (+17 / +3.9%)
  win rate     49.5% → 49.9%  (+0.4pp)
  avg PnL      +5.90% → +6.22%  (+5.4%)
  total PnL    +7834% → +8099%  (+265%)
  PF           2.69 → 2.81   (+0.12)
  max loss     -57.4% → -57.4%  (unchanged)
  avg hold     34.6d → 35.4d  (+0.8d)
  trades       1327 → 1302  (-25, filter quality cải thiện)

Exit reason breakdown V34:
  end                 n=225   WR=77.8%  avg=+17.36%  (period boundary)
  signal              n=929   WR=49.5%  avg=+4.95%
  v32_hap_preempt     n=113   WR=0.0%   avg=-6.89%   (improved vs V33: -6.93%)
  fast_exit_loss      n=21    WR=0.0%   avg=-6.37%
  peak_protect_dist   n=14    WR=100%   avg=+35.88%

RPF engine patch không cần: leading_v4 model tự học tốt hơn để tránh fomo entries,
RPF thêm vào không thay đổi kết quả (V32 engine = V33 engine cho V34).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET
from experiments.run_v32_final import V32_DELTA, backtest_v32
from experiments.run_v33_final import V33_DELTA, backtest_v33


V34_FEATURE_SET = "leading_v4"

V34_TARGET = dict(
    type="early_wave",
    forward_window=8,
    short_window=8,
    long_window=20,
    gain_threshold=0.06,   # 6% (vs V29/V33: 5%) — signal quality cao hơn
    loss_threshold=0.03,   # 3% (vs V29/V33: 4%) — không cắt lệnh tốt quá sớm
    classes=3,
)

V34_DELTA = dict(
    # V34 không dùng RPF engine patch — leading_v4 model xử lý tốt hơn
    v33_recovery_peak_filter=False,
    v33_signal_confirm_exit=False,
    v33_trailing_ratchet=False,
    v33_trend_rev_exit=False,
    v33_hap_consec_drop=False,
    v33_rsi_oversold_block=False,
)


def backtest_v34(y_pred, returns, df_test, feature_cols, **kwargs):
    """V34 = V32 engine + leading_v4 retrain.
    Engine không thay đổi so với V32; cải thiện đến từ model layer (HA features + target).
    """
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v35a(y_pred, returns, df_test, feature_cols, **kwargs):
    """V35a = V34 engine + early_wave_v2 target (handled at training layer)."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v35b(y_pred, returns, df_test, feature_cols, **kwargs):
    """V35b = V34 engine + V35 entry filter relaxations (flags from yaml)."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v35c(y_pred, returns, df_test, feature_cols, **kwargs):
    """V35c = V34 engine + hybrid rule entry (flags from yaml)."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v36a(y_pred, returns, df_test, feature_cols, **kwargs):
    """V36a = V34 + rule_override only."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v36b(y_pred, returns, df_test, feature_cols, **kwargs):
    """V36b = V36a + skip_price_proximity."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


def backtest_v36c(y_pred, returns, df_test, feature_cols, **kwargs):
    """V36c = V36b + moderate relax_cooldown."""
    return backtest_v32(y_pred, returns, df_test, feature_cols, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    print("=" * 130)
    print("V34 FINAL — leading_v4 (HA features) + target(gain=6%, loss=3%) retrain")
    print("=" * 130)
    print(f"  Feature set: {V34_FEATURE_SET}")
    print(f"  Target     : {V34_TARGET}")
    print(f"  Engine     : V32 (hap_preempt + weak_oversold_exit)")
    print()

    t_rule = run_rule_test(SYMBOLS); m_rule = calc_metrics(t_rule)

    t0 = time.time()
    # V33 baseline
    t_v33 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=lambda *a, **kw: backtest_v33(*a, **{**V33_DELTA, **kw}),
                          feature_set="leading_v3", target_override=V29_TARGET)
    m_v33 = calc_metrics(t_v33); cs_v33 = comp_score(m_v33, t_v33)

    # V34 final
    t_v34 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v34,
                          feature_set=V34_FEATURE_SET, target_override=V34_TARGET)
    dt = time.time() - t0
    m_v34 = calc_metrics(t_v34); cs_v34 = comp_score(m_v34, t_v34)

    HDR = f"  {'Config':<38} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)
    print(HDR); print(SEP)
    print(f"  {'Rule baseline':<38} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V33 (leading_v3+V33engine)':<38} | {m_v33['trades']:>5} {m_v33['wr']:>5.1f}% {m_v33['avg_pnl']:>+7.2f}% {m_v33['total_pnl']:>+9.1f}% {m_v33['pf']:>5.2f} {m_v33['max_loss']:>+7.1f}% {m_v33['avg_hold']:>5.1f}d | {cs_v33:>6.0f}")
    print(f"  {'V34 (leading_v4+V32engine)':<38} | {m_v34['trades']:>5} {m_v34['wr']:>5.1f}% {m_v34['avg_pnl']:>+7.2f}% {m_v34['total_pnl']:>+9.1f}% {m_v34['pf']:>5.2f} {m_v34['max_loss']:>+7.1f}% {m_v34['avg_hold']:>5.1f}d | {cs_v34:>6.0f}")
    print(SEP)
    print(f"  Delta vs V33:  Comp={cs_v34-cs_v33:+.0f}  WR={m_v34['wr']-m_v33['wr']:+.2f}pp  "
          f"AvgPnL={m_v34['avg_pnl']-m_v33['avg_pnl']:+.3f}pp  "
          f"TotPnL={m_v34['total_pnl']-m_v33['total_pnl']:+.1f}%  "
          f"PF={m_v34['pf']-m_v33['pf']:+.3f}")
    print(f"  Time: {dt:.1f}s")

    # Save trades
    df_v34 = pd.DataFrame(t_v34)
    if len(df_v34) > 0:
        out_path = os.path.join(OUT, "trades_v34.csv")
        df_v34.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_v34)} V34 trades → results/trades_v34.csv")

    # Exit reason breakdown
    print("\n  Exit reason breakdown:")
    for reason, grp in df_v34.groupby("exit_reason"):
        pnl = grp["pnl_pct"]
        wr = (pnl > 0).mean() * 100
        print(f"    {reason:<30} n={len(grp):4d} WR={wr:5.1f}% avg={pnl.mean():+.2f}%")

    print("\n" + "=" * 130)
    print("  DONE — V34 Final")
    print("=" * 130)
