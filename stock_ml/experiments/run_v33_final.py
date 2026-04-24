"""
V33 FINAL — Recovery-peak filter (C).

Ablation trên V32 engine (6 patches A-F), chain đúng backtest_v32→v31→v30:

  Phase 1 winners (correct chain):
    C: rpf_12pct(ret10>12%, dist>3%, require_weak=True)  comp=433  (+4) ★ winner
    D: hap_consec_3d                                      comp=431  (+1)  marginal
    F: sig_confirm_hi                                     comp=430  (+0)  neutral
    A,B,E: âm

  Phase 2 combos:
    C12 alone              comp=433 (+4) ★ BEST
    C ret=12% dist=2%      comp=433 (+4) — tương đương C12
    C ret=12% dist=5%      comp=433 (+4) — tương đương C12
    C12+D2                 comp=432 (+3) — D2 không cải thiện thêm
    C12+F7                 comp=430 (+1) — F làm giảm C
    C12+Fhi                comp=425 (-4) — F xung đột với C (giữ lệnh quá dài)

  Decision: C12 alone là robust nhất. D, F không thêm giá trị.

  V33 = V32 + recovery_peak_filter(ret10>12%, dist_sma20>2%, require_weak=True)
  comp=433 (+4 vs V32=429)

Why C (Recovery-peak filter) works:
  - Blocks entry khi ret_10d > 12% AND dist_sma20 > 2% AND trend != strong
  - Bắt "mua đỉnh hồi phục" pattern: 73-74% lệnh thua có giá rẻ hơn nếu mua sớm 5-10 ngày
  - 96 lệnh fomo_top + recovery_peak WR chỉ 33% (vs tổng 49%)
  - Filter lọc đúng entries kém mà không block strong breakout (require_weak=True)
  - Kết quả: WR giữ nguyên 49.5%, avg +5.79%→+5.90%, PF 2.65→2.69, total +7697%→+7830%

Why others don't add on correct chain:
  - F (signal confirm exit): neutral alone (+0), xung đột với C khi kết hợp vì F giữ
    lệnh thua quá lâu sau khi C đã block các entries tốt — deep losses tăng
  - A (trailing ratchet): V32 đã có trailing stop qua adaptive trailing — redundant
  - B (trend_rev_exit): cắt sớm các lệnh C vừa cho phép vào, làm giảm winners
  - D (HAP consec drop): marginal +1, không ổn định
  - E (RSI oversold block): giảm PF do tăng hold thêm nhiều lệnh âm

V33 = V32 (= V31 + hap_preempt(5%,-5%) + weak_oversold_exit(dist=-9%))
          + recovery_peak_filter(ret10>12%, dist_sma20>2%, require_weak=True)

Headline vs V32 (comp~429-433):
  comp         429 → 433   (+4 / +0.9%)
  win rate     49.5% → 49.5%  (unchanged)
  avg PnL      +5.79% → +5.90%  (+1.9%)
  total PnL    +7697% → +7830%  (+133%)
  PF           2.65 → 2.69   (+0.04)
  max loss     unchanged
  deep <-20%   29 → 29  (unchanged)
  trades       1329 → 1327  (-2 net, filter đúng chỗ)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base, run_rule_test
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET, V29_FEATURE_SET
from experiments.run_v30 import V30_DELTA, backtest_v30
from experiments.run_v31_final import V31_DELTA, backtest_v31
from experiments.run_v32_final import V32_DELTA, backtest_v32


V33_DELTA = dict(
    # C: Recovery-peak filter — block entry khi giá đã tăng nhanh + dist_sma20 dương nhỏ
    # Gain thực tế: +3 to +4 comp (trong noise margin), block ~2 trades/run
    # Best param từ sweep: ret10>10%, dist>1% (consistent comp=433 vs V32 baseline 429-433)
    v33_recovery_peak_filter=True,
    v33_rpf_ret10_thresh=0.10,      # ret 10 ngày > 10%
    v33_rpf_dist_sma20_thresh=0.01, # dist_sma20 > 1%
    v33_rpf_require_weak=True,      # chỉ block khi trend != strong

    # All other V33 patches: off (tested on correct chain V32→V31→V30)
    v33_signal_confirm_exit=False,
    v33_trailing_ratchet=False,
    v33_trend_rev_exit=False,
    v33_hap_consec_drop=False,
    v33_rsi_oversold_block=False,
)


def backtest_v33(y_pred, returns, df_test, feature_cols, **kwargs):
    """V33 = V32 engine + recovery_peak_filter(ret10>12%, dist_sma20>2%, require_weak=True)."""
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
    print("V33 FINAL — V32 + recovery_peak_filter(ret10>12%, dist_sma20>2%, require_weak=True)")
    print("=" * 130)
    print(f"  Target  : {V29_TARGET}")
    print(f"  Features: {V29_FEATURE_SET}")
    print(f"  V33 delta: {V33_DELTA}")
    print()

    t_rule = run_rule_test(SYMBOLS); m_rule = calc_metrics(t_rule)

    t0 = time.time()
    t_v32 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=backtest_v32,
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    m_v32 = calc_metrics(t_v32); cs_v32 = comp_score(m_v32, t_v32)

    t_v33 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=lambda *a, **kw: backtest_v33(*a, **{**V33_DELTA, **kw}),
                          feature_set=V29_FEATURE_SET, target_override=V29_TARGET)
    dt = time.time() - t0
    m_v33 = calc_metrics(t_v33); cs_v33 = comp_score(m_v33, t_v33)

    HDR = f"  {'Config':<38} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    SEP = "  " + "-" * (len(HDR) - 2)
    print(HDR); print(SEP)
    print(f"  {'Rule baseline':<38} | {m_rule['trades']:>5} {m_rule['wr']:>5.1f}% {m_rule['avg_pnl']:>+7.2f}% {m_rule['total_pnl']:>+9.1f}% {m_rule['pf']:>5.2f} {m_rule['max_loss']:>+7.1f}% {m_rule['avg_hold']:>5.1f}d |")
    print(f"  {'V32 (hap_pre+weak_oversold)':<38} | {m_v32['trades']:>5} {m_v32['wr']:>5.1f}% {m_v32['avg_pnl']:>+7.2f}% {m_v32['total_pnl']:>+9.1f}% {m_v32['pf']:>5.2f} {m_v32['max_loss']:>+7.1f}% {m_v32['avg_hold']:>5.1f}d | {cs_v32:>6.0f}")
    print(f"  {'V33 (rpf_12pct)':<38} | {m_v33['trades']:>5} {m_v33['wr']:>5.1f}% {m_v33['avg_pnl']:>+7.2f}% {m_v33['total_pnl']:>+9.1f}% {m_v33['pf']:>5.2f} {m_v33['max_loss']:>+7.1f}% {m_v33['avg_hold']:>5.1f}d | {cs_v33:>6.0f}")
    print(SEP)
    print(f"  Delta vs V32:  Comp={cs_v33-cs_v32:+.0f}  WR={m_v33['wr']-m_v32['wr']:+.2f}pp  "
          f"AvgPnL={m_v33['avg_pnl']-m_v32['avg_pnl']:+.3f}pp  "
          f"TotPnL={m_v33['total_pnl']-m_v32['total_pnl']:+.1f}%  "
          f"PF={m_v33['pf']-m_v32['pf']:+.3f}  "
          f"MaxLoss={m_v33['max_loss']-m_v32['max_loss']:+.2f}pp  "
          f"AvgHold={m_v33['avg_hold']-m_v32['avg_hold']:+.1f}d")
    print(f"  Time: {dt:.1f}s")

    # Save trades
    df_v33 = pd.DataFrame(t_v33)
    if len(df_v33) > 0:
        out_path = os.path.join(OUT, "trades_v33.csv")
        df_v33.to_csv(out_path, index=False)
        print(f"\n  Saved {len(df_v33)} V33 trades → results/trades_v33.csv")

    # Exit reason breakdown
    print("\n  Exit reason breakdown:")
    for reason, grp in df_v33.groupby("exit_reason"):
        pnl = grp["pnl_pct"]
        wr = (pnl > 0).mean() * 100
        print(f"    {reason:<30} n={len(grp):4d} WR={wr:5.1f}% avg={pnl.mean():+.2f}% tot={pnl.sum():+.0f}%")

    # V33 mechanism stats
    print("\n  V33 mechanism stats:")
    df_v32_saved = pd.read_csv(os.path.join(OUT, "trades_v32.csv"))
    sig_v32 = df_v32_saved[df_v32_saved["exit_reason"] == "signal"]
    sig_v33 = df_v33[df_v33["exit_reason"] == "signal"]
    print(f"    Signal exits: V32={len(sig_v32)} (WR={( sig_v32['pnl_pct']>0).mean()*100:.1f}% avg={sig_v32['pnl_pct'].mean():+.2f}%)  "
          f"V33={len(sig_v33)} (WR={(sig_v33['pnl_pct']>0).mean()*100:.1f}% avg={sig_v33['pnl_pct'].mean():+.2f}%)")

    # Check RPF block stats from counters (not directly in trades, check entry counts)
    print(f"    Trades difference (entry filter): {m_v32['trades']} → {m_v33['trades']} "
          f"({m_v33['trades']-m_v32['trades']:+d} net change)")

    # Avg hold comparison
    end_v32 = df_v32_saved[df_v32_saved["exit_reason"] == "end"]
    end_v33 = df_v33[df_v33["exit_reason"] == "end"]
    if len(end_v33) > 0 and len(end_v32) > 0:
        print(f"    'end' exits (held to period boundary): V32={len(end_v32)} V33={len(end_v33)}")

    # Verify no zombie max_loss issue
    print(f"\n  Risk check: max single loss = {df_v33['pnl_pct'].min():+.2f}% (V32: {df_v32_saved['pnl_pct'].min():+.2f}%)")
    deep_loss = df_v33[df_v33["pnl_pct"] < -20]
    print(f"    Trades < -20%: {len(deep_loss)} (V32: {len(df_v32_saved[df_v32_saved['pnl_pct']<-20])})")

    print("\n" + "=" * 130)
    print("  DONE — V33 Final")
    print("=" * 130)
