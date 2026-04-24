"""V41 — Sweep toàn diện tất cả cải tiến trên V39a2 base (score=410, best hiện tại).

Phân tích V39a2 (best=410):
  - Signal exit 20-35d: n=529, WR=21.4%, avg=-4.79%, tot=-2535%  ← DRAG 1
  - HAP preempt      : n= 89, WR= 0.0%, avg=-8.90%, tot= -792%  ← DRAG 2
  - fast_exit_loss   : n= 27, WR= 0.0%, avg=-7.51%, tot= -203%  ← DRAG 3

V39x series chỉ fix một vấn đề mỗi lần; V41 thử tất cả combo chưa test:

Direction A — Chặn signal exit sớm (20-35d window):
  A1: min_hold=25 (nhẹ hơn 35 — giảm ảnh hưởng zombie)
  A2: min_hold=30
  A3: rule_confirm + min_hold=25 (V39a2 base: rule_confirm only, không min_hold)

Direction B — HAP reform (89 trades, all <-5%, hold=20d):
  B1: HAP floor=-0.07 (wider room before trigger)
  B2: HAP trigger=0.08 + floor=-0.07 (raise threshold + widen floor)
  B3: HAP min_hold=20 (block HAP trong 20d đầu)
  B4: HAP trigger=0.10 (chỉ trigger sau 10% profit — bắt sóng dài hơn)

Direction C — Profit protection (tránh big winners thoát quá sớm):
  C1: V33 trailing ratchet tier: 12%→40%, 25%→60%, 40%→70%
  C2: V33 trend reversal exit (close<ema8 2 ngày + rsi<50 → exit khi max_profit>8%)

Direction D — Dead trade cleanup (fast_exit_loss + stall):
  D1: V38b stall exit: hold>=10d + max_profit<2% + cur_ret<-2.5%

Direction E — Combo tốt nhất:
  E1: A1(min_hold=25) + B3(hap_min_hold=20)
  E2: A2(min_hold=30) + B2(trigger=8%+floor=-7%)
  E3: A1 + C1(trailing_ratchet)
  E4: B3(hap_min_hold=20) + C1(trailing_ratchet)
  E5: A1 + B3 + C1  ← "all three"
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.safe_io  # noqa: F401

import pandas as pd
from src.experiment_runner import run_test as run_test_base
from src.evaluation.scoring import calc_metrics, composite_score as comp_score
from src.config_loader import get_pipeline_symbols
from experiments.run_v29 import V29_TARGET
from experiments.run_v34_final import V34_FEATURE_SET
from experiments.run_v39a2 import backtest_v39a2


# V39a2 (score=410) được train với V29_TARGET (g5%,l4%), leading_v4
# → giữ nguyên để so sánh fair với file trades_v39a2.csv
V41_TARGET   = V29_TARGET      # early_wave fw=8,g5%,l4% — same as V39a2
V41_FEATURES = V34_FEATURE_SET # leading_v4


# ── Base layer flags (V39a2 = rule_confirm_exit) ──────────────────────────────
V39A2_FLAGS = dict(v39a_rule_confirm_exit=True)


def _make_bt(extra: dict):
    """Return a backtest fn = V39a2 + extra flags."""
    flags = {**V39A2_FLAGS, **extra}
    def bt(y_pred, returns, df_test, feature_cols, **kwargs):
        merged = {**flags, **kwargs}
        for k, v in flags.items():
            merged[k] = v
        return backtest_v39a2(y_pred, returns, df_test, feature_cols, **merged)
    return bt


VARIANTS = {
    # Baseline
    "V39a2-base": (dict(), "rule_confirm only (baseline)"),

    # Direction A — signal exit hold
    "A1-hold25":  (dict(v39a_signal_exit_min_hold=25),
                   "A1: signal exit min_hold=25"),
    "A2-hold30":  (dict(v39a_signal_exit_min_hold=30),
                   "A2: signal exit min_hold=30"),
    "A3-hold25+rc": (dict(v39a_signal_exit_min_hold=25,
                          v39a_rule_confirm_exit=True),
                     "A3: min_hold=25 + rule_confirm"),

    # Direction B — HAP reform
    "B1-floor7":  (dict(v32_hap_pre_floor=-0.07),
                   "B1: HAP floor=-7%"),
    "B2-trig8f7": (dict(v39b_hap_trigger=0.08, v32_hap_pre_floor=-0.07),
                   "B2: HAP trig=8%+floor=-7%"),
    "B3-hold20":  (dict(v39b_hap_min_hold=20),
                   "B3: HAP min_hold=20d"),
    "B4-trig10":  (dict(v39b_hap_trigger=0.10),
                   "B4: HAP trigger=10%"),
    "B5-trig8h15":(dict(v39b_hap_trigger=0.08, v39b_hap_min_hold=15),
                   "B5: HAP trig=8%+min_hold=15 (V39b params on V39a2)"),

    # Direction C — profit protection
    "C1-ratchet": (dict(v33_trailing_ratchet=True,
                        v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                        v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.60,
                        v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.70),
                   "C1: trailing ratchet 12/25/40%"),
    "C2-trev":    (dict(v33_trend_rev_exit=True,
                        v33_tre_min_profit=0.08,
                        v33_tre_rsi_thresh=50.0,
                        v33_tre_hold_min=5),
                   "C2: trend reversal exit (profit>8%+rsi<50)"),

    # Direction D — dead trade cleanup
    "D1-stall":   (dict(v38b_stall_exit=True,
                        v38b_stall_min_hold=10,
                        v38b_stall_max_profit=0.02,
                        v38b_stall_pnl_thresh=-0.025),
                   "D1: stall exit h>=10+mp<2%+ret<-2.5%"),

    # Direction E — combos
    "E1-A1B3":    (dict(v39a_signal_exit_min_hold=25,
                        v39b_hap_min_hold=20),
                   "E1: A1(hold25)+B3(hap_h20)"),
    "E2-A2B2":    (dict(v39a_signal_exit_min_hold=30,
                        v39b_hap_trigger=0.08, v32_hap_pre_floor=-0.07),
                   "E2: A2(hold30)+B2(trig8+fl7)"),
    "E3-A1C1":    (dict(v39a_signal_exit_min_hold=25,
                        v33_trailing_ratchet=True,
                        v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                        v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.60,
                        v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.70),
                   "E3: A1(hold25)+C1(ratchet)"),
    "E4-B3C1":    (dict(v39b_hap_min_hold=20,
                        v33_trailing_ratchet=True,
                        v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                        v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.60,
                        v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.70),
                   "E4: B3(hap_h20)+C1(ratchet)"),
    "E5-A1B3C1":  (dict(v39a_signal_exit_min_hold=25,
                        v39b_hap_min_hold=20,
                        v33_trailing_ratchet=True,
                        v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                        v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.60,
                        v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.70),
                   "E5: A1+B3+C1 all three"),
    "E6-A1B3D1":  (dict(v39a_signal_exit_min_hold=25,
                        v39b_hap_min_hold=20,
                        v38b_stall_exit=True,
                        v38b_stall_min_hold=10,
                        v38b_stall_max_profit=0.02,
                        v38b_stall_pnl_thresh=-0.025),
                   "E6: A1+B3+D1(stall)"),
    "E7-A1B3C1D1":(dict(v39a_signal_exit_min_hold=25,
                        v39b_hap_min_hold=20,
                        v33_trailing_ratchet=True,
                        v33_tr_tier1_trigger=0.12, v33_tr_tier1_keep=0.40,
                        v33_tr_tier2_trigger=0.25, v33_tr_tier2_keep=0.60,
                        v33_tr_tier3_trigger=0.40, v33_tr_tier3_keep=0.70,
                        v38b_stall_exit=True,
                        v38b_stall_min_hold=10,
                        v38b_stall_max_profit=0.02,
                        v38b_stall_pnl_thresh=-0.025),
                   "E7: A1+B3+C1+D1 kitchen sink"),
}


def _fmt_row(label, m, cs, base_cs):
    delta = f"  ({cs - base_cs:+.0f})" if base_cs is not None else ""
    return (
        f"  {label:<24} | {m['trades']:>5} {m['wr']:>5.1f}%"
        f" {m['avg_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}%"
        f" {m['pf']:>5.2f} {m['max_loss']:>+7.1f}% {m['avg_hold']:>5.1f}d"
        f" | {cs:>6.0f}{delta}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--variants", type=str, default="",
                        help="Comma-separated variant keys to run (empty=all)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save trades CSV for variants that beat base")
    args = parser.parse_args()

    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    SYMBOLS = ",".join(get_pipeline_symbols(args.symbols))
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(OUT, exist_ok=True)

    requested = (
        {v.strip() for v in args.variants.split(",") if v.strip()}
        if args.variants else set(VARIANTS.keys())
    )

    print("=" * 140)
    print("V41 — Sweep toàn diện tất cả cải tiến trên V39a2 base (score=410)")
    print("=" * 140)
    print(f"  Engine  : V37a → V39a2 (rule_confirm_exit)")
    print(f"  Features: {V41_FEATURES}")
    print(f"  Target  : {V41_TARGET}")
    print(f"  Variants: {len(requested)} to run")
    print()

    HDR = (
        f"  {'Config':<24} | {'#':>5} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10}"
        f" {'PF':>6} {'MaxLoss':>8} {'AvgH':>6} | {'Comp':>7}"
    )
    SEP = "  " + "-" * (len(HDR) - 2)
    print(HDR); print(SEP)

    base_cs = None
    results = {}
    t0 = time.time()

    for key, (extra_flags, desc) in VARIANTS.items():
        if key not in requested:
            continue

        bt_fn = _make_bt(extra_flags)
        trades = run_test_base(
            SYMBOLS, True, True, False, False, True, True, True, True, True, True,
            backtest_fn=bt_fn,
            feature_set=V41_FEATURES,
            target_override=V41_TARGET,
        )
        m = calc_metrics(trades)
        cs = comp_score(m, trades)
        results[key] = (m, trades, cs, desc)

        if key == "V39a2-base":
            base_cs = cs

        label = f"{key} [{desc[:18]}]" if len(desc) > 18 else f"{key} [{desc}]"
        print(_fmt_row(label, m, cs, base_cs))

    dt = time.time() - t0
    print(SEP)
    print(f"  Time: {dt:.1f}s  ({dt/60:.1f} min)")

    # Ranked summary
    ranked = sorted(results.items(), key=lambda x: -x[1][2])
    print(f"\n{'='*60}")
    print("RANKING (top 10):")
    print(f"{'='*60}")
    for rank, (key, (m, t, cs, desc)) in enumerate(ranked[:10], 1):
        delta = cs - (base_cs or cs)
        print(f"  #{rank:<2} {key:<16} score={cs:>6.0f}  ({delta:+.0f})  {desc}")

    # Save trades for best variants (beat base)
    if args.save_best and base_cs is not None:
        print("\nSaving trades for variants that beat base:")
        for key, (m, t, cs, desc) in results.items():
            if cs > base_cs:
                df_t = pd.DataFrame(t)
                fname = f"trades_v41_{key.lower().replace('-', '_')}.csv"
                out_path = os.path.join(OUT, fname)
                df_t.to_csv(out_path, index=False)
                print(f"  +{cs-base_cs:.0f} | {key} → results/{fname}")

                # Exit breakdown
                if "exit_reason" in df_t.columns:
                    print(f"  Exit breakdown ({key}):")
                    for reason, grp in df_t.groupby("exit_reason"):
                        pnl = grp["pnl_pct"]
                        wr = (pnl > 0).mean() * 100
                        print(
                            f"    {reason:<28} n={len(grp):<5} WR={wr:>5.1f}%"
                            f"  avg={pnl.mean():>+7.2f}%  tot={pnl.sum():>+9.1f}%"
                        )
