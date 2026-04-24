"""
V26 Feature Comparison: leading (V1) vs leading_v2 (V2)
Per-symbol analysis with detailed breakdown of improved/degraded symbols.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import src.safe_io  # noqa: F401

from run_v19_1_compare import run_test, calc_metrics
from run_v26 import backtest_v26
from run_v25 import comp_score
from run_v24 import resolve_symbols
from src.config_loader import get_training_device
from src.models.registry import detect_device

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def per_symbol_metrics(trades):
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    result = {}
    for sym, grp in df.groupby("symbol"):
        n = len(grp)
        wins = (grp["pnl_pct"] > 0).sum()
        wr = wins / n * 100 if n > 0 else 0
        total_pnl = grp["pnl_pct"].sum()
        avg_pnl = grp["pnl_pct"].mean()
        gp = grp.loc[grp["pnl_pct"] > 0, "pnl_pct"].sum()
        gl = abs(grp.loc[grp["pnl_pct"] < 0, "pnl_pct"].sum())
        pf = gp / gl if gl > 0 else 99
        max_loss = grp["pnl_pct"].min()
        avg_hold = grp["holding_days"].mean() if "holding_days" in grp.columns else 0
        result[sym] = {
            "trades": n, "wins": wins, "wr": wr, "avg_pnl": avg_pnl,
            "total_pnl": total_pnl, "pf": pf, "max_loss": max_loss, "avg_hold": avg_hold,
        }
    return result


def main():
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    device = get_training_device()
    resolved = detect_device(device)
    print(f"Device: {resolved.upper()}")

    symbols_list = resolve_symbols("", min_rows=2000)
    symbols_str = ",".join(symbols_list)
    print(f"Using {len(symbols_list)} symbols")

    print(f"\n{'='*130}")
    print(f"V26 FEATURE COMPARISON: leading (V1, ~67 feat) vs leading_v2 (V2, ~99 feat)")
    print(f"{'='*130}")

    # --- Run V26 with leading (V1) ---
    print(f"\n[1/2] Running V26 with feature set: leading ...")
    t0 = time.time()
    trades_v1 = run_test(symbols_str, True, True, False, False, True, True, True, True, True, True,
                         backtest_fn=backtest_v26, device=device, feature_set="leading")
    dt1 = time.time() - t0
    m_v1 = calc_metrics(trades_v1)
    print(f"  Done in {dt1:.0f}s: {m_v1['trades']} trades, WR={m_v1['wr']:.1f}%, "
          f"TotalPnL={m_v1['total_pnl']:+.1f}%, PF={m_v1['pf']:.2f}")

    # --- Run V26 with leading_v2 (V2) ---
    print(f"\n[2/2] Running V26 with feature set: leading_v2 ...")
    t0 = time.time()
    trades_v2 = run_test(symbols_str, True, True, False, False, True, True, True, True, True, True,
                         backtest_fn=backtest_v26, device=device, feature_set="leading_v2")
    dt2 = time.time() - t0
    m_v2 = calc_metrics(trades_v2)
    print(f"  Done in {dt2:.0f}s: {m_v2['trades']} trades, WR={m_v2['wr']:.1f}%, "
          f"TotalPnL={m_v2['total_pnl']:+.1f}%, PF={m_v2['pf']:.2f}")

    # === OVERALL COMPARISON ===
    print(f"\n{'='*130}")
    print(f"OVERALL COMPARISON")
    print(f"{'='*130}")
    print(f"  {'Metric':<20} {'leading (V1)':>15} {'leading_v2 (V2)':>15} {'Delta':>12}")
    print(f"  {'-'*62}")
    print(f"  {'Trades':<20} {m_v1['trades']:>15} {m_v2['trades']:>15} {m_v2['trades']-m_v1['trades']:>+12}")
    print(f"  {'Win Rate %':<20} {m_v1['wr']:>14.2f}% {m_v2['wr']:>14.2f}% {m_v2['wr']-m_v1['wr']:>+11.2f}%")
    print(f"  {'Avg PnL %':<20} {m_v1['avg_pnl']:>+14.3f}% {m_v2['avg_pnl']:>+14.3f}% {m_v2['avg_pnl']-m_v1['avg_pnl']:>+11.3f}%")
    print(f"  {'Total PnL %':<20} {m_v1['total_pnl']:>+14.1f}% {m_v2['total_pnl']:>+14.1f}% {m_v2['total_pnl']-m_v1['total_pnl']:>+11.1f}%")
    print(f"  {'Profit Factor':<20} {m_v1['pf']:>15.3f} {m_v2['pf']:>15.3f} {m_v2['pf']-m_v1['pf']:>+12.3f}")
    print(f"  {'Max Loss %':<20} {m_v1['max_loss']:>+14.2f}% {m_v2['max_loss']:>+14.2f}% {m_v2['max_loss']-m_v1['max_loss']:>+11.2f}%")
    print(f"  {'Avg Hold (days)':<20} {m_v1['avg_hold']:>14.1f}d {m_v2['avg_hold']:>14.1f}d {m_v2['avg_hold']-m_v1['avg_hold']:>+11.1f}d")

    # === PER-SYMBOL COMPARISON ===
    sym_v1 = per_symbol_metrics(trades_v1)
    sym_v2 = per_symbol_metrics(trades_v2)
    all_syms = sorted(set(list(sym_v1.keys()) + list(sym_v2.keys())))

    rows = []
    for sym in all_syms:
        s1 = sym_v1.get(sym, {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "pf": 0, "max_loss": 0})
        s2 = sym_v2.get(sym, {"trades": 0, "wr": 0, "avg_pnl": 0, "total_pnl": 0, "pf": 0, "max_loss": 0})
        d_pnl = s2["total_pnl"] - s1["total_pnl"]
        d_wr = s2["wr"] - s1["wr"]
        d_pf = s2["pf"] - s1["pf"]
        rows.append({
            "symbol": sym,
            "v1_trades": s1["trades"], "v1_wr": round(s1["wr"], 2), "v1_avg_pnl": round(s1["avg_pnl"], 3),
            "v1_total_pnl": round(s1["total_pnl"], 2), "v1_pf": round(s1["pf"], 3), "v1_max_loss": round(s1["max_loss"], 2),
            "v2_trades": s2["trades"], "v2_wr": round(s2["wr"], 2), "v2_avg_pnl": round(s2["avg_pnl"], 3),
            "v2_total_pnl": round(s2["total_pnl"], 2), "v2_pf": round(s2["pf"], 3), "v2_max_loss": round(s2["max_loss"], 2),
            "delta_pnl": round(d_pnl, 2), "delta_wr": round(d_wr, 2), "delta_pf": round(d_pf, 3),
        })

    df_compare = pd.DataFrame(rows).sort_values("delta_pnl", ascending=False)

    # === TOP IMPROVED ===
    improved = df_compare[df_compare["delta_pnl"] > 0].sort_values("delta_pnl", ascending=False)
    degraded = df_compare[df_compare["delta_pnl"] < 0].sort_values("delta_pnl", ascending=True)
    neutral = df_compare[df_compare["delta_pnl"] == 0]

    print(f"\n{'='*130}")
    print(f"TOP {len(improved)} IMPROVED SYMBOLS (V2 > V1)")
    print(f"{'='*130}")
    print(f"  {'Symbol':<8} {'V1 PnL':>10} {'V2 PnL':>10} {'Delta':>10} {'V1 WR':>8} {'V2 WR':>8} "
          f"{'V1 PF':>7} {'V2 PF':>7} {'V1 #':>5} {'V2 #':>5}")
    print(f"  {'-'*88}")
    for _, r in improved.head(25).iterrows():
        print(f"  {r['symbol']:<8} {r['v1_total_pnl']:>+9.1f}% {r['v2_total_pnl']:>+9.1f}% "
              f"{r['delta_pnl']:>+9.1f}% {r['v1_wr']:>7.1f}% {r['v2_wr']:>7.1f}% "
              f"{r['v1_pf']:>6.2f} {r['v2_pf']:>6.2f} {r['v1_trades']:>5} {r['v2_trades']:>5}")

    print(f"\n{'='*130}")
    print(f"TOP {len(degraded)} DEGRADED SYMBOLS (V2 < V1)")
    print(f"{'='*130}")
    print(f"  {'Symbol':<8} {'V1 PnL':>10} {'V2 PnL':>10} {'Delta':>10} {'V1 WR':>8} {'V2 WR':>8} "
          f"{'V1 PF':>7} {'V2 PF':>7} {'V1 #':>5} {'V2 #':>5}")
    print(f"  {'-'*88}")
    for _, r in degraded.iterrows():
        print(f"  {r['symbol']:<8} {r['v1_total_pnl']:>+9.1f}% {r['v2_total_pnl']:>+9.1f}% "
              f"{r['delta_pnl']:>+9.1f}% {r['v1_wr']:>7.1f}% {r['v2_wr']:>7.1f}% "
              f"{r['v1_pf']:>6.2f} {r['v2_pf']:>6.2f} {r['v1_trades']:>5} {r['v2_trades']:>5}")

    # === ANALYSIS: Characteristics of degraded symbols ===
    print(f"\n{'='*130}")
    print(f"ANALYSIS: Characteristics of DEGRADED symbols")
    print(f"{'='*130}")

    df_v1 = pd.DataFrame(trades_v1)
    df_v2 = pd.DataFrame(trades_v2)
    degraded_syms = degraded["symbol"].tolist()

    if degraded_syms and len(df_v2) > 0:
        deg_v2 = df_v2[df_v2["symbol"].isin(degraded_syms)]
        rest_v2 = df_v2[~df_v2["symbol"].isin(degraded_syms)]

        print(f"\n  Degraded group ({len(degraded_syms)} symbols): {', '.join(degraded_syms)}")

        # Trade characteristics
        print(f"\n  {'Metric':<30} {'Degraded':>12} {'Rest':>12}")
        print(f"  {'-'*54}")
        print(f"  {'Avg PnL per trade':<30} {deg_v2['pnl_pct'].mean():>+11.2f}% {rest_v2['pnl_pct'].mean():>+11.2f}%")
        print(f"  {'Median PnL':<30} {deg_v2['pnl_pct'].median():>+11.2f}% {rest_v2['pnl_pct'].median():>+11.2f}%")
        print(f"  {'Avg Holding Days':<30} {deg_v2['holding_days'].mean():>11.1f}d {rest_v2['holding_days'].mean():>11.1f}d")

        if "entry_trend" in deg_v2.columns:
            print(f"\n  Trend distribution (V2):")
            for label, grp in [("Degraded", deg_v2), ("Rest", rest_v2)]:
                trend_dist = grp["entry_trend"].value_counts(normalize=True) * 100
                parts = [f"{t}={v:.0f}%" for t, v in trend_dist.items()]
                print(f"    {label}: {', '.join(parts)}")

        if "exit_reason" in deg_v2.columns:
            print(f"\n  Exit reason distribution:")
            for label, grp in [("Degraded", deg_v2), ("Rest", rest_v2)]:
                exit_dist = grp["exit_reason"].value_counts(normalize=True) * 100
                parts = [f"{r}={v:.0f}%" for r, v in exit_dist.head(6).items()]
                print(f"    {label}: {', '.join(parts)}")

        if "entry_choppy_regime" in deg_v2.columns:
            print(f"\n  Choppy regime rate:")
            for label, grp in [("Degraded", deg_v2), ("Rest", rest_v2)]:
                choppy_pct = grp["entry_choppy_regime"].astype(str).isin(["True", "1"]).mean() * 100
                print(f"    {label}: {choppy_pct:.1f}%")

        if "entry_score" in deg_v2.columns:
            print(f"\n  Avg entry score:")
            for label, grp in [("Degraded", deg_v2), ("Rest", rest_v2)]:
                print(f"    {label}: {grp['entry_score'].mean():.2f}")

        # Compare V1 vs V2 for degraded symbols specifically
        print(f"\n  V1 vs V2 for degraded symbols:")
        for sym in degraded_syms[:10]:
            s1_trades = df_v1[df_v1["symbol"] == sym] if len(df_v1) > 0 else pd.DataFrame()
            s2_trades = df_v2[df_v2["symbol"] == sym] if len(df_v2) > 0 else pd.DataFrame()
            v1_n = len(s1_trades)
            v2_n = len(s2_trades)
            v1_wr = (s1_trades["pnl_pct"] > 0).mean() * 100 if v1_n > 0 else 0
            v2_wr = (s2_trades["pnl_pct"] > 0).mean() * 100 if v2_n > 0 else 0
            v1_pnl = s1_trades["pnl_pct"].sum() if v1_n > 0 else 0
            v2_pnl = s2_trades["pnl_pct"].sum() if v2_n > 0 else 0

            new_trades = v2_n - v1_n
            tag = f"+{new_trades} new trades" if new_trades > 0 else f"{new_trades} trades" if new_trades < 0 else "same #"
            print(f"    {sym:<6}: V1({v1_n} trades, WR={v1_wr:.0f}%, PnL={v1_pnl:+.1f}%) -> "
                  f"V2({v2_n} trades, WR={v2_wr:.0f}%, PnL={v2_pnl:+.1f}%) [{tag}]")

    # === REVIEW: Losing trades of degraded symbols (V2) ===
    print(f"\n{'='*130}")
    print(f"LOSING TRADES REVIEW: Top losing trades of DEGRADED symbols (V2)")
    print(f"{'='*130}")

    if degraded_syms and len(df_v2) > 0:
        deg_losses = df_v2[(df_v2["symbol"].isin(degraded_syms)) & (df_v2["pnl_pct"] < 0)].copy()
        deg_losses = deg_losses.sort_values("pnl_pct", ascending=True)

        print(f"\n  Total losing trades in degraded group: {len(deg_losses)}")
        print(f"  Total loss: {deg_losses['pnl_pct'].sum():+.1f}%")

        print(f"\n  TOP 30 WORST TRADES:")
        print(f"  {'Symbol':<6} {'Entry Date':<12} {'Exit Date':<12} {'Hold':>5} {'PnL':>8} {'MaxProf':>8} "
              f"{'Exit Reason':<20} {'Trend':<10} {'Score':>5} {'Choppy':>6}")
        print(f"  {'-'*110}")

        for _, t in deg_losses.head(30).iterrows():
            entry_d = str(t.get("entry_date", ""))[:10]
            exit_d = str(t.get("exit_date", ""))[:10]
            trend = str(t.get("entry_trend", ""))
            score = t.get("entry_score", 0)
            choppy = str(t.get("entry_choppy_regime", ""))
            max_prof = t.get("max_profit_pct", 0)
            exit_r = str(t.get("exit_reason", ""))
            print(f"  {t['symbol']:<6} {entry_d:<12} {exit_d:<12} {t['holding_days']:>5.0f} "
                  f"{t['pnl_pct']:>+7.2f}% {max_prof:>+7.1f}% {exit_r:<20} {trend:<10} {score:>5.0f} {choppy:>6}")

        # Per-symbol loss analysis for degraded
        print(f"\n  PER-SYMBOL LOSS BREAKDOWN (degraded):")
        print(f"  {'Symbol':<8} {'#Loss':>6} {'TotalLoss':>12} {'AvgLoss':>10} {'MaxLoss':>10} {'Top Exit Reasons'}")
        print(f"  {'-'*90}")

        for sym in degraded_syms:
            sym_losses = deg_losses[deg_losses["symbol"] == sym]
            if len(sym_losses) == 0:
                continue
            total_loss = sym_losses["pnl_pct"].sum()
            avg_loss = sym_losses["pnl_pct"].mean()
            max_loss = sym_losses["pnl_pct"].min()
            exit_reasons = sym_losses["exit_reason"].value_counts().head(3)
            reasons_str = ", ".join([f"{r}({c})" for r, c in exit_reasons.items()])
            print(f"  {sym:<8} {len(sym_losses):>6} {total_loss:>+11.1f}% {avg_loss:>+9.2f}% "
                  f"{max_loss:>+9.2f}% {reasons_str}")

        # NEW vs MISSING trades analysis
        print(f"\n  NEW/CHANGED TRADES ANALYSIS (degraded symbols):")
        print(f"  Comparing trades that exist in V2 but not V1 (or different):")
        for sym in degraded_syms[:10]:
            s1 = df_v1[df_v1["symbol"] == sym]
            s2 = df_v2[df_v2["symbol"] == sym]
            if len(s1) == 0 or len(s2) == 0:
                continue

            s1_dates = set(s1["entry_date"].astype(str).values) if "entry_date" in s1.columns else set()
            s2_dates = set(s2["entry_date"].astype(str).values) if "entry_date" in s2.columns else set()

            new_dates = s2_dates - s1_dates
            lost_dates = s1_dates - s2_dates
            common_dates = s1_dates & s2_dates

            if new_dates:
                new_trades_df = s2[s2["entry_date"].astype(str).isin(new_dates)]
                new_pnl = new_trades_df["pnl_pct"].sum()
                new_wins = (new_trades_df["pnl_pct"] > 0).sum()
                print(f"    {sym}: {len(new_dates)} NEW entries (PnL={new_pnl:+.1f}%, {new_wins}W/{len(new_dates)}T), "
                      f"{len(lost_dates)} REMOVED entries, {len(common_dates)} COMMON")
                if new_pnl < -5:
                    print(f"      Harmful new trades:")
                    for _, nt in new_trades_df[new_trades_df["pnl_pct"] < 0].sort_values("pnl_pct").head(5).iterrows():
                        print(f"        {nt.get('entry_date','')}: PnL={nt['pnl_pct']:+.2f}%, "
                              f"exit={nt.get('exit_reason','')}, trend={nt.get('entry_trend','')}")
            elif lost_dates:
                lost_trades_df = s1[s1["entry_date"].astype(str).isin(lost_dates)]
                lost_pnl = lost_trades_df["pnl_pct"].sum()
                print(f"    {sym}: 0 new, {len(lost_dates)} REMOVED (had PnL={lost_pnl:+.1f}%)")

    # === SAVE RESULTS ===
    df_compare.to_csv(os.path.join(OUT, "v26_feature_compare_per_symbol.csv"), index=False)
    print(f"\n  Saved per-symbol comparison to results/v26_feature_compare_per_symbol.csv")

    # Save V2 trades
    df_v2_out = pd.DataFrame(trades_v2)
    df_v2_out.to_csv(os.path.join(OUT, "trades_v26_v2.csv"), index=False)
    print(f"  Saved V2 trades to results/trades_v26_v2.csv")

    # Save V1 trades
    df_v1_out = pd.DataFrame(trades_v1)
    df_v1_out.to_csv(os.path.join(OUT, "trades_v26_v1.csv"), index=False)
    print(f"  Saved V1 trades to results/trades_v26_v1.csv")

    # === SUMMARY ===
    print(f"\n{'='*130}")
    print(f"SUMMARY")
    print(f"{'='*130}")
    print(f"  Feature V1 (leading):    {m_v1['trades']} trades, WR={m_v1['wr']:.2f}%, PnL={m_v1['total_pnl']:+.1f}%, PF={m_v1['pf']:.3f}")
    print(f"  Feature V2 (leading_v2): {m_v2['trades']} trades, WR={m_v2['wr']:.2f}%, PnL={m_v2['total_pnl']:+.1f}%, PF={m_v2['pf']:.3f}")
    print(f"  Delta:                   trades={m_v2['trades']-m_v1['trades']:+d}, WR={m_v2['wr']-m_v1['wr']:+.2f}%, "
          f"PnL={m_v2['total_pnl']-m_v1['total_pnl']:+.1f}%, PF={m_v2['pf']-m_v1['pf']:+.3f}")
    print(f"  Improved: {len(improved)} symbols | Degraded: {len(degraded)} symbols | Neutral: {len(neutral)} symbols")
    print(f"  Total time: {dt1+dt2:.0f}s ({(dt1+dt2)/60:.1f} min)")
    print(f"\n{'='*130}")


if __name__ == "__main__":
    main()
