"""Deep analysis of V19.3 per-symbol + comparison with Rule."""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_1_compare import run_test as run_test_v191, run_rule_test, calc_metrics
from run_v19_3_compare import backtest_v19_3

SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"

if __name__ == "__main__":
    print("Running V19.3...")
    trades_v193 = run_test_v191(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                                backtest_fn=backtest_v19_3)
    print("Running Rule...")
    trades_rule = run_rule_test(SYMBOLS)

    df_v = pd.DataFrame(trades_v193)
    df_r = pd.DataFrame(trades_rule)

    # ====== 1. PER-SYMBOL DETAILED ======
    print("\n" + "=" * 120)
    print("1. V19.3 PER-SYMBOL DETAIL")
    print("=" * 120)
    print(f"{'Sym':<6} | {'#':>4} {'Win':>4} {'Loss':>5} {'BE':>3} | {'WR':>6} {'AvgPnL':>8} {'TotPnL':>10} {'PF':>6} | "
          f"{'AvgWin':>8} {'AvgLoss':>8} {'MaxLoss':>8} {'MaxWin':>8} {'AvgHold':>8}")
    print("-" * 120)

    syms = sorted(df_v["symbol"].unique())
    for sym in syms:
        st = df_v[df_v["symbol"] == sym]
        wins = st[st["pnl_pct"] > 0]
        losses = st[st["pnl_pct"] < 0]
        be = st[st["pnl_pct"] == 0]
        m = calc_metrics(st.to_dict("records"))
        avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0
        max_loss = st["pnl_pct"].min() if len(st) > 0 else 0
        max_win = st["pnl_pct"].max() if len(st) > 0 else 0
        avg_hold = st["holding_days"].mean() if "holding_days" in st.columns else 0
        print(f"{sym:<6} | {len(st):>4} {len(wins):>4} {len(losses):>5} {len(be):>3} | "
              f"{m['wr']:>5.1f}% {m['avg_pnl']:>+7.2f}% {m['total_pnl']:>+9.1f}% {m['pf']:>5.2f} | "
              f"{avg_win:>+7.2f}% {avg_loss:>+7.2f}% {max_loss:>+7.2f}% {max_win:>+7.2f}% {avg_hold:>6.1f}d")

    # ====== 2. CATEGORIES ======
    print("\n" + "=" * 120)
    print("2. SYMBOL CATEGORIES BY V19.3 PERFORMANCE")
    print("=" * 120)

    sym_pnl = {}
    for sym in syms:
        st = df_v[df_v["symbol"] == sym]
        sym_pnl[sym] = st["pnl_pct"].sum()

    good = {s: p for s, p in sym_pnl.items() if p > 50}
    medium = {s: p for s, p in sym_pnl.items() if 0 < p <= 50}
    bad = {s: p for s, p in sym_pnl.items() if p <= 0}

    print(f"\n  PROFITABLE (>50% total): {len(good)} symbols")
    for s in sorted(good, key=good.get, reverse=True):
        print(f"    {s}: {good[s]:>+.1f}%")

    print(f"\n  MARGINAL (0-50% total): {len(medium)} symbols")
    for s in sorted(medium, key=medium.get, reverse=True):
        print(f"    {s}: {medium[s]:>+.1f}%")

    print(f"\n  LOSING (<=0% total): {len(bad)} symbols")
    for s in sorted(bad, key=bad.get):
        print(f"    {s}: {bad[s]:>+.1f}%")

    # ====== 3. V19.3 vs RULE PER-SYMBOL ======
    print("\n" + "=" * 120)
    print("3. V19.3 vs RULE: WHERE RULE WINS")
    print("=" * 120)

    print(f"\n{'Sym':<6} | {'V19.3 Tot':>10} {'V19.3 WR':>9} {'V19.3 PF':>9} | "
          f"{'Rule Tot':>10} {'Rule WR':>8} {'Rule PF':>8} | {'Gap':>8} {'Winner':>8}")
    print("-" * 110)

    for sym in syms:
        mv = calc_metrics(df_v[df_v["symbol"] == sym].to_dict("records"))
        mr = calc_metrics(df_r[df_r["symbol"] == sym].to_dict("records"))
        gap = mv["total_pnl"] - mr["total_pnl"]
        winner = "V19.3" if gap > 0 else ("Rule" if gap < -10 else "~TIE")
        print(f"{sym:<6} | {mv['total_pnl']:>+9.1f}% {mv['wr']:>7.1f}% {mv['pf']:>8.2f} | "
              f"{mr['total_pnl']:>+9.1f}% {mr['wr']:>6.1f}% {mr['pf']:>7.2f} | {gap:>+7.1f}% {winner:>8}")

    # ====== 4. PERIOD ANALYSIS: WHERE V19.3 LOSES ======
    print("\n" + "=" * 120)
    print("4. V19.3 PERFORMANCE BY YEAR")
    print("=" * 120)

    if "entry_date" in df_v.columns:
        df_v["year"] = df_v["entry_date"].astype(str).str[:4]
        df_r2 = df_r.copy()
        if "entry_date" in df_r2.columns:
            df_r2["year"] = df_r2["entry_date"].astype(str).str[:4]
        elif "entry_day" in df_r2.columns:
            df_r2["year"] = "N/A"

        print(f"\n{'Year':<6} | {'V19.3 #':>8} {'V19.3 Tot':>10} {'V19.3 WR':>9} {'V19.3 PF':>9} | "
              f"{'Rule #':>7} {'Rule Tot':>9} {'Rule WR':>8} | {'Gap':>8}")
        print("-" * 100)
        for yr in sorted(df_v["year"].unique()):
            yv = df_v[df_v["year"] == yr]
            mv = calc_metrics(yv.to_dict("records"))
            if "year" in df_r2.columns:
                yr2 = df_r2[df_r2["year"] == yr]
                mr = calc_metrics(yr2.to_dict("records"))
            else:
                mr = {"trades": 0, "total_pnl": 0, "wr": 0, "pf": 0}
            gap = mv["total_pnl"] - mr["total_pnl"]
            print(f"{yr:<6} | {mv['trades']:>8} {mv['total_pnl']:>+9.1f}% {mv['wr']:>7.1f}% {mv['pf']:>8.2f} | "
                  f"{mr['trades']:>7} {mr['total_pnl']:>+8.1f}% {mr['wr']:>6.1f}% | {gap:>+7.1f}%")

    # ====== 5. WORST LOSING TRADES ======
    print("\n" + "=" * 120)
    print("5. V19.3 ALL LOSING TRADES > -5%")
    print("=" * 120)

    big_losses = df_v[df_v["pnl_pct"] < -5].sort_values("pnl_pct")
    print(f"\n{'Sym':<6} {'Entry':>12} {'Exit':>12} {'PnL':>8} {'ExitR':>20} {'Hold':>5} {'Trend':>8} {'Size':>5} "
          f"{'Score':>6} {'Profile':>10} {'Choppy':>7}")
    print("-" * 120)
    for _, t in big_losses.iterrows():
        print(f"{t.get('symbol','?'):<6} {str(t.get('entry_date',''))[:10]:>12} {str(t.get('exit_date',''))[:10]:>12} "
              f"{t['pnl_pct']:>+7.2f}% {t.get('exit_reason',''):>20} {t.get('holding_days',0):>5}d "
              f"{t.get('entry_trend',''):>8} {t.get('position_size',0):>4.2f} "
              f"{t.get('entry_score',0):>6} {t.get('entry_profile',''):>10} {str(t.get('entry_choppy_regime','')):>7}")

    # ====== 6. EXIT REASON P&L ======
    print("\n" + "=" * 120)
    print("6. V19.3 EXIT REASON DETAILED P&L")
    print("=" * 120)

    for reason, grp in df_v.groupby("exit_reason"):
        wins = len(grp[grp["pnl_pct"] > 0])
        losses_n = len(grp[grp["pnl_pct"] < 0])
        avg = grp["pnl_pct"].mean()
        total_pnl = grp["pnl_pct"].sum()
        avg_hold = grp["holding_days"].mean() if "holding_days" in grp.columns else 0
        print(f"  {reason:<25}: {len(grp):>4} trades ({wins}W/{losses_n}L), WR={wins/len(grp)*100:>5.1f}%, "
              f"avg={avg:>+6.2f}%, total={total_pnl:>+8.1f}%, avg_hold={avg_hold:.1f}d")

    # ====== 7. KEY GAPS: WHERE RULE CATCHES BIG MOVES THAT V19.3 MISSES ======
    print("\n" + "=" * 120)
    print("7. RULE BIG WINS (>15%) VS V19.3 SAME PERIOD")
    print("=" * 120)

    if "entry_date" in df_r.columns:
        rule_big = df_r[df_r["pnl_pct"] > 15].sort_values("pnl_pct", ascending=False)
        print(f"\n{'Sym':<6} {'RuleEntry':>12} {'RuleExit':>12} {'RulePnL':>8} | {'V19.3 same-period trades':>30}")
        print("-" * 90)
        for _, rt in rule_big.iterrows():
            sym = rt.get("symbol", "?")
            r_entry = str(rt.get("entry_date", ""))[:10]
            r_exit = str(rt.get("exit_date", ""))[:10]
            r_pnl = rt["pnl_pct"]
            # Find overlapping V19.3 trades
            v_sym = df_v[(df_v["symbol"] == sym)]
            overlaps = []
            for _, vt in v_sym.iterrows():
                v_entry = str(vt.get("entry_date", ""))[:10]
                v_exit = str(vt.get("exit_date", ""))[:10]
                # Simple overlap check: rule entry <= v_exit AND rule exit >= v_entry
                if v_entry <= r_exit and v_exit >= r_entry:
                    overlaps.append(f"{v_entry}->{v_exit}:{vt['pnl_pct']:+.1f}%({vt.get('exit_reason','')})")
            overlap_str = "; ".join(overlaps) if overlaps else "NO V19.3 TRADE"
            print(f"{sym:<6} {r_entry:>12} {r_exit:>12} {r_pnl:>+7.1f}% | {overlap_str}")

    # ====== 8. ENTRY TREND vs OUTCOME ======
    print("\n" + "=" * 120)
    print("8. V19.3 ENTRY TREND vs OUTCOME PER SYMBOL")
    print("=" * 120)

    for sym in syms:
        st = df_v[df_v["symbol"] == sym]
        if len(st) == 0:
            continue
        print(f"\n  {sym}:")
        for trend in ["strong", "moderate", "weak"]:
            tt = st[st.get("entry_trend", "") == trend] if "entry_trend" in st.columns else pd.DataFrame()
            if len(tt) == 0:
                continue
            m = calc_metrics(tt.to_dict("records"))
            print(f"    {trend:<10}: {m['trades']:>3} trades, WR={m['wr']:>5.1f}%, avg={m['avg_pnl']:>+6.2f}%, total={m['total_pnl']:>+8.1f}%")

    # ====== 9. FAST_EXIT_LOSS ANALYSIS ======
    print("\n" + "=" * 120)
    print("9. FAST_EXIT_LOSS TRADES BREAKDOWN")
    print("=" * 120)

    fast_exits = df_v[df_v["exit_reason"] == "fast_exit_loss"]
    if len(fast_exits) > 0:
        print(f"\n  Total fast_exit_loss: {len(fast_exits)} trades, total PnL = {fast_exits['pnl_pct'].sum():+.1f}%")
        print(f"\n  Per symbol:")
        for sym in syms:
            fe = fast_exits[fast_exits["symbol"] == sym]
            if len(fe) == 0:
                continue
            print(f"    {sym}: {len(fe)} trades, total={fe['pnl_pct'].sum():+.1f}%, avg={fe['pnl_pct'].mean():+.2f}%")

        print(f"\n  Per trend:")
        if "entry_trend" in fast_exits.columns:
            for trend in ["strong", "moderate", "weak"]:
                fe = fast_exits[fast_exits["entry_trend"] == trend]
                if len(fe) == 0:
                    continue
                print(f"    {trend}: {len(fe)} trades, total={fe['pnl_pct'].sum():+.1f}%, avg={fe['pnl_pct'].mean():+.2f}%")

    print("\n" + "=" * 120)
    print("ANALYSIS COMPLETE")
    print("=" * 120)
