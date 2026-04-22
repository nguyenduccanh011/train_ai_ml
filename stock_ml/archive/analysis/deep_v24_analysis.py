"""Deep analysis of V24 actual results vs rule, V23, V19.1, V21, V22"""
import pandas as pd
import numpy as np
from datetime import datetime

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

RESULTS = "results/"
COMMON_SYMBOLS = ['AAS','AAV','ACB','BID','DGC','FPT','HPG','MBB','REE','SSI','TCB','VIC','VND','VNM']

def load_all():
    v24 = pd.read_csv(RESULTS + "trades_v24.csv")
    v24['model'] = 'v24'
    v24['entry_date'] = pd.to_datetime(v24['entry_date'])
    v24['exit_date'] = pd.to_datetime(v24['exit_date'])

    rule = pd.read_csv(RESULTS + "trades_rule.csv")
    rule['model'] = 'rule'
    rule['entry_date'] = pd.to_datetime(rule['entry_date'])
    rule['exit_date'] = pd.to_datetime(rule['exit_date'])

    frames = {'v24': v24, 'rule': rule}
    for name in ['v23', 'v19_1', 'v22', 'v16']:
        try:
            df = pd.read_csv(RESULTS + f"trades_{name}.csv")
            df['model'] = name if 'model' not in df.columns else df['model']
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            frames[name] = df
        except:
            pass
    return frames

def summary_stats(df, label=""):
    n = len(df)
    if n == 0:
        return {}
    wins = (df['pnl_pct'] > 0).sum()
    wr = wins / n * 100
    avg_pnl = df['pnl_pct'].mean()
    total_pnl = df['pnl_pct'].sum()
    avg_win = df.loc[df['pnl_pct'] > 0, 'pnl_pct'].mean() if wins > 0 else 0
    avg_loss = df.loc[df['pnl_pct'] <= 0, 'pnl_pct'].mean() if (n - wins) > 0 else 0
    max_loss = df['pnl_pct'].min()
    max_win = df['pnl_pct'].max()
    pf = abs(df.loc[df['pnl_pct'] > 0, 'pnl_pct'].sum() / df.loc[df['pnl_pct'] < 0, 'pnl_pct'].sum()) if df.loc[df['pnl_pct'] < 0, 'pnl_pct'].sum() != 0 else 999
    avg_hold = df['holding_days'].mean() if 'holding_days' in df.columns else 0
    return {
        'model': label, 'trades': n, 'WR%': wr, 'AvgPnL': avg_pnl,
        'TotalPnL': total_pnl, 'PF': pf, 'AvgWin': avg_win, 'AvgLoss': avg_loss,
        'MaxWin': max_win, 'MaxLoss': max_loss, 'AvgHold': avg_hold
    }

def print_section(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def main():
    data = load_all()
    v24 = data['v24']
    rule = data['rule']

    # ====================================================================
    # SECTION 1: V24 OVERALL STATS
    # ====================================================================
    print_section("1. V24 TỔNG QUAN - TOÀN BỘ 367 SYMBOLS")

    stats_all = []
    stats_all.append(summary_stats(v24, 'V24 (ALL 367 symbols)'))
    v24_14 = v24[v24['symbol'].isin(COMMON_SYMBOLS)]
    stats_all.append(summary_stats(v24_14, 'V24 (14 common symbols)'))
    for name, df in data.items():
        if name != 'v24':
            stats_all.append(summary_stats(df, name.upper()))
    print(pd.DataFrame(stats_all).to_string(index=False))

    # ====================================================================
    # SECTION 2: V24 PER-SYMBOL vs RULE vs V23 vs V19.1
    # ====================================================================
    print_section("2. V24 PER-SYMBOL TRÊN 14 MÃ CHUNG - SO SÁNH CHI TIẾT")

    for sym in COMMON_SYMBOLS:
        print(f"\n--- {sym} ---")
        rows = []
        for name, df in data.items():
            sub = df[df['symbol'] == sym]
            if len(sub) > 0:
                rows.append(summary_stats(sub, name.upper()))
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))

    # ====================================================================
    # SECTION 3: V24 on VND - DEEP DIVE
    # ====================================================================
    print_section("3. VND - DEEP DIVE: V24 vs RULE vs V23 vs V19.1")

    vnd_v24 = v24[v24['symbol'] == 'VND'].sort_values('entry_date')
    vnd_rule = rule[rule['symbol'] == 'VND'].sort_values('entry_date')

    print(f"\nV24 trades on VND: {len(vnd_v24)}")
    print(f"Rule trades on VND: {len(vnd_rule)}")

    # V24 VND detailed trades
    cols_show = ['entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'exit_reason', 'position_size', 'entry_trend']
    if 'entry_trend' not in vnd_v24.columns:
        cols_show = [c for c in cols_show if c in vnd_v24.columns]
    print("\nV24 ALL VND TRADES:")
    print(vnd_v24[cols_show].to_string(index=False))

    print("\nRULE ALL VND TRADES:")
    rule_cols = [c for c in ['entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'exit_reason'] if c in vnd_rule.columns]
    print(vnd_rule[rule_cols].to_string(index=False))

    # Show other models on VND
    for name in ['v23', 'v19_1', 'v22', 'v16']:
        if name in data:
            sub = data[name][data[name]['symbol'] == 'VND'].sort_values('entry_date')
            if len(sub) > 0:
                print(f"\n{name.upper()} ALL VND TRADES:")
                show_cols = [c for c in ['entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'exit_reason', 'entry_trend'] if c in sub.columns]
                print(sub[show_cols].to_string(index=False))

    # ====================================================================
    # SECTION 4: TRADE-BY-TRADE MATCHING VND (V24 vs RULE)
    # ====================================================================
    print_section("4. VND TRADE MATCHING: V24 vs RULE - Tìm giao dịch rule tốt mà V24 bỏ lỡ/thua")

    for _, rt in vnd_rule.iterrows():
        r_entry = rt['entry_date']
        r_exit = rt['exit_date']
        r_pnl = rt['pnl_pct']

        # Find V24 trades overlapping with this rule trade period
        overlapping = vnd_v24[
            (vnd_v24['entry_date'] <= r_exit) &
            (vnd_v24['exit_date'] >= r_entry)
        ]

        # Find V24 trades entering within ±5 days of rule entry
        nearby = vnd_v24[
            (vnd_v24['entry_date'] >= r_entry - pd.Timedelta(days=5)) &
            (vnd_v24['entry_date'] <= r_entry + pd.Timedelta(days=5))
        ]

        merged = pd.concat([overlapping, nearby]).drop_duplicates()

        if len(merged) == 0:
            if r_pnl > 5:
                print(f"\n  MISSED OPPORTUNITY: Rule {r_entry.date()} → {r_exit.date()}, PnL={r_pnl:.1f}%")
                print(f"    V24 had NO trade during this period!")
        else:
            v24_total = merged['pnl_pct'].sum()
            v24_count = len(merged)
            delta = v24_total - r_pnl
            flag = ""
            if r_pnl > 10 and v24_total < r_pnl * 0.5:
                flag = " *** V24 SIGNIFICANTLY UNDERPERFORMED ***"
            elif r_pnl > 0 and v24_total < 0:
                flag = " *** V24 LOST WHILE RULE WON ***"
            elif v24_count >= 3 and r_pnl > 0:
                flag = " *** V24 NOISY: too many trades ***"

            print(f"\n  Rule: {r_entry.date()} → {r_exit.date()}, PnL={r_pnl:.1f}%")
            print(f"    V24: {v24_count} trades, total PnL={v24_total:.1f}%, delta={delta:+.1f}%{flag}")
            for _, vt in merged.iterrows():
                er = vt.get('exit_reason', '?')
                et = vt.get('entry_trend', '?')
                print(f"      V24: {vt['entry_date'].date()} → {vt['exit_date'].date()}, hold={vt['holding_days']}d, pnl={vt['pnl_pct']:.1f}%, exit={er}, trend={et}")

    # ====================================================================
    # SECTION 5: V24 WORST TRADES ANALYSIS
    # ====================================================================
    print_section("5. V24 WORST TRADES (top 30 lỗ lớn nhất)")

    worst = v24.nsmallest(30, 'pnl_pct')
    cols_worst = ['symbol', 'entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'exit_reason', 'entry_trend', 'position_size']
    cols_worst = [c for c in cols_worst if c in worst.columns]
    print(worst[cols_worst].to_string(index=False))

    # Worst trades on common 14 symbols
    print("\n\nV24 WORST TRADES on 14 COMMON SYMBOLS:")
    worst14 = v24_14.nsmallest(20, 'pnl_pct')
    print(worst14[cols_worst].to_string(index=False))

    # ====================================================================
    # SECTION 6: V24 EXIT REASON ANALYSIS
    # ====================================================================
    print_section("6. V24 EXIT REASON BREAKDOWN")

    for scope, df_scope in [("ALL 367", v24), ("14 COMMON", v24_14)]:
        print(f"\n--- {scope} symbols ---")
        er = df_scope.groupby('exit_reason').agg(
            count=('pnl_pct', 'count'),
            total_pnl=('pnl_pct', 'sum'),
            avg_pnl=('pnl_pct', 'mean'),
            win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100),
            max_loss=('pnl_pct', 'min'),
            max_win=('pnl_pct', 'max')
        ).sort_values('total_pnl', ascending=False)
        print(er.to_string())

    # ====================================================================
    # SECTION 7: V24 NOISY TRADING ANALYSIS
    # ====================================================================
    print_section("7. V24 NOISY TRADING - Symbols với quá nhiều giao dịch và hiệu quả thấp")

    sym_stats = v24.groupby('symbol').agg(
        count=('pnl_pct', 'count'),
        total_pnl=('pnl_pct', 'sum'),
        avg_pnl=('pnl_pct', 'mean'),
        win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100),
        loss_trades=('pnl_pct', lambda x: (x < 0).sum()),
        total_loss=('pnl_pct', lambda x: x[x < 0].sum())
    ).sort_values('total_pnl')

    print("\nTOP 30 WORST PERFORMING SYMBOLS (by total PnL):")
    print(sym_stats.head(30).to_string())

    print("\nTOP 20 MOST TRADED SYMBOLS (potential noise):")
    most_traded = sym_stats.sort_values('count', ascending=False).head(20)
    print(most_traded.to_string())

    # Identify noisy: high trade count but low/negative PnL
    noisy = sym_stats[(sym_stats['count'] >= 15) & (sym_stats['avg_pnl'] < 1)].sort_values('total_pnl')
    print(f"\nNOISY SYMBOLS (≥15 trades, avg PnL < 1%):")
    print(noisy.to_string())

    # ====================================================================
    # SECTION 8: V24 vs V23 TRADE-BY-TRADE on COMMON SYMBOLS
    # ====================================================================
    print_section("8. V24 vs V23/V19.1 TRADE-BY-TRADE trên từng mã chung")

    if 'v23' in data:
        for sym in COMMON_SYMBOLS:
            v24s = v24[v24['symbol'] == sym].sort_values('entry_date')
            v23s = data['v23'][data['v23']['symbol'] == sym].sort_values('entry_date')

            if len(v24s) == 0 and len(v23s) == 0:
                continue

            v24_total = v24s['pnl_pct'].sum()
            v23_total = v23s['pnl_pct'].sum()
            delta = v24_total - v23_total

            sign = "+" if delta >= 0 else ""
            print(f"\n{sym}: V24={v24_total:.1f}% ({len(v24s)} trades) | V23={v23_total:.1f}% ({len(v23s)} trades) | Delta={sign}{delta:.1f}%")

            # Find V23 trades that V24 missed
            for _, t23 in v23s.iterrows():
                nearby_v24 = v24s[
                    (v24s['entry_date'] >= t23['entry_date'] - pd.Timedelta(days=5)) &
                    (v24s['entry_date'] <= t23['entry_date'] + pd.Timedelta(days=5))
                ]
                if len(nearby_v24) == 0 and t23['pnl_pct'] > 10:
                    print(f"  V23 WIN missed by V24: {t23['entry_date'].date()}, pnl={t23['pnl_pct']:.1f}%, exit={t23.get('exit_reason','?')}")

            # Find V24 trades during V23 trade periods that performed worse
            for _, t23 in v23s.iterrows():
                nearby_v24 = v24s[
                    (v24s['entry_date'] >= t23['entry_date'] - pd.Timedelta(days=3)) &
                    (v24s['entry_date'] <= t23['entry_date'] + pd.Timedelta(days=3))
                ]
                if len(nearby_v24) > 0:
                    v24_sum = nearby_v24['pnl_pct'].sum()
                    if t23['pnl_pct'] > 5 and v24_sum < t23['pnl_pct'] * 0.5:
                        print(f"  V24 UNDERPERFORMED: date={t23['entry_date'].date()}, V23={t23['pnl_pct']:.1f}%, V24={v24_sum:.1f}% ({len(nearby_v24)} trades)")
                    elif t23['pnl_pct'] > 0 and v24_sum < -5:
                        print(f"  V24 LOST while V23 WON: date={t23['entry_date'].date()}, V23={t23['pnl_pct']:.1f}%, V24={v24_sum:.1f}%")

    # ====================================================================
    # SECTION 9: V24 YEARLY PERFORMANCE
    # ====================================================================
    print_section("9. V24 HIỆU SUẤT THEO NĂM")

    v24['year'] = v24['entry_date'].dt.year
    yearly = v24.groupby('year').agg(
        trades=('pnl_pct', 'count'),
        total_pnl=('pnl_pct', 'sum'),
        avg_pnl=('pnl_pct', 'mean'),
        win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100),
        max_loss=('pnl_pct', 'min')
    )
    print("\nALL symbols:")
    print(yearly.to_string())

    v24_14['year'] = v24_14['entry_date'].dt.year
    yearly14 = v24_14.groupby('year').agg(
        trades=('pnl_pct', 'count'),
        total_pnl=('pnl_pct', 'sum'),
        avg_pnl=('pnl_pct', 'mean'),
        win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100),
        max_loss=('pnl_pct', 'min')
    )
    print("\n14 COMMON symbols:")
    print(yearly14.to_string())

    # Compare yearly with other models
    print("\nYEARLY COMPARISON (14 common symbols):")
    for name in ['v23', 'v19_1', 'v22', 'rule']:
        if name in data:
            df = data[name]
            df['year'] = pd.to_datetime(df['entry_date']).dt.year
            yrly = df.groupby('year')['pnl_pct'].agg(['sum','count','mean'])
            yrly.columns = [f'{name}_total', f'{name}_count', f'{name}_avg']
            print(f"\n{name.upper()}:")
            print(yrly.to_string())

    # ====================================================================
    # SECTION 10: V24 TREND PERFORMANCE
    # ====================================================================
    print_section("10. V24 PERFORMANCE BY ENTRY TREND")

    if 'entry_trend' in v24.columns:
        for scope, df_scope in [("ALL", v24), ("14 COMMON", v24_14)]:
            print(f"\n--- {scope} ---")
            trend_stats = df_scope.groupby('entry_trend').agg(
                trades=('pnl_pct', 'count'),
                total_pnl=('pnl_pct', 'sum'),
                avg_pnl=('pnl_pct', 'mean'),
                win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100),
                max_loss=('pnl_pct', 'min')
            )
            print(trend_stats.to_string())

    # ====================================================================
    # SECTION 11: BIG WINNERS / LOSERS COMPARISON
    # ====================================================================
    print_section("11. BIG WINNERS CỦA TỪNG MODEL mà MODEL KHÁC KHÔNG CÓ")

    for name, df in data.items():
        if name == 'v24':
            continue
        big_wins = df[df['pnl_pct'] > 20].copy()
        if len(big_wins) == 0:
            continue

        print(f"\n--- {name.upper()} big wins (>20%) ---")
        for _, bw in big_wins.iterrows():
            sym = bw['symbol'] if 'symbol' in bw.index else bw.get('entry_symbol', '?')
            entry = bw['entry_date']

            # Check if V24 captured this
            v24_nearby = v24[
                (v24['symbol'] == sym) &
                (v24['entry_date'] >= entry - pd.Timedelta(days=5)) &
                (v24['entry_date'] <= entry + pd.Timedelta(days=5))
            ]

            if len(v24_nearby) == 0:
                print(f"  {sym} {entry.date()}: {name}={bw['pnl_pct']:.1f}% | V24=MISSED")
            else:
                v24_best = v24_nearby['pnl_pct'].max()
                v24_sum = v24_nearby['pnl_pct'].sum()
                status = "OK" if v24_sum >= bw['pnl_pct'] * 0.7 else "WEAK"
                print(f"  {sym} {entry.date()}: {name}={bw['pnl_pct']:.1f}% | V24 best={v24_best:.1f}% sum={v24_sum:.1f}% ({len(v24_nearby)} trades) [{status}]")

    # ====================================================================
    # SECTION 12: V24 UNIQUE STRENGTHS
    # ====================================================================
    print_section("12. V24 UNIQUE STRENGTHS - Trades tốt mà chỉ V24 có (ngoài 14 mã)")

    v24_non14 = v24[~v24['symbol'].isin(COMMON_SYMBOLS)]
    print(f"\nV24 trades outside 14 common symbols: {len(v24_non14)}")
    print(f"Total PnL from non-common symbols: {v24_non14['pnl_pct'].sum():.1f}%")

    non14_sym = v24_non14.groupby('symbol').agg(
        trades=('pnl_pct', 'count'),
        total_pnl=('pnl_pct', 'sum'),
        avg_pnl=('pnl_pct', 'mean'),
        win_rate=('pnl_pct', lambda x: (x > 0).mean() * 100)
    ).sort_values('total_pnl', ascending=False)

    print("\nTOP 20 BEST non-common symbols:")
    print(non14_sym.head(20).to_string())

    print("\nTOP 20 WORST non-common symbols:")
    print(non14_sym.tail(20).to_string())

    # ====================================================================
    # SECTION 13: SHORT HOLDING PERIOD ANALYSIS (NOISE)
    # ====================================================================
    print_section("13. SHORT HOLDING ANALYSIS - Trades giữ ≤3 ngày (potential noise)")

    short = v24[v24['holding_days'] <= 3]
    print(f"\nShort trades (≤3d): {len(short)} / {len(v24)} = {len(short)/len(v24)*100:.1f}%")
    print(f"Short trades total PnL: {short['pnl_pct'].sum():.1f}%")
    print(f"Short trades avg PnL: {short['pnl_pct'].mean():.2f}%")
    print(f"Short trades WR: {(short['pnl_pct']>0).mean()*100:.1f}%")

    short14 = v24_14[v24_14['holding_days'] <= 3]
    print(f"\nShort trades on 14 common (≤3d): {len(short14)} / {len(v24_14)} = {len(short14)/len(v24_14)*100:.1f}%")
    print(f"Short trades total PnL: {short14['pnl_pct'].sum():.1f}%")

    # ====================================================================
    # SECTION 14: SPECIFIC PROBLEM AREAS
    # ====================================================================
    print_section("14. V24 PROBLEM AREAS - Chi tiết từng mã có vấn đề")

    # Identify symbols where V24 significantly underperforms rule
    problem_syms = []
    for sym in COMMON_SYMBOLS:
        v24s = v24[v24['symbol'] == sym]
        rs = rule[rule['symbol'] == sym]
        if len(rs) > 0:
            v24_tot = v24s['pnl_pct'].sum()
            r_tot = rs['pnl_pct'].sum()
            delta = v24_tot - r_tot
            if delta < -20:
                problem_syms.append((sym, v24_tot, r_tot, delta, len(v24s), len(rs)))

    if problem_syms:
        print("\nSymbols where V24 significantly underperforms RULE (delta < -20%):")
        for sym, v24t, rt, delta, v24n, rn in sorted(problem_syms, key=lambda x: x[3]):
            print(f"  {sym}: V24={v24t:.1f}% ({v24n} trades) vs Rule={rt:.1f}% ({rn} trades), Delta={delta:.1f}%")

            # Show detailed V24 trades on this symbol
            v24_sym = v24[v24['symbol'] == sym].sort_values('entry_date')
            rule_sym = rule[rule['symbol'] == sym].sort_values('entry_date')

            print(f"    V24 trades:")
            for _, t in v24_sym.iterrows():
                er = t.get('exit_reason', '?')
                et = t.get('entry_trend', '?')
                print(f"      {t['entry_date'].date()} → {t['exit_date'].date()}, hold={t['holding_days']}d, pnl={t['pnl_pct']:.1f}%, exit={er}, trend={et}, size={t.get('position_size','?')}")

            print(f"    Rule trades:")
            for _, t in rule_sym.iterrows():
                print(f"      {t['entry_date'].date()} → {t['exit_date'].date()}, hold={t['holding_days']}d, pnl={t['pnl_pct']:.1f}%, exit={t.get('exit_reason','?')}")

    # ====================================================================
    # SECTION 15: SUMMARY AND ROOT CAUSE
    # ====================================================================
    print_section("15. ROOT CAUSE SUMMARY & IMPROVEMENT IDEAS")

    v24_14_total = v24_14['pnl_pct'].sum()
    v24_all_total = v24['pnl_pct'].sum()

    print(f"""
KEY FINDINGS:
  V24 Total PnL (ALL 367 symbols): {v24_all_total:.1f}%
  V24 Total PnL (14 common symbols): {v24_14_total:.1f}%
  V24 trades count (ALL): {len(v24)}
  V24 trades count (14 common): {len(v24_14)}

  V23 Total PnL (14 symbols): {data.get('v23', pd.DataFrame({'pnl_pct':[]}))['pnl_pct'].sum():.1f}%
  V19.1 Total PnL (14 symbols): {data.get('v19_1', pd.DataFrame({'pnl_pct':[]}))['pnl_pct'].sum():.1f}%
  Rule Total PnL (14 symbols): {rule['pnl_pct'].sum():.1f}%
""")

    # Identify specific issues
    v24_losses = v24[v24['pnl_pct'] < -10]
    print(f"  V24 trades with >10% loss: {len(v24_losses)}")
    print(f"  V24 total from those losses: {v24_losses['pnl_pct'].sum():.1f}%")

    if 'exit_reason' in v24.columns:
        hc = v24[v24['exit_reason'].str.contains('hard_cap', na=False)]
        print(f"  V24 hard_cap trades: {len(hc)}, total PnL: {hc['pnl_pct'].sum():.1f}%")

    print("\nDONE - Analysis complete")


if __name__ == "__main__":
    main()
