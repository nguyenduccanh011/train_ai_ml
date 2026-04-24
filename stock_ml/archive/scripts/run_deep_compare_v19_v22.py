"""Deep comparison: V19.1 vs V19.3 vs V22-Final vs Rule."""
import sys, os, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_v19_3_compare import backtest_v19_3, run_test, run_rule_test, calc_metrics
from run_v19_1_compare import backtest_v19_1
from run_v22_final import backtest_v22

SYMBOLS = 'ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC'

print('Running V19.1...')
t_v191 = run_test(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_1)
m_v191 = calc_metrics(t_v191)

print('Running V19.3...')
t_v193 = run_test(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=backtest_v19_3)
m_v193 = calc_metrics(t_v193)

print('Running V22 Final...')
def v22_fn(y_pred, returns, df_test, feature_cols, **kwargs):
    return backtest_v22(y_pred, returns, df_test, feature_cols, **kwargs)
t_v22 = run_test(SYMBOLS, True, True, False, False, True, True, True, True, True, True, backtest_fn=v22_fn)
m_v22 = calc_metrics(t_v22)

print('Running Rule...')
t_rule = run_rule_test(SYMBOLS)
m_rule = calc_metrics(t_rule)

print()
print('='*120)
print('OVERALL COMPARISON: V19.1 vs V19.3 vs V22-Final vs Rule')
print('='*120)
for lbl, m in [('V19.1', m_v191), ('V19.3', m_v193), ('V22-Final', m_v22), ('Rule', m_rule)]:
    print(f'  {lbl:<12} | #{m["trades"]:>4} WR={m["wr"]:>5.1f}% AvgPnL={m["avg_pnl"]:>+7.2f}% '
          f'TotPnL={m["total_pnl"]:>+9.1f}% PF={m["pf"]:>5.2f} MaxLoss={m["max_loss"]:>+7.1f}% '
          f'AvgHold={m["avg_hold"]:>5.1f}d')

# Per-symbol
print()
print('PER-SYMBOL COMPARISON')
print('='*120)
df_191 = pd.DataFrame(t_v191)
df_193 = pd.DataFrame(t_v193)
df_22 = pd.DataFrame(t_v22)
df_rule = pd.DataFrame(t_rule)

syms = sorted(set(df_191['symbol'].unique()) | set(df_22['symbol'].unique()))
print(f"{'Sym':<6}| {'V19.1':>9} | {'V19.3':>9} | {'V22':>9} | {'Rule':>9} | "
      f"{'V22-V191':>9} | {'V22-V193':>9} | {'V22-Rule':>9}")
print('-'*100)
totals = {'v191':0, 'v193':0, 'v22':0, 'rule':0}
for sym in syms:
    m1 = calc_metrics(df_191[df_191['symbol']==sym].to_dict('records'))
    m3 = calc_metrics(df_193[df_193['symbol']==sym].to_dict('records'))
    m22 = calc_metrics(df_22[df_22['symbol']==sym].to_dict('records'))
    mr = calc_metrics(df_rule[df_rule['symbol']==sym].to_dict('records'))
    totals['v191']+=m1['total_pnl']; totals['v193']+=m3['total_pnl']
    totals['v22']+=m22['total_pnl']; totals['rule']+=mr['total_pnl']
    print(f"{sym:<6}| {m1['total_pnl']:>+8.1f}% | {m3['total_pnl']:>+8.1f}% | "
          f"{m22['total_pnl']:>+8.1f}% | {mr['total_pnl']:>+8.1f}% | "
          f"{m22['total_pnl']-m1['total_pnl']:>+8.1f}% | "
          f"{m22['total_pnl']-m3['total_pnl']:>+8.1f}% | "
          f"{m22['total_pnl']-mr['total_pnl']:>+8.1f}%")
print('-'*100)
print(f"{'TOTAL':<6}| {totals['v191']:>+8.1f}% | {totals['v193']:>+8.1f}% | "
      f"{totals['v22']:>+8.1f}% | {totals['rule']:>+8.1f}% | "
      f"{totals['v22']-totals['v191']:>+8.1f}% | "
      f"{totals['v22']-totals['v193']:>+8.1f}% | "
      f"{totals['v22']-totals['rule']:>+8.1f}%")

# Exit Reason breakdown
print()
print('EXIT REASON BREAKDOWN')
print('='*120)
for lbl, df in [('V19.1', df_191), ('V19.3', df_193), ('V22-Final', df_22)]:
    print(f'\n  {lbl}:')
    for reason, grp in df.groupby('exit_reason'):
        wins = len(grp[grp['pnl_pct']>0])
        print(f'    {reason:<25}: {len(grp):>4} trades, WR={wins/len(grp)*100:>5.1f}%, '
              f'avg={grp["pnl_pct"].mean():>+6.2f}%, total={grp["pnl_pct"].sum():>+8.1f}%')

# Worst trades
print()
print('WORST 15 TRADES PER MODEL')
print('='*120)
for lbl, df in [('V19.1', df_191), ('V19.3', df_193), ('V22-Final', df_22)]:
    print(f'\n  {lbl} - worst 15:')
    worst = df.nsmallest(15, 'pnl_pct')
    for _, t in worst.iterrows():
        sym = t.get("symbol", "?")
        ed = t.get("entry_date", "?")
        xd = t.get("exit_date", "?")
        pnl = t["pnl_pct"]
        hd = t.get("holding_days", 0)
        er = t["exit_reason"]
        ps = t.get("position_size", 1.0)
        tr = t.get("entry_trend", "?")
        print(f'    {sym:>5} {ed:>12} -> {xd:>12} | PnL={pnl:>+7.2f}% | hold={hd:>3}d | '
              f'exit={er:>20} | size={ps:>4.2f} | trend={tr:>8}')

# Best trades
print()
print('BEST 15 TRADES PER MODEL')
print('='*120)
for lbl, df in [('V19.1', df_191), ('V19.3', df_193), ('V22-Final', df_22)]:
    print(f'\n  {lbl} - best 15:')
    best = df.nlargest(15, 'pnl_pct')
    for _, t in best.iterrows():
        sym = t.get("symbol", "?")
        ed = t.get("entry_date", "?")
        xd = t.get("exit_date", "?")
        pnl = t["pnl_pct"]
        hd = t.get("holding_days", 0)
        er = t["exit_reason"]
        ps = t.get("position_size", 1.0)
        tr = t.get("entry_trend", "?")
        print(f'    {sym:>5} {ed:>12} -> {xd:>12} | PnL={pnl:>+7.2f}% | hold={hd:>3}d | '
              f'exit={er:>20} | size={ps:>4.2f} | trend={tr:>8}')

# Trade divergence
print()
print('TRADE DIVERGENCE: V22 vs V19.3')
print('='*120)

def find_corresponding(row, other_df):
    sym_trades = other_df[other_df['symbol']==row['symbol']]
    if len(sym_trades) == 0:
        return None
    if 'entry_date' in sym_trades.columns and 'entry_date' in row.index:
        sym_trades = sym_trades.copy()
        sym_trades['date_diff'] = abs(pd.to_datetime(sym_trades['entry_date']) - pd.to_datetime(row['entry_date']))
        closest = sym_trades.loc[sym_trades['date_diff'].idxmin()]
        if closest['date_diff'].days <= 5:
            return closest
    return None

print('\n  V22 much BETTER than V19.3 (diff > +5%):')
count = 0
for _, t22 in df_22.iterrows():
    t193 = find_corresponding(t22, df_193)
    if t193 is not None and (t22['pnl_pct'] - t193['pnl_pct']) > 5:
        count += 1
        if count <= 10:
            print(f'    {t22["symbol"]:>5} entry={t22.get("entry_date","?"):>12} | '
                  f'V22={t22["pnl_pct"]:>+7.2f}% exit={t22["exit_reason"]:>20} | '
                  f'V19.3={t193["pnl_pct"]:>+7.2f}% exit={t193["exit_reason"]:>20} | '
                  f'diff={t22["pnl_pct"]-t193["pnl_pct"]:>+7.2f}%')
print(f'  Total: {count} trades where V22 >> V19.3')

print('\n  V22 much WORSE than V19.3 (diff < -5%):')
count = 0
for _, t22 in df_22.iterrows():
    t193 = find_corresponding(t22, df_193)
    if t193 is not None and (t22['pnl_pct'] - t193['pnl_pct']) < -5:
        count += 1
        if count <= 10:
            print(f'    {t22["symbol"]:>5} entry={t22.get("entry_date","?"):>12} | '
                  f'V22={t22["pnl_pct"]:>+7.2f}% exit={t22["exit_reason"]:>20} | '
                  f'V19.3={t193["pnl_pct"]:>+7.2f}% exit={t193["exit_reason"]:>20} | '
                  f'diff={t22["pnl_pct"]-t193["pnl_pct"]:>+7.2f}%')
print(f'  Total: {count} trades where V22 << V19.3')

# BY YEAR
print()
print('BY YEAR COMPARISON')
print('='*120)
for lbl, df in [('V19.1', df_191), ('V19.3', df_193), ('V22', df_22), ('Rule', df_rule)]:
    df_c = df.copy()
    if 'entry_date' in df_c.columns:
        df_c['year'] = df_c['entry_date'].astype(str).str[:4]
    elif 'exit_date' in df_c.columns:
        df_c['year'] = df_c['exit_date'].astype(str).str[:4]
    else:
        continue
    print(f'  {lbl}:')
    for yr in sorted(df_c['year'].unique()):
        m = calc_metrics(df_c[df_c['year']==yr].to_dict('records'))
        print(f'    {yr}: #{m["trades"]:>3} WR={m["wr"]:>5.1f}% avg={m["avg_pnl"]:>+6.2f}% '
              f'tot={m["total_pnl"]:>+8.1f}% PF={m["pf"]:>5.2f}')

# By entry_trend
print()
print('BY ENTRY TREND')
print('='*120)
for lbl, df in [('V19.1', df_191), ('V19.3', df_193), ('V22', df_22)]:
    if 'entry_trend' not in df.columns:
        continue
    print(f'  {lbl}:')
    for trend in ['strong', 'moderate', 'weak']:
        tt = df[df['entry_trend']==trend]
        if len(tt)==0: continue
        m = calc_metrics(tt.to_dict('records'))
        print(f'    {trend:<10}: #{m["trades"]:>3} WR={m["wr"]:>5.1f}% avg={m["avg_pnl"]:>+6.2f}% '
              f'tot={m["total_pnl"]:>+8.1f}% PF={m["pf"]:>5.2f}')

# fast_exit_loss analysis
print()
print('FAST_EXIT_LOSS DEEP ANALYSIS')
print('='*120)
for lbl, df in [('V19.3', df_193), ('V22-Final', df_22)]:
    fel = df[df['exit_reason']=='fast_exit_loss']
    if len(fel) == 0:
        print(f'  {lbl}: no fast_exit_loss trades')
        continue
    print(f'\n  {lbl}: {len(fel)} fast_exit_loss trades')
    print(f'    Total PnL: {fel["pnl_pct"].sum():>+8.1f}%')
    print(f'    Avg PnL: {fel["pnl_pct"].mean():>+6.2f}%')
    print(f'    Worst: {fel["pnl_pct"].min():>+6.2f}%')
    if 'entry_trend' in fel.columns:
        for trend in ['strong', 'moderate', 'weak']:
            tt = fel[fel['entry_trend']==trend]
            if len(tt)==0: continue
            print(f'    trend={trend}: {len(tt)} trades, avg={tt["pnl_pct"].mean():>+6.2f}%, total={tt["pnl_pct"].sum():>+8.1f}%')
    if 'symbol' in fel.columns:
        print(f'    By symbol:')
        for sym in sorted(fel['symbol'].unique()):
            st = fel[fel['symbol']==sym]
            print(f'      {sym}: {len(st)} trades, total={st["pnl_pct"].sum():>+8.1f}%, avg={st["pnl_pct"].mean():>+6.2f}%')

print()
print('DONE')
