"""Detailed comparison of all active models."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import src.safe_io  # noqa: F401 — fix UnicodeEncodeError on Windows console

import pandas as pd
import numpy as np

versions = ['v26', 'v25', 'v24', 'v23', 'v22', 'v19_1', 'rule']
names = {
    'v26': 'V26 Refined',
    'v25': 'V25 Ablation',
    'v24': 'V24 Adaptive',
    'v23': 'V23 Optimal',
    'v22': 'V22 Final',
    'v19_1': 'V19.1 ML',
    'rule': 'Rule-Based',
}

dfs = {}
for v in versions:
    dfs[v] = pd.read_csv(f'results/trades_{v}.csv')

W = 140

print('=' * W)
print('CHI TIẾT SO SÁNH CÁC MODEL ACTIVE')
print('=' * W)

# ===== 1. TỔNG QUAN =====
print()
print('1. TỔNG QUAN HIỆU SUẤT')
print('-' * W)

def print_row(label, values):
    row = f'{label:<30}'
    for val in values:
        row += f' | {str(val):>18}'
    print(row)

header = f'{"Metric":<30}'
for v in versions:
    header += f' | {names[v]:>18}'
print(header)
print('-' * W)

def safe_pf(df):
    gp = df.loc[df['pnl_pct'] > 0, 'pnl_pct'].sum()
    gl = abs(df.loc[df['pnl_pct'] <= 0, 'pnl_pct'].sum())
    return gp / gl if gl > 0 else 99.0

rows = [
    ('Tổng trades', [len(dfs[v]) for v in versions]),
    ('Win Rate (%)', [f"{(dfs[v]['pnl_pct']>0).sum()/len(dfs[v])*100:.1f}" for v in versions]),
    ('Avg PnL/trade (%)', [f"{dfs[v]['pnl_pct'].mean():+.2f}" for v in versions]),
    ('Median PnL/trade (%)', [f"{dfs[v]['pnl_pct'].median():+.2f}" for v in versions]),
    ('Total PnL (%)', [f"{dfs[v]['pnl_pct'].sum():+.1f}" for v in versions]),
    ('Profit Factor', [f"{safe_pf(dfs[v]):.2f}" for v in versions]),
    ('Gross Profit (%)', [f"{dfs[v].loc[dfs[v]['pnl_pct']>0,'pnl_pct'].sum():+.1f}" for v in versions]),
    ('Gross Loss (%)', [f"{dfs[v].loc[dfs[v]['pnl_pct']<=0,'pnl_pct'].sum():+.1f}" for v in versions]),
    ('Max Win (%)', [f"{dfs[v]['pnl_pct'].max():+.1f}" for v in versions]),
    ('Max Loss (%)', [f"{dfs[v]['pnl_pct'].min():+.1f}" for v in versions]),
    ('Avg Hold (days)', [f"{dfs[v]['holding_days'].mean():.1f}" if 'holding_days' in dfs[v].columns else 'N/A' for v in versions]),
    ('Std PnL (%)', [f"{dfs[v]['pnl_pct'].std():.2f}" for v in versions]),
    ('Sharpe (PnL)', [f"{dfs[v]['pnl_pct'].mean()/dfs[v]['pnl_pct'].std():.3f}" if dfs[v]['pnl_pct'].std()>0 else 'N/A' for v in versions]),
]

for label, vals in rows:
    print_row(label, vals)

# ===== 2. WIN/LOSS DISTRIBUTION =====
print()
print('2. PHÂN BỔ WIN/LOSS')
print('-' * W)

buckets = [
    ('Big Win (>20%)', lambda df: (df['pnl_pct'] > 20).sum()),
    ('Win (5-20%)', lambda df: ((df['pnl_pct'] > 5) & (df['pnl_pct'] <= 20)).sum()),
    ('Small Win (0-5%)', lambda df: ((df['pnl_pct'] > 0) & (df['pnl_pct'] <= 5)).sum()),
    ('Breakeven (0%)', lambda df: (df['pnl_pct'] == 0).sum()),
    ('Small Loss (0 to -5%)', lambda df: ((df['pnl_pct'] < 0) & (df['pnl_pct'] >= -5)).sum()),
    ('Loss (-5 to -20%)', lambda df: ((df['pnl_pct'] < -5) & (df['pnl_pct'] >= -20)).sum()),
    ('Big Loss (<-20%)', lambda df: (df['pnl_pct'] < -20).sum()),
]

header = f'{"Bucket":<30}'
for v in versions:
    header += f' | {names[v]:>18}'
print(header)
print('-' * W)

for label, func in buckets:
    vals = []
    for v in versions:
        cnt = func(dfs[v])
        pct = cnt / len(dfs[v]) * 100
        vals.append(f"{cnt:>6} ({pct:4.1f}%)")
    print_row(label, vals)

# ===== 3. EXIT REASON =====
print()
print('3. EXIT REASON BREAKDOWN')
print('-' * W)

all_reasons = set()
for v in versions:
    if 'exit_reason' in dfs[v].columns:
        all_reasons.update(dfs[v]['exit_reason'].dropna().unique())
all_reasons = sorted(all_reasons)

header = f'{"Exit Reason":<30}'
for v in versions:
    header += f' | {names[v]:>18}'
print(header)
print('-' * W)

for reason in all_reasons:
    vals = []
    for v in versions:
        if 'exit_reason' in dfs[v].columns:
            cnt = (dfs[v]['exit_reason'] == reason).sum()
            pct = cnt / len(dfs[v]) * 100
            vals.append(f"{cnt:>6} ({pct:4.1f}%)")
        else:
            vals.append("N/A")
    print_row(str(reason)[:29], vals)

# ===== 4. EXIT REASON PnL =====
print()
print('4. AVG PnL BY EXIT REASON')
print('-' * W)

header = f'{"Exit Reason":<30}'
for v in versions:
    header += f' | {names[v]:>18}'
print(header)
print('-' * W)

for reason in all_reasons:
    vals = []
    for v in versions:
        if 'exit_reason' in dfs[v].columns:
            sub = dfs[v][dfs[v]['exit_reason'] == reason]
            if len(sub) > 0:
                avg = sub['pnl_pct'].mean()
                vals.append(f"{avg:+.2f}%")
            else:
                vals.append("-")
        else:
            vals.append("N/A")
    print_row(str(reason)[:29], vals)

# ===== 5. ENTRY TREND =====
print()
print('5. ENTRY TREND BREAKDOWN')
print('-' * W)

for v in versions:
    df = dfs[v]
    if 'entry_trend' not in df.columns:
        print(f'  {names[v]}: no entry_trend column')
        continue
    print(f'  {names[v]}:')
    for trend in sorted(df['entry_trend'].dropna().unique()):
        sub = df[df['entry_trend'] == trend]
        wr = (sub['pnl_pct'] > 0).sum() / len(sub) * 100 if len(sub) > 0 else 0
        avg = sub['pnl_pct'].mean()
        total = sub['pnl_pct'].sum()
        print(f'    {trend:<15}: {len(sub):>5} trades, WR={wr:5.1f}%, AvgPnL={avg:+.2f}%, TotalPnL={total:+.1f}%')

# ===== 6. PER-YEAR BREAKDOWN =====
print()
print('6. PER-YEAR PERFORMANCE')
print('-' * W)

for v in versions:
    df = dfs[v]
    if 'entry_date' in df.columns:
        df['year'] = pd.to_datetime(df['entry_date']).dt.year

header = f'{"Year":<8}'
for v in versions:
    header += f' |  {names[v]:>16}'
print(header)
print('-' * W)

all_years = sorted(set().union(*[dfs[v]['year'].unique() for v in versions if 'year' in dfs[v].columns]))
for yr in all_years:
    row = f'{yr:<8}'
    for v in versions:
        df = dfs[v]
        if 'year' in df.columns:
            sub = df[df['year'] == yr]
            if len(sub) > 0:
                wr = (sub['pnl_pct'] > 0).sum() / len(sub) * 100
                total = sub['pnl_pct'].sum()
                n = len(sub)
                row += f' | {total:>+7.0f}% ({n:>4}t)'
            else:
                row += f' |         no data'
        else:
            row += f' |             N/A'
    print(row)

# ===== 7. TOP/BOTTOM SYMBOLS =====
print()
print('7. TOP 10 & BOTTOM 10 SYMBOLS (by V25 Total PnL)')
print('-' * W)

sym_col = 'symbol' if 'symbol' in dfs['v25'].columns else 'entry_symbol'
sym_pnl = dfs['v25'].groupby(sym_col)['pnl_pct'].sum().sort_values(ascending=False)

print('  TOP 10:')
header = f'  {"Symbol":<8}'
for v in versions:
    header += f' | {names[v]:>14}'
print(header)

for sym in sym_pnl.head(10).index:
    row = f'  {sym:<8}'
    for v in versions:
        df = dfs[v]
        sc = 'symbol' if 'symbol' in df.columns else 'entry_symbol'
        sub = df[df[sc] == sym]
        total = sub['pnl_pct'].sum() if len(sub) > 0 else 0
        row += f' | {total:>+13.1f}%'
    print(row)

print()
print('  BOTTOM 10:')
header = f'  {"Symbol":<8}'
for v in versions:
    header += f' | {names[v]:>14}'
print(header)

for sym in sym_pnl.tail(10).index:
    row = f'  {sym:<8}'
    for v in versions:
        df = dfs[v]
        sc = 'symbol' if 'symbol' in df.columns else 'entry_symbol'
        sub = df[df[sc] == sym]
        total = sub['pnl_pct'].sum() if len(sub) > 0 else 0
        row += f' | {total:>+13.1f}%'
    print(row)

# ===== 8. SPECIAL ENTRIES =====
print()
print('8. SPECIAL ENTRY TYPES')
print('-' * W)

for col, label in [('vshape_entry', 'V-shape Entry'), ('breakout_entry', 'Breakout Entry'), ('quick_reentry', 'Quick Re-entry')]:
    row = f'{label:<25}'
    for v in versions:
        df = dfs[v]
        if col in df.columns:
            mask = df[col].fillna(False).astype(bool)
            cnt = mask.sum()
            pct = cnt / len(df) * 100
            sub = df[mask]
            avg = sub['pnl_pct'].mean() if len(sub) > 0 else 0
            row += f' | {cnt:>5} ({pct:4.1f}%) avg={avg:+.1f}%'
        else:
            row += f' | {"N/A":>24}'
    print(row)

# ===== 9. DRAWDOWN SIMULATION =====
print()
print('9. DRAWDOWN ANALYSIS (Sequential capital simulation)')
print('-' * W)

for v in versions:
    df = dfs[v].sort_values('entry_date' if 'entry_date' in dfs[v].columns else 'exit_date')
    capital = 100_000_000  # 100M VND
    equity = [capital]
    for _, trade in df.iterrows():
        pnl_pct = trade['pnl_pct'] / 100
        pos_size = trade.get('position_size', 1.0)
        if pd.isna(pos_size):
            pos_size = 1.0
        change = equity[-1] * pnl_pct * pos_size
        equity.append(equity[-1] + change)

    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100
    max_dd = dd.min()
    final_equity = equity[-1]
    total_return = (final_equity / capital - 1) * 100

    print(f'  {names[v]:<20}: Final={final_equity/1e6:>10.1f}M  Return={total_return:>+8.1f}%  MaxDD={max_dd:>+7.2f}%')

# ===== 10. CONSISTENCY (% profitable symbols) =====
print()
print('10. CONSISTENCY: % SYMBOLS PROFITABLE')
print('-' * W)

for v in versions:
    df = dfs[v]
    sc = 'symbol' if 'symbol' in df.columns else 'entry_symbol'
    sym_pnl = df.groupby(sc)['pnl_pct'].sum()
    profitable = (sym_pnl > 0).sum()
    total_syms = len(sym_pnl)
    pct = profitable / total_syms * 100 if total_syms > 0 else 0
    avg_sym_pnl = sym_pnl.mean()
    med_sym_pnl = sym_pnl.median()
    print(f'  {names[v]:<20}: {profitable}/{total_syms} symbols profitable ({pct:.1f}%), AvgSymPnL={avg_sym_pnl:+.1f}%, MedianSymPnL={med_sym_pnl:+.1f}%')

print()
print('=' * W)
print('KẾT LUẬN & KHUYẾN NGHỊ')
print('=' * W)

# Find best model per metric
best_wr = max(versions, key=lambda v: (dfs[v]['pnl_pct']>0).sum()/len(dfs[v]))
best_pnl = max(versions, key=lambda v: dfs[v]['pnl_pct'].sum())
best_pf = max(versions, key=lambda v: safe_pf(dfs[v]))
best_sharpe = max(versions, key=lambda v: dfs[v]['pnl_pct'].mean()/dfs[v]['pnl_pct'].std() if dfs[v]['pnl_pct'].std()>0 else -99)
least_maxloss = max(versions, key=lambda v: dfs[v]['pnl_pct'].min())

print(f'  Best Win Rate:      {names[best_wr]} ({(dfs[best_wr]["pnl_pct"]>0).sum()/len(dfs[best_wr])*100:.1f}%)')
print(f'  Best Total PnL:     {names[best_pnl]} ({dfs[best_pnl]["pnl_pct"].sum():+.1f}%)')
print(f'  Best Profit Factor: {names[best_pf]} ({safe_pf(dfs[best_pf]):.2f})')
print(f'  Best Sharpe:        {names[best_sharpe]} ({dfs[best_sharpe]["pnl_pct"].mean()/dfs[best_sharpe]["pnl_pct"].std():.3f})')
print(f'  Smallest Max Loss:  {names[least_maxloss]} ({dfs[least_maxloss]["pnl_pct"].min():+.1f}%)')

print()
print('  ML vs Rule:')
ml_avg_pnl = np.mean([dfs[v]['pnl_pct'].mean() for v in versions if v != 'rule'])
rule_avg_pnl = dfs['rule']['pnl_pct'].mean()
print(f'    ML models avg PnL/trade:  {ml_avg_pnl:+.2f}%')
print(f'    Rule avg PnL/trade:       {rule_avg_pnl:+.2f}%')
print(f'    ML advantage:             {ml_avg_pnl - rule_avg_pnl:+.2f}% per trade')

ml_avg_wr = np.mean([(dfs[v]['pnl_pct']>0).sum()/len(dfs[v])*100 for v in versions if v != 'rule'])
rule_wr = (dfs['rule']['pnl_pct']>0).sum()/len(dfs['rule'])*100
print(f'    ML models avg WR:         {ml_avg_wr:.1f}%')
print(f'    Rule WR:                  {rule_wr:.1f}%')

print()
print('=' * W)
