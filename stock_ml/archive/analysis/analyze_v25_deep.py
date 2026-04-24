import pandas as pd
import numpy as np
import json, os, sys, warnings
warnings.filterwarnings('ignore')

RESULTS = r"c:\Users\DUC CANH PC\Desktop\train_ai_ml\stock_ml\results"

def load_trades(fname, model_name):
    fp = os.path.join(RESULTS, fname)
    df = pd.read_csv(fp)
    if 'model' not in df.columns:
        df['model'] = model_name
    if 'symbol' not in df.columns and 'entry_symbol' in df.columns:
        df['symbol'] = df['entry_symbol']
    return df

models = {}
model_files = {
    'v25': 'trades_v25.csv',
    'v24': 'trades_v24.csv',
    'v23': 'trades_v23.csv',
    'v22': 'trades_v22.csv',
    'v19_1': 'trades_v19_1.csv',
    'v19_3': 'trades_v19_3.csv',
    'v16': 'trades_v16.csv',
    'rule': 'trades_rule.csv',
}
for name, fname in model_files.items():
    try:
        models[name] = load_trades(fname, name)
        models[name]['model'] = name
    except Exception as e:
        print(f"Skip {name}: {e}")

print("="*100)
print("PART 1: TONG QUAN HIEU SUAT TUNG MODEL")
print("="*100)

summary_rows = []
for name, df in models.items():
    n_trades = len(df)
    n_symbols = df['symbol'].nunique()
    avg_pnl = df['pnl_pct'].mean()
    median_pnl = df['pnl_pct'].median()
    total_pnl = df['pnl_pct'].sum()
    win_rate = (df['pnl_pct'] > 0).mean() * 100
    avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if (df['pnl_pct'] > 0).any() else 0
    avg_loss = df[df['pnl_pct'] <= 0]['pnl_pct'].mean() if (df['pnl_pct'] <= 0).any() else 0
    max_gain = df['pnl_pct'].max()
    max_loss = df['pnl_pct'].min()
    loss_sum = df[df['pnl_pct']<0]['pnl_pct'].sum()
    profit_factor = abs(df[df['pnl_pct']>0]['pnl_pct'].sum() / loss_sum) if loss_sum != 0 else 999
    avg_hold = df['holding_days'].mean()
    trades_per_sym = n_trades / n_symbols if n_symbols > 0 else 0

    summary_rows.append({
        'model': name, 'trades': n_trades, 'symbols': n_symbols,
        'avg_pnl': round(avg_pnl, 2), 'median_pnl': round(median_pnl, 2),
        'total_pnl': round(total_pnl, 1), 'win_rate%': round(win_rate, 1),
        'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'max_gain': round(max_gain, 1), 'max_loss': round(max_loss, 1),
        'avg_hold_days': round(avg_hold, 1), 'trades/sym': round(trades_per_sym, 1)
    })

summary_df = pd.DataFrame(summary_rows).sort_values('avg_pnl', ascending=False)
print(summary_df.to_string(index=False))

# ====================================================================
print("\n" + "="*100)
print("PART 2: HIEU SUAT THEO TUNG MA CO PHIEU - V25 vs RULE vs TOP MODELS")
print("="*100)

v25 = models['v25']
rule = models['rule']
common_symbols = sorted(set(v25['symbol'].unique()) & set(rule['symbol'].unique()))

symbol_compare = []
for sym in common_symbols:
    row = {'symbol': sym}
    for mname in ['v25', 'rule', 'v24', 'v23', 'v22']:
        if mname in models:
            df_s = models[mname][models[mname]['symbol'] == sym]
            if len(df_s) > 0:
                row[f'{mname}_trades'] = len(df_s)
                row[f'{mname}_avg_pnl'] = round(df_s['pnl_pct'].mean(), 2)
                row[f'{mname}_total_pnl'] = round(df_s['pnl_pct'].sum(), 1)
                row[f'{mname}_winrate'] = round((df_s['pnl_pct'] > 0).mean()*100, 1)
            else:
                row[f'{mname}_trades'] = 0
                row[f'{mname}_avg_pnl'] = 0
                row[f'{mname}_total_pnl'] = 0
                row[f'{mname}_winrate'] = 0
    symbol_compare.append(row)

sym_df = pd.DataFrame(symbol_compare)
if 'v25_total_pnl' in sym_df.columns and 'rule_total_pnl' in sym_df.columns:
    sym_df['v25_vs_rule_gap'] = sym_df['v25_total_pnl'] - sym_df['rule_total_pnl']
else:
    sym_df['v25_vs_rule_gap'] = 0

sym_df_sorted = sym_df.sort_values('v25_vs_rule_gap')

print("\n--- TOP 20 MA V25 THUA XA RULE (Gap am lon nhat) ---")
cols_show = ['symbol', 'v25_trades', 'v25_avg_pnl', 'v25_total_pnl', 'v25_winrate',
             'rule_trades', 'rule_avg_pnl', 'rule_total_pnl', 'rule_winrate', 'v25_vs_rule_gap']
available_cols = [c for c in cols_show if c in sym_df_sorted.columns]
print(sym_df_sorted.head(20)[available_cols].to_string(index=False))

print("\n--- TOP 20 MA V25 THANG RULE (Gap duong lon nhat) ---")
print(sym_df_sorted.tail(20)[available_cols].to_string(index=False))

# ====================================================================
print("\n" + "="*100)
print("PART 3: PHAN TICH CHI TIET CAC MA V25 HIEU SUAT KEM NHAT")
print("="*100)

worst_symbols = sym_df_sorted.head(15)['symbol'].tolist()
for sym in worst_symbols[:10]:
    print(f"\n{'='*80}")
    print(f"  MA: {sym}")
    print(f"{'='*80}")

    v25_s = v25[v25['symbol'] == sym].copy()
    rule_s = rule[rule['symbol'] == sym].copy()

    v25_wr = (v25_s['pnl_pct']>0).mean()*100
    rule_wr = (rule_s['pnl_pct']>0).mean()*100
    print(f"\n  V25: {len(v25_s)} trades, avg_pnl={v25_s['pnl_pct'].mean():.2f}%, total={v25_s['pnl_pct'].sum():.1f}%, WR={v25_wr:.1f}%")
    print(f"  Rule: {len(rule_s)} trades, avg_pnl={rule_s['pnl_pct'].mean():.2f}%, total={rule_s['pnl_pct'].sum():.1f}%, WR={rule_wr:.1f}%")

    print(f"\n  V25 Giao dich chi tiet (sorted by pnl):")
    v25_sorted = v25_s.sort_values('pnl_pct')
    show_cols_v25 = ['entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'max_profit_pct', 'exit_reason', 'entry_trend', 'entry_score', 'position_size']
    ac = [c for c in show_cols_v25 if c in v25_sorted.columns]
    print(v25_sorted[ac].to_string(index=False))

    print(f"\n  V25 Exit reasons:")
    print(v25_s['exit_reason'].value_counts().to_string())

    if 'entry_date' in rule_s.columns:
        print(f"\n  Rule Giao dich chi tiet (sorted by pnl):")
        rule_sorted = rule_s.sort_values('pnl_pct')
        rule_show = ['entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'exit_reason']
        ar = [c for c in rule_show if c in rule_sorted.columns]
        print(rule_sorted[ar].to_string(index=False))

    big_losses = v25_s[v25_s['pnl_pct'] < -5]
    if len(big_losses) > 0:
        print(f"\n  V25 Big losses (< -5%): {len(big_losses)} trades, total = {big_losses['pnl_pct'].sum():.1f}%")

    for mname in ['v24', 'v23', 'v22', 'v19_1']:
        if mname in models:
            ms = models[mname][models[mname]['symbol'] == sym]
            if len(ms) > 0:
                ms_wr = (ms['pnl_pct']>0).mean()*100
                print(f"  {mname}: {len(ms)} trades, avg={ms['pnl_pct'].mean():.2f}%, total={ms['pnl_pct'].sum():.1f}%, WR={ms_wr:.1f}%")

# ====================================================================
print("\n" + "="*100)
print("PART 4: PHAN TICH GIAO DICH NHIEU & NOISE CUA V25")
print("="*100)

v25_noise = v25[(v25['holding_days'] <= 5) & (v25['pnl_pct'].abs() < 3)]
v25_big_loss = v25[v25['pnl_pct'] < -10]
v25_whipsaw = v25[v25['exit_reason'].str.contains('signal', na=False) & (v25['holding_days'] <= 3)]

print(f"\n  Tong giao dich v25: {len(v25)}")
print(f"  Giao dich nhieu (hold<=5d, |pnl|<3%): {len(v25_noise)} ({len(v25_noise)/len(v25)*100:.1f}%)")
print(f"  Giao dich thua lo lon (pnl<-10%): {len(v25_big_loss)} ({len(v25_big_loss)/len(v25)*100:.1f}%)")
print(f"  Whipsaw (signal exit <=3d): {len(v25_whipsaw)} ({len(v25_whipsaw)/len(v25)*100:.1f}%)")

print(f"\n  Impact giao dich nhieu: total_pnl = {v25_noise['pnl_pct'].sum():.1f}%")
print(f"  Impact giao dich thua lon: total_pnl = {v25_big_loss['pnl_pct'].sum():.1f}%")
print(f"  Impact whipsaw: total_pnl = {v25_whipsaw['pnl_pct'].sum():.1f}%")

print(f"\n  Top 15 ma nhieu nhieu nhat (hold<=5d, |pnl|<3%):")
noise_by_sym = v25_noise.groupby('symbol').agg(
    count=('pnl_pct', 'count'),
    total_pnl=('pnl_pct', 'sum'),
    avg_pnl=('pnl_pct', 'mean')
).sort_values('count', ascending=False).head(15)
print(noise_by_sym.to_string())

print(f"\n  Top 15 ma thua lo lon nhat:")
bigloss_by_sym = v25_big_loss.groupby('symbol').agg(
    count=('pnl_pct', 'count'),
    total_pnl=('pnl_pct', 'sum'),
    avg_pnl=('pnl_pct', 'mean')
).sort_values('total_pnl').head(15)
print(bigloss_by_sym.to_string())

print(f"\n  V25 Exit Reason breakdown:")
exit_stats = v25.groupby('exit_reason').agg(
    count=('pnl_pct', 'count'),
    avg_pnl=('pnl_pct', 'mean'),
    total_pnl=('pnl_pct', 'sum'),
    win_rate=('pnl_pct', lambda x: (x>0).mean()*100)
).sort_values('count', ascending=False)
print(exit_stats.to_string())

# ====================================================================
print("\n" + "="*100)
print("PART 5: SO SANH DIEM MUA/BAN - NOI MODEL KHAC LAM TOT HON V25")
print("="*100)

model_vs_v25 = []
for mname in ['rule', 'v24', 'v23', 'v22', 'v19_1']:
    if mname not in models:
        continue
    m_df = models[mname]
    for sym in v25['symbol'].unique():
        v25_s = v25[v25['symbol'] == sym]
        m_s = m_df[m_df['symbol'] == sym]
        if len(v25_s) > 0 and len(m_s) > 0:
            gap = m_s['pnl_pct'].mean() - v25_s['pnl_pct'].mean()
            if gap > 2:
                model_vs_v25.append({
                    'symbol': sym,
                    'better_model': mname,
                    'v25_avg': round(v25_s['pnl_pct'].mean(), 2),
                    'v25_trades': len(v25_s),
                    'other_avg': round(m_s['pnl_pct'].mean(), 2),
                    'other_trades': len(m_s),
                    'gap': round(gap, 2),
                    'v25_winrate': round((v25_s['pnl_pct']>0).mean()*100, 1),
                    'other_winrate': round((m_s['pnl_pct']>0).mean()*100, 1),
                })

if model_vs_v25:
    compare_df = pd.DataFrame(model_vs_v25).sort_values('gap', ascending=False)
    print(f"\n  Tong {len(compare_df)} truong hop model khac tot hon v25 >2% avg_pnl:")
    print(compare_df.head(30).to_string(index=False))

    print(f"\n  So ma ma moi model tot hon v25:")
    print(compare_df.groupby('better_model')['gap'].agg(['count', 'mean', 'sum']).sort_values('sum', ascending=False).to_string())

# ====================================================================
print("\n" + "="*100)
print("PART 6: PHAN TICH THEO ENTRY TREND & PROFILE")
print("="*100)

if 'entry_trend' in v25.columns:
    print("\n  V25 theo entry_trend:")
    trend_stats = v25.groupby('entry_trend').agg(
        count=('pnl_pct', 'count'),
        avg_pnl=('pnl_pct', 'mean'),
        total_pnl=('pnl_pct', 'sum'),
        win_rate=('pnl_pct', lambda x: (x>0).mean()*100),
        avg_hold=('holding_days', 'mean')
    ).round(2)
    print(trend_stats.to_string())

if 'entry_profile' in v25.columns:
    print("\n  V25 theo entry_profile:")
    profile_stats = v25.groupby('entry_profile').agg(
        count=('pnl_pct', 'count'),
        avg_pnl=('pnl_pct', 'mean'),
        total_pnl=('pnl_pct', 'sum'),
        win_rate=('pnl_pct', lambda x: (x>0).mean()*100),
        avg_hold=('holding_days', 'mean')
    ).round(2)
    print(profile_stats.to_string())

if 'entry_score' in v25.columns:
    print("\n  V25 theo entry_score:")
    score_stats = v25.groupby('entry_score').agg(
        count=('pnl_pct', 'count'),
        avg_pnl=('pnl_pct', 'mean'),
        total_pnl=('pnl_pct', 'sum'),
        win_rate=('pnl_pct', lambda x: (x>0).mean()*100),
        avg_hold=('holding_days', 'mean')
    ).round(2)
    print(score_stats.to_string())

# ====================================================================
print("\n" + "="*100)
print("PART 7: PHAN TICH POSITION SIZE & RISK")
print("="*100)

if 'position_size' in v25.columns:
    print("\n  V25 theo position_size:")
    ps_stats = v25.groupby(pd.cut(v25['position_size'], bins=[0, 0.25, 0.35, 0.45, 0.55, 1.0])).agg(
        count=('pnl_pct', 'count'),
        avg_pnl=('pnl_pct', 'mean'),
        total_pnl=('pnl_pct', 'sum'),
        win_rate=('pnl_pct', lambda x: (x>0).mean()*100)
    ).round(2)
    print(ps_stats.to_string())

if 'max_profit_pct' in v25.columns:
    v25_profitable = v25[v25['max_profit_pct'] > 0].copy()
    v25_profitable['capture_ratio'] = v25_profitable['pnl_pct'] / v25_profitable['max_profit_pct']
    print(f"\n  V25 Capture Ratio (actual_pnl / max_profit):")
    print(f"    Mean capture ratio: {v25_profitable['capture_ratio'].mean():.3f}")
    print(f"    Median capture ratio: {v25_profitable['capture_ratio'].median():.3f}")

    wasted = v25[(v25['max_profit_pct'] > 50) & (v25['pnl_pct'] < 0)]
    print(f"\n  Trades with max_profit>50% but exited at loss: {len(wasted)}")
    if len(wasted) > 0:
        print(f"    Total pnl lost: {wasted['pnl_pct'].sum():.1f}%")
        print(f"    Total max_profit missed: {wasted['max_profit_pct'].sum():.1f}%")
        wasted_cols = ['entry_date', 'exit_date', 'holding_days', 'pnl_pct', 'max_profit_pct', 'exit_reason', 'entry_trend', 'symbol']
        wc = [c for c in wasted_cols if c in wasted.columns]
        print(wasted.sort_values('max_profit_pct', ascending=False).head(15)[wc].to_string(index=False))

# ====================================================================
print("\n" + "="*100)
print("PART 8: TIM CAC GIAO DICH MA RULE/MODEL KHAC CO LOI NHUAN TOT MA V25 KHONG BAT DUOC")
print("="*100)

for sym in worst_symbols[:8]:
    v25_s = v25[v25['symbol'] == sym].copy()
    rule_s = rule[rule['symbol'] == sym].copy()

    if len(rule_s) == 0 or len(v25_s) == 0:
        continue

    rule_best = rule_s[rule_s['pnl_pct'] > 5].sort_values('pnl_pct', ascending=False)
    if len(rule_best) > 0:
        print(f"\n  {sym} - Rule co {len(rule_best)} giao dich >5% ma v25 performance kem:")

        for _, rt in rule_best.iterrows():
            rule_entry = rt['entry_date']
            rule_exit = rt['exit_date']
            rule_pnl = rt['pnl_pct']

            if 'exit_date' in v25_s.columns:
                v25_overlap = v25_s[
                    (v25_s['entry_date'] <= rule_exit) &
                    (v25_s['exit_date'] >= rule_entry)
                ]
            else:
                v25_overlap = pd.DataFrame()

            if len(v25_overlap) > 0:
                v25_pnl = v25_overlap['pnl_pct'].sum()
                n_trades = len(v25_overlap)
                if v25_pnl < rule_pnl:
                    status = f"V25 CUNG VAO nhung kem hon"
                else:
                    status = "V25 tot hon"
                print(f"    Rule: {rule_entry} -> {rule_exit}, pnl={rule_pnl:.1f}% | {status} pnl={v25_pnl:.1f}% ({n_trades} trades)")
            else:
                print(f"    Rule: {rule_entry} -> {rule_exit}, pnl={rule_pnl:.1f}% | V25 BO LO co hoi nay!")

# ====================================================================
print("\n" + "="*100)
print("PART 9: HOLDING DAYS DISTRIBUTION & OPTIMAL EXIT")
print("="*100)

for mname in ['v25', 'rule', 'v24', 'v23']:
    if mname in models:
        df = models[mname]
        print(f"\n  {mname}: avg_hold={df['holding_days'].mean():.1f}d, median={df['holding_days'].median():.0f}d")
        short = df[df['holding_days'] <= 5]
        med = df[(df['holding_days'] > 5) & (df['holding_days'] <= 20)]
        long_t = df[df['holding_days'] > 20]
        print(f"    Short(<=5d): {len(short)} trades, avg_pnl={short['pnl_pct'].mean():.2f}%")
        print(f"    Med(6-20d): {len(med)} trades, avg_pnl={med['pnl_pct'].mean():.2f}%")
        print(f"    Long(>20d): {len(long_t)} trades, avg_pnl={long_t['pnl_pct'].mean():.2f}%")

# ====================================================================
print("\n" + "="*100)
print("PART 10: V25 ENTRY FEATURES CUA GIAO DICH THANG vs THUA")
print("="*100)

v25_win = v25[v25['pnl_pct'] > 0]
v25_loss = v25[v25['pnl_pct'] <= 0]

feature_cols = ['entry_wp', 'entry_dp', 'entry_rs', 'entry_vs', 'entry_bs', 'entry_hl',
                'entry_od', 'entry_bb', 'entry_score', 'entry_ret_5d', 'entry_drop20d', 'entry_dist_sma20']
available_feats = [c for c in feature_cols if c in v25.columns]

try:
    from scipy import stats as scipy_stats
    has_scipy = True
except:
    has_scipy = False

print(f"\n  {'Feature':<20} {'Win_mean':>10} {'Loss_mean':>10} {'Diff':>10} {'Signif':>10}")
print("  " + "-"*60)
for feat in available_feats:
    w_mean = v25_win[feat].mean()
    l_mean = v25_loss[feat].mean()
    diff = w_mean - l_mean
    if has_scipy:
        t, p = scipy_stats.ttest_ind(v25_win[feat].dropna(), v25_loss[feat].dropna())
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:<20} {w_mean:>10.4f} {l_mean:>10.4f} {diff:>10.4f} {t:>8.2f}{sig}")
    else:
        print(f"  {feat:<20} {w_mean:>10.4f} {l_mean:>10.4f} {diff:>10.4f}")

v25_bigloss = v25[v25['pnl_pct'] < -10]
v25_bigwin = v25[v25['pnl_pct'] > 10]
print(f"\n  Features: Big Win (>10%) vs Big Loss (<-10%)")
print(f"  {'Feature':<20} {'BigWin_mean':>12} {'BigLoss_mean':>12} {'Diff':>10}")
print("  " + "-"*60)
for feat in available_feats:
    w = v25_bigwin[feat].mean()
    l = v25_bigloss[feat].mean()
    print(f"  {feat:<20} {w:>12.4f} {l:>12.4f} {(w-l):>10.4f}")

# ====================================================================
print("\n" + "="*100)
print("PART 11: PHAN TICH CHOPPY REGIME & BREAKOUT/VSHAPE")
print("="*100)

if 'entry_choppy_regime' in v25.columns:
    print("\n  V25 theo choppy_regime:")
    choppy_stats = v25.groupby('entry_choppy_regime').agg(
        count=('pnl_pct', 'count'),
        avg_pnl=('pnl_pct', 'mean'),
        total_pnl=('pnl_pct', 'sum'),
        win_rate=('pnl_pct', lambda x: (x>0).mean()*100)
    ).round(2)
    print(choppy_stats.to_string())

for entry_type in ['breakout_entry', 'vshape_entry', 'quick_reentry']:
    if entry_type in v25.columns:
        print(f"\n  V25 theo {entry_type}:")
        et_stats = v25.groupby(entry_type).agg(
            count=('pnl_pct', 'count'),
            avg_pnl=('pnl_pct', 'mean'),
            total_pnl=('pnl_pct', 'sum'),
            win_rate=('pnl_pct', lambda x: (x>0).mean()*100)
        ).round(2)
        print(et_stats.to_string())

# ====================================================================
print("\n" + "="*100)
print("PART 12: MODEL KHAC CO DIEM MUA/BAN TOT HON - CHI TIET TUNG MA")
print("="*100)

interesting_symbols = ['HPG', 'VND', 'FPT', 'TCB', 'ACB', 'SSI', 'MBB', 'VIC', 'VNM', 'REE', 'DGC', 'BID']
for sym in interesting_symbols:
    results = {}
    for mname in models:
        ms = models[mname][models[mname]['symbol'] == sym]
        if len(ms) > 0:
            results[mname] = {
                'trades': len(ms),
                'avg_pnl': round(ms['pnl_pct'].mean(), 2),
                'total_pnl': round(ms['pnl_pct'].sum(), 1),
                'win_rate': round((ms['pnl_pct']>0).mean()*100, 1),
                'avg_hold': round(ms['holding_days'].mean(), 1),
            }

    if len(results) > 1:
        print(f"\n  === {sym} ===")
        for mname, r in sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            print(f"    {mname:>6}: {r['trades']:>3} trades, avg={r['avg_pnl']:>6.2f}%, total={r['total_pnl']:>7.1f}%, WR={r['win_rate']:>5.1f}%, hold={r['avg_hold']:>5.1f}d")

print("\n\nDONE.")
