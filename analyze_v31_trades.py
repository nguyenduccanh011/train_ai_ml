import pandas as pd, numpy as np

df = pd.read_csv(r'C:\Users\DUC CANH PC\Desktop\train_ai_ml\stock_ml\results\trades_v31.csv')
total = len(df)
print(f"=== TOTAL TRADES: {total} ===\n")

# ============================================================
# 1. EXIT REASON BREAKDOWN
# ============================================================
print("=" * 60)
print("1. EXIT REASON BREAKDOWN")
print("=" * 60)
grp = df.groupby('exit_reason').agg(
    trades=('pnl_pct','count'),
    wins=('pnl_pct', lambda x: (x>0).sum()),
    avg_pnl=('pnl_pct','mean'),
    total_pnl=('pnl_pct','sum')
).reset_index()
grp['WR%'] = (grp['wins']/grp['trades']*100).round(1)
grp['avg_pnl'] = grp['avg_pnl'].round(2)
grp['total_pnl'] = grp['total_pnl'].round(1)
grp = grp.sort_values('total_pnl', ascending=False)
print(grp[['exit_reason','trades','WR%','avg_pnl','total_pnl']].to_string(index=False))

# ============================================================
# 2. HARD CAP DEEP DIVE
# ============================================================
print("\n" + "=" * 60)
print("2. HARD CAP DEEP DIVE (exit_reason == signal_hard_cap)")
print("=" * 60)
hc = df[df['exit_reason'] == 'signal_hard_cap'].copy()
print(f"Total hard cap trades: {len(hc)}")
print(f"WR: {(hc['pnl_pct']>0).mean()*100:.1f}%")
print(f"Avg pnl: {hc['pnl_pct'].mean():.2f}%")
print(f"Total pnl: {hc['pnl_pct'].sum():.1f}%")

print("\n--- holding_days distribution ---")
bins = [0,3,5,10,15,20,30,999]
labels = ['1-3','4-5','6-10','11-15','16-20','21-30','31+']
hc['hd_bin'] = pd.cut(hc['holding_days'], bins=bins, labels=labels)
print(hc['hd_bin'].value_counts().sort_index().to_string())

print("\n--- pnl_pct distribution ---")
pnl_bins = [-999,-20,-15,-10,-5,0,5,10,999]
pnl_labels = ['<-20','-20~-15','-15~-10','-10~-5','-5~0','0~5','5~10','>10']
hc['pnl_bin'] = pd.cut(hc['pnl_pct'], bins=pnl_bins, labels=pnl_labels)
print(hc['pnl_bin'].value_counts().sort_index().to_string())

print("\n--- price_max_profit_pct distribution ---")
mp_bins = [-999,0,5,10,20,50,999]
mp_labels = ['<0','0~5','5~10','10~20','20~50','>50']
hc['mp_bin'] = pd.cut(hc['price_max_profit_pct'], bins=mp_bins, labels=mp_labels)
print(hc['mp_bin'].value_counts().sort_index().to_string())

print("\n--- entry_profile distribution ---")
print(hc['entry_profile'].value_counts().to_string())

print("\n--- entry_type: vshape/breakout/quick_reentry ---")
print(f"vshape_entry:    {hc['vshape_entry'].sum()} ({hc['vshape_entry'].mean()*100:.1f}%)")
print(f"breakout_entry:  {hc['breakout_entry'].sum()} ({hc['breakout_entry'].mean()*100:.1f}%)")
print(f"quick_reentry:   {hc['quick_reentry'].sum()} ({hc['quick_reentry'].mean()*100:.1f}%)")

print("\n--- entry_trend distribution ---")
print(hc['entry_trend'].value_counts().to_string())

print("\n--- Had max_profit > 0% before hard cap? ---")
ran_up = (hc['price_max_profit_pct'] > 0).sum()
print(f"price_max_profit_pct > 0%:  {ran_up} / {len(hc)}  ({ran_up/len(hc)*100:.1f}%)")
print(f"price_max_profit_pct > 5%:  {(hc['price_max_profit_pct'] > 5).sum()}")
print(f"price_max_profit_pct > 10%: {(hc['price_max_profit_pct'] > 10).sum()}")
print(f"price_max_profit_pct > 20%: {(hc['price_max_profit_pct'] > 20).sum()}")

print("\n--- avg entry_ret_5d & entry_dist_sma20: hard cap vs all ---")
print(f"{'Metric':<25} {'Hard Cap':>10} {'All Trades':>12}")
for col in ['entry_ret_5d','entry_dist_sma20']:
    print(f"{col:<25} {hc[col].mean():>10.2f} {df[col].mean():>12.2f}")

print("\n--- exit_trend for hard cap trades ---")
print(hc['exit_trend'].value_counts().to_string())

print("\n--- exit_dist_sma20 distribution ---")
below_5 = (hc['exit_dist_sma20'] < -5).sum()
mid = ((hc['exit_dist_sma20'] >= -5) & (hc['exit_dist_sma20'] <= 0)).sum()
above_0 = (hc['exit_dist_sma20'] > 0).sum()
print(f"Below -5%:   {below_5} ({below_5/len(hc)*100:.1f}%)")
print(f"-5% to 0%:   {mid} ({mid/len(hc)*100:.1f}%)")
print(f"Above 0%:    {above_0} ({above_0/len(hc)*100:.1f}%)")
print(f"Avg exit_dist_sma20: {hc['exit_dist_sma20'].mean():.2f}%")

# ============================================================
# 3. SIGNAL EXITS ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("3. SIGNAL EXITS ANALYSIS (exit_reason == signal)")
print("=" * 60)
sig = df[df['exit_reason'] == 'signal'].copy()
print(f"Total: {len(sig)} | WR: {(sig['pnl_pct']>0).mean()*100:.1f}% | Avg pnl: {sig['pnl_pct'].mean():.2f}%")

print("\n--- exit_trend distribution ---")
print(sig['exit_trend'].value_counts().to_string())

print("\n--- exit_dist_sma20 distribution ---")
bins2 = [-999,-10,-5,0,5,10,999]
labels2 = ['<-10','-10~-5','-5~0','0~5','5~10','>10']
sig['ed_bin'] = pd.cut(sig['exit_dist_sma20'], bins=bins2, labels=labels2)
print(sig['ed_bin'].value_counts().sort_index().to_string())

print("\n--- exit_rsi14 distribution ---")
rsi_bins = [0,30,40,50,60,70,100]
rsi_labels = ['<30','30-40','40-50','50-60','60-70','>70']
sig['rsi_bin'] = pd.cut(sig['exit_rsi14'], bins=rsi_bins, labels=rsi_labels)
print(sig['rsi_bin'].value_counts().sort_index().to_string())

print("\n--- exit_above_sma20 distribution ---")
print(sig['exit_above_sma20'].value_counts().to_string())

print("\n--- WR by exit_trend ---")
for trend in ['strong','moderate','weak']:
    sub = sig[sig['exit_trend']==trend]
    if len(sub):
        wr = (sub['pnl_pct']>0).mean()*100
        print(f"  {trend:<12}: {len(sub):>4} trades | WR {wr:.1f}% | avg pnl {sub['pnl_pct'].mean():.2f}%")

print("\n--- WR by exit_above_sma20 ---")
for val in sorted(sig['exit_above_sma20'].unique()):
    sub = sig[sig['exit_above_sma20']==val]
    wr = (sub['pnl_pct']>0).mean()*100
    print(f"  exit_above_sma20={val:.0f}: {len(sub):>4} trades | WR {wr:.1f}% | avg pnl {sub['pnl_pct'].mean():.2f}%")

# ============================================================
# 4. v31_hap_exit ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("4. v31_hap_exit ANALYSIS")
print("=" * 60)
hap = df[df['exit_reason'] == 'v31_hap_exit'].copy()
print(f"Count: {len(hap)} | WR: {(hap['pnl_pct']>0).mean()*100:.1f}% | Avg pnl: {hap['pnl_pct'].mean():.2f}%")
print(f"Total pnl: {hap['pnl_pct'].sum():.1f}%")
print("\n--- pnl_pct distribution ---")
hap['pnl_bin'] = pd.cut(hap['pnl_pct'], bins=[-999,-10,-5,0,5,10,20,999],
                          labels=['<-10','-10~-5','-5~0','0~5','5~10','10~20','>20'])
print(hap['pnl_bin'].value_counts().sort_index().to_string())
print(f"\n--- price_max_profit_pct before exit ---")
print(f"Avg max_profit:  {hap['price_max_profit_pct'].mean():.2f}%")
print(f"Med max_profit:  {hap['price_max_profit_pct'].median():.2f}%")
print(f">10%: {(hap['price_max_profit_pct']>10).sum()}")
print(f">20%: {(hap['price_max_profit_pct']>20).sum()}")
print(f"Avg pnl at exit: {hap['pnl_pct'].mean():.2f}%")
print(f"Avg missed gain (max-pnl): {(hap['price_max_profit_pct'] - hap['pnl_pct']).mean():.2f}%")

# ============================================================
# 5. ENTRY NEAR PEAK (entry_dist_sma20 > 8%)
# ============================================================
print("\n" + "=" * 60)
print("5. ENTRY NEAR PEAK (entry_dist_sma20 > 8%)")
print("=" * 60)
peak = df[df['entry_dist_sma20'] > 8].copy()
print(f"Total: {len(peak)} / {total} ({len(peak)/total*100:.1f}%)")
print(f"WR:      {(peak['pnl_pct']>0).mean()*100:.1f}%  vs all {(df['pnl_pct']>0).mean()*100:.1f}%")
print(f"Avg pnl: {peak['pnl_pct'].mean():.2f}%  vs all {df['pnl_pct'].mean():.2f}%")
print("\n--- Exit reasons for peak-entry trades ---")
print(peak['exit_reason'].value_counts().to_string())

# ============================================================
# 6. QUICK LOSERS (hold < 10 days, pnl < -3%)
# ============================================================
print("\n" + "=" * 60)
print("6. QUICK LOSERS (holding_days < 10, pnl < -3%)")
print("=" * 60)
ql = df[(df['holding_days'] < 10) & (df['pnl_pct'] < -3)].copy()
print(f"Count: {len(ql)} | Avg pnl: {ql['pnl_pct'].mean():.2f}%")
print("\n--- exit_reason ---")
print(ql['exit_reason'].value_counts().to_string())
print("\n--- entry_profile ---")
print(ql['entry_profile'].value_counts().to_string())
print("\n--- entry_trend ---")
print(ql['entry_trend'].value_counts().to_string())
print("\n--- pnl stats ---")
print(f"Min: {ql['pnl_pct'].min():.2f}% | P25: {ql['pnl_pct'].quantile(.25):.2f}% | Med: {ql['pnl_pct'].median():.2f}%")

# ============================================================
# 7. LARGE LOSS TRADES (pnl < -10%)
# ============================================================
print("\n" + "=" * 60)
print("7. LARGE LOSS TRADES (pnl < -10%)")
print("=" * 60)
ll = df[df['pnl_pct'] < -10].copy()
print(f"Count: {len(ll)} | Avg pnl: {ll['pnl_pct'].mean():.2f}%")
print(f"\n--- max_profit before exit ---")
print(f"price_max_profit_pct > 0%:  {(ll['price_max_profit_pct']>0).sum()} ({(ll['price_max_profit_pct']>0).mean()*100:.1f}%)")
print(f"price_max_profit_pct > 5%:  {(ll['price_max_profit_pct']>5).sum()}")
print(f"price_max_profit_pct > 10%: {(ll['price_max_profit_pct']>10).sum()}")
print(f"price_max_profit_pct > 15%: {(ll['price_max_profit_pct']>15).sum()}")
print(f"Avg price_max_profit_pct:   {ll['price_max_profit_pct'].mean():.2f}%")

print(f"\n--- exit_trend ---")
print(ll['exit_trend'].value_counts().to_string())

print(f"\n--- exit_reason ---")
print(ll['exit_reason'].value_counts().to_string())

potential_save = ll[(ll['price_max_profit_pct'] > 10) & (ll['pnl_pct'] < -10)]
print(f"\n--- V31-C (hardcap_after_profit) potential ---")
print(f"Trades with max_profit>10% then ended >-10% loss: {len(potential_save)}")
if len(potential_save):
    print(f"Avg final pnl: {potential_save['pnl_pct'].mean():.2f}%")
    print(f"Avg max_profit: {potential_save['price_max_profit_pct'].mean():.2f}%")
    er = potential_save['exit_reason'].value_counts().to_dict()
    print(f"Exit reasons: {er}")

potential_save2 = ll[(ll['price_max_profit_pct'] > 5) & (ll['pnl_pct'] < -10)]
print(f"\nTrades with max_profit>5% then ended >-10% loss: {len(potential_save2)}")
print(f"  Total pnl of these: {potential_save2['pnl_pct'].sum():.1f}%")

# ============================================================
# 8. PROFITABLE WITH MISSED UPSIDE
# ============================================================
print("\n" + "=" * 60)
print("8. PROFITABLE TRADES WITH MISSED UPSIDE (pnl>0, max_profit > pnl+5%)")
print("=" * 60)
missed = df[(df['pnl_pct'] > 0) & (df['price_max_profit_pct'] > df['pnl_pct'] + 5)].copy()
prof_total = (df['pnl_pct']>0).sum()
print(f"Count: {len(missed)} / {prof_total} profitable ({len(missed)/prof_total*100:.1f}%)")
print(f"Avg pnl:         {missed['pnl_pct'].mean():.2f}%")
print(f"Avg max_profit:  {missed['price_max_profit_pct'].mean():.2f}%")
print(f"Avg missed:      {(missed['price_max_profit_pct'] - missed['pnl_pct']).mean():.2f}%")
print(f"Avg holding_days:{missed['holding_days'].mean():.1f}")
print(f"\n--- exit_reason ---")
print(missed['exit_reason'].value_counts().to_string())

missed10 = df[(df['pnl_pct'] > 0) & (df['price_max_profit_pct'] > df['pnl_pct'] + 10)]
print(f"\nWith missed > 10%: {len(missed10)} trades | avg missed: {(missed10['price_max_profit_pct'] - missed10['pnl_pct']).mean():.2f}%")

# ============================================================
# 9. QUICK REENTRY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("9. QUICK REENTRY ANALYSIS")
print("=" * 60)
re_df = df[df['quick_reentry'] == True].copy()
nre_df = df[df['quick_reentry'] == False].copy()
print(f"quick_reentry=True:  {len(re_df):4d} trades | WR: {(re_df['pnl_pct']>0).mean()*100:.1f}% | Avg pnl: {re_df['pnl_pct'].mean():.2f}% | Total: {re_df['pnl_pct'].sum():.1f}%")
print(f"quick_reentry=False: {len(nre_df):4d} trades | WR: {(nre_df['pnl_pct']>0).mean()*100:.1f}% | Avg pnl: {nre_df['pnl_pct'].mean():.2f}% | Total: {nre_df['pnl_pct'].sum():.1f}%")
print(f"\n--- Reentry exit reasons ---")
print(re_df['exit_reason'].value_counts().to_string())
print(f"\n--- Reentry pnl distribution ---")
re_df['pnl_bin'] = pd.cut(re_df['pnl_pct'], bins=[-999,-10,-5,0,5,10,20,999],
                        labels=['<-10','-10~-5','-5~0','0~5','5~10','10~20','>20'])
print(re_df['pnl_bin'].value_counts().sort_index().to_string())

# BONUS
print("\n" + "=" * 60)
print("BONUS: HARD CAP by entry_profile vs all")
print("=" * 60)
ep_all = df.groupby('entry_profile').agg(trades=('pnl_pct','count'), avg_pnl=('pnl_pct','mean')).round(2)
ep_hc  = hc.groupby('entry_profile').agg(hc_trades=('pnl_pct','count'), hc_avg_pnl=('pnl_pct','mean')).round(2)
ep_merged = ep_all.join(ep_hc)
ep_merged['hc_rate%'] = (ep_merged['hc_trades']/ep_merged['trades']*100).round(1)
print(ep_merged.to_string())

print("\n" + "=" * 60)
print("SUMMARY STATS")
print("=" * 60)
print(f"Total trades:  {total}")
print(f"Total PnL:     {df['pnl_pct'].sum():.1f}%")
print(f"Overall WR:    {(df['pnl_pct']>0).mean()*100:.1f}%")
print(f"Avg PnL/trade: {df['pnl_pct'].mean():.2f}%")
print(f"\nHard cap trades: {len(hc)} ({len(hc)/total*100:.1f}%) | Drag: {hc['pnl_pct'].sum():.1f}%")
print(f"If hard cap avg was 0 (breakeven): composite gain ~{-hc['pnl_pct'].sum():.0f} pts")
