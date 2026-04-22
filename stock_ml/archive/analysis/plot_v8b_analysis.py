"""Generate charts comparing V7 vs V8b trade analysis."""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v8b_compare import backtest_v8b
from run_v7_compare import backtest_v7
from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model

# Load data
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'portable_data', 'vn_stock_ai_dataset_cleaned')
config = {
    'data': {'data_dir': data_dir},
    'split': {'method': 'walk_forward', 'train_years': 4, 'test_years': 1, 'gap_days': 0, 'first_test_year': 2020, 'last_test_year': 2025},
    'target': {'type': 'trend_regime', 'trend_method': 'dual_ma', 'short_window': 10, 'long_window': 40, 'classes': 3},
}
loader = DataLoader(data_dir)
splitter = WalkForwardSplitter.from_config(config)
target_gen = TargetGenerator.from_config(config)
pick = 'ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC'.split(',')
symbols = [s for s in pick if s in loader.symbols]
raw_df = loader.load_all(symbols=symbols)
engine = FeatureEngine(feature_set='leading')
df = engine.compute_for_all_symbols(raw_df)
df = target_gen.generate_for_all_symbols(df)
feature_cols = engine.get_feature_columns(df)
df = df.dropna(subset=feature_cols + ['target'])

print("Running backtests...")
v7_all, v8b_all = [], []
for window, train_df, test_df in splitter.split(df):
    model = build_model('lightgbm')
    X_train = np.nan_to_num(train_df[feature_cols].values)
    y_train = train_df['target'].values.astype(int)
    model.fit(X_train, y_train)
    for sym in test_df['symbol'].unique():
        if sym not in symbols: continue
        sym_test = test_df[test_df['symbol'] == sym].reset_index(drop=True)
        if len(sym_test) < 10: continue
        X_sym = np.nan_to_num(sym_test[feature_cols].values)
        y_pred = model.predict(X_sym)
        rets = sym_test['return_1d'].values if 'return_1d' in sym_test.columns else np.zeros(len(sym_test))
        r7 = backtest_v7(y_pred, rets, sym_test, feature_cols)
        r8 = backtest_v8b(y_pred, rets, sym_test, feature_cols)
        for t in r7['trades']: t['symbol'] = sym; t['window'] = window.label
        for t in r8['trades']: t['symbol'] = sym; t['window'] = window.label
        v7_all.extend(r7['trades'])
        v8b_all.extend(r8['trades'])

print(f"V7: {len(v7_all)} trades, V8b: {len(v8b_all)} trades")

# ═══════════════════════════════════════════════════════════
# Create charts
# ═══════════════════════════════════════════════════════════
plt.rcParams.update({'font.size': 11, 'figure.facecolor': 'white'})
fig = plt.figure(figsize=(20, 24))
fig.suptitle('V7 vs V8b — Trade Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)

colors_v7 = '#2196F3'
colors_v8b = '#FF5722'

# ─── 1. PnL Distribution Histogram ───
ax1 = fig.add_subplot(4, 2, 1)
pnls_v7 = [t['pnl_pct'] for t in v7_all]
pnls_v8b = [t['pnl_pct'] for t in v8b_all]
bins = np.arange(-15, 180, 5)
ax1.hist(pnls_v7, bins=bins, alpha=0.5, label=f'V7 (n={len(pnls_v7)}, avg={np.mean(pnls_v7):+.1f}%)', color=colors_v7, edgecolor='white')
ax1.hist(pnls_v8b, bins=bins, alpha=0.5, label=f'V8b (n={len(pnls_v8b)}, avg={np.mean(pnls_v8b):+.1f}%)', color=colors_v8b, edgecolor='white')
ax1.axvline(np.mean(pnls_v7), color=colors_v7, linestyle='--', linewidth=2, label=f'V7 mean: {np.mean(pnls_v7):+.1f}%')
ax1.axvline(np.mean(pnls_v8b), color=colors_v8b, linestyle='--', linewidth=2, label=f'V8b mean: {np.mean(pnls_v8b):+.1f}%')
ax1.axvline(np.median(pnls_v7), color=colors_v7, linestyle=':', linewidth=2, label=f'V7 median: {np.median(pnls_v7):+.1f}%')
ax1.axvline(np.median(pnls_v8b), color=colors_v8b, linestyle=':', linewidth=2, label=f'V8b median: {np.median(pnls_v8b):+.1f}%')
ax1.set_xlabel('PnL per Trade (%)')
ax1.set_ylabel('Number of Trades')
ax1.set_title('① PnL Distribution (Histogram)')
ax1.legend(fontsize=8)
ax1.set_xlim(-15, 180)

# ─── 2. PnL Box Plot by Version ───
ax2 = fig.add_subplot(4, 2, 2)
bp = ax2.boxplot([pnls_v7, pnls_v8b], labels=['V7', 'V8b'], patch_artist=True, showmeans=True,
                  meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
bp['boxes'][0].set_facecolor(colors_v7); bp['boxes'][0].set_alpha(0.5)
bp['boxes'][1].set_facecolor(colors_v8b); bp['boxes'][1].set_alpha(0.5)
ax2.set_ylabel('PnL per Trade (%)')
ax2.set_title('② PnL Box Plot (diamond = mean)')
ax2.grid(axis='y', alpha=0.3)
# Add text annotations
for i, (label, pnls) in enumerate([(f'V7', pnls_v7), (f'V8b', pnls_v8b)]):
    ax2.text(i+1.3, np.median(pnls), f'Med: {np.median(pnls):+.1f}%', fontsize=9, va='center')
    ax2.text(i+1.3, np.mean(pnls), f'Mean: {np.mean(pnls):+.1f}%', fontsize=9, va='center', color='red')

# ─── 3. PnL Bucket Bar Chart ───
ax3 = fig.add_subplot(4, 2, 3)
bucket_labels = ['<-5%', '-5~-2%', '-2~0%', '0~2%', '2~5%', '5~10%', '10~20%', '20~50%', '50~100%', '>100%']
bucket_ranges = [(-999,-5), (-5,-2), (-2,0), (0,2), (2,5), (5,10), (10,20), (20,50), (50,100), (100,999)]
v7_counts = [sum(1 for p in pnls_v7 if lo <= p < hi) for lo,hi in bucket_ranges]
v8b_counts = [sum(1 for p in pnls_v8b if lo <= p < hi) for lo,hi in bucket_ranges]
x = np.arange(len(bucket_labels))
w = 0.35
ax3.bar(x - w/2, [c/len(pnls_v7)*100 for c in v7_counts], w, label='V7', color=colors_v7, alpha=0.7)
ax3.bar(x + w/2, [c/len(pnls_v8b)*100 for c in v8b_counts], w, label='V8b', color=colors_v8b, alpha=0.7)
ax3.set_xticks(x); ax3.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=9)
ax3.set_ylabel('% of Total Trades')
ax3.set_title('③ Trade Distribution by PnL Bucket')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# ─── 4. Cumulative Profit Contribution ───
ax4 = fig.add_subplot(4, 2, 4)
for label, pnls, color in [('V7', pnls_v7, colors_v7), ('V8b', pnls_v8b, colors_v8b)]:
    sorted_pnls = sorted(pnls, reverse=True)
    cum = np.cumsum(sorted_pnls) / sum(sorted_pnls) * 100
    pct_trades = np.arange(1, len(sorted_pnls)+1) / len(sorted_pnls) * 100
    ax4.plot(pct_trades, cum, color=color, linewidth=2, label=label)
ax4.axhline(80, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(20, color='gray', linestyle='--', alpha=0.5)
ax4.text(21, 82, '20% trades → 80% profit', fontsize=9, color='gray')
ax4.set_xlabel('% of Trades (ranked best to worst)')
ax4.set_ylabel('Cumulative % of Total Profit')
ax4.set_title('④ Profit Concentration (Pareto)')
ax4.legend()
ax4.grid(alpha=0.3)

# ─── 5. Hold Days vs PnL Scatter ───
ax5 = fig.add_subplot(4, 2, 5)
holds_v7 = [t['holding_days'] for t in v7_all]
holds_v8b = [t['holding_days'] for t in v8b_all]
ax5.scatter(holds_v7, pnls_v7, alpha=0.3, s=20, c=colors_v7, label='V7')
ax5.scatter(holds_v8b, pnls_v8b, alpha=0.3, s=20, c=colors_v8b, label='V8b', marker='x')
ax5.axhline(0, color='black', linewidth=0.5)
ax5.set_xlabel('Holding Days')
ax5.set_ylabel('PnL (%)')
ax5.set_title('⑤ Hold Duration vs PnL')
ax5.legend()
ax5.grid(alpha=0.3)

# ─── 6. Exit Reason Breakdown ───
ax6 = fig.add_subplot(4, 2, 6)
for label, trades, color in [('V7', v7_all, colors_v7), ('V8b', v8b_all, colors_v8b)]:
    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t.get('exit_reason', '?')].append(t['pnl_pct'])
    reasons = sorted(by_exit.keys())
    avgs = [np.mean(by_exit[r]) for r in reasons]
    counts = [len(by_exit[r]) for r in reasons]
    x_pos = np.arange(len(reasons))
    offset = -0.2 if label == 'V7' else 0.2
    bars = ax6.bar(x_pos + offset, avgs, 0.35, label=f'{label}', color=color, alpha=0.7)
    for j, (bar, cnt) in enumerate(zip(bars, counts)):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'n={cnt}', 
                ha='center', va='bottom', fontsize=7, rotation=45)
ax6.set_xticks(np.arange(len(reasons)))
ax6.set_xticklabels(reasons, rotation=30, ha='right')
ax6.set_ylabel('Avg PnL (%)')
ax6.set_title('⑥ Avg PnL by Exit Reason')
ax6.legend()
ax6.axhline(0, color='black', linewidth=0.5)
ax6.grid(axis='y', alpha=0.3)

# ─── 7. Per-Symbol Comparison ───
ax7 = fig.add_subplot(4, 2, 7)
v7_sym = defaultdict(list); v8b_sym = defaultdict(list)
for t in v7_all: v7_sym[t['symbol']].append(t['pnl_pct'])
for t in v8b_all: v8b_sym[t['symbol']].append(t['pnl_pct'])
syms = sorted(set(list(v7_sym.keys()) + list(v8b_sym.keys())))
x = np.arange(len(syms))
v7_avgs = [np.mean(v7_sym[s]) if s in v7_sym else 0 for s in syms]
v8b_avgs = [np.mean(v8b_sym[s]) if s in v8b_sym else 0 for s in syms]
ax7.bar(x - 0.2, v7_avgs, 0.35, label='V7', color=colors_v7, alpha=0.7)
ax7.bar(x + 0.2, v8b_avgs, 0.35, label='V8b', color=colors_v8b, alpha=0.7)
ax7.set_xticks(x); ax7.set_xticklabels(syms, rotation=45, ha='right', fontsize=9)
ax7.set_ylabel('Avg PnL per Trade (%)')
ax7.set_title('⑦ Per-Symbol: Avg PnL/Trade')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)
# Mark which is better
for i in range(len(syms)):
    if v8b_avgs[i] > v7_avgs[i]:
        ax7.text(i, max(v7_avgs[i], v8b_avgs[i]) + 1, '✓', ha='center', color='green', fontsize=12, fontweight='bold')

# ─── 8. By Year Performance ───
ax8 = fig.add_subplot(4, 2, 8)
v7_year = defaultdict(list); v8b_year = defaultdict(list)
for t in v7_all: v7_year[t['window'][-4:]].append(t['pnl_pct'])
for t in v8b_all: v8b_year[t['window'][-4:]].append(t['pnl_pct'])
years = sorted(set(list(v7_year.keys()) + list(v8b_year.keys())))
x = np.arange(len(years))
v7_yr_avgs = [np.mean(v7_year[y]) for y in years]
v8b_yr_avgs = [np.mean(v8b_year[y]) for y in years]
ax8.plot(x, v7_yr_avgs, 'o-', color=colors_v7, linewidth=2, markersize=8, label='V7')
ax8.plot(x, v8b_yr_avgs, 's-', color=colors_v8b, linewidth=2, markersize=8, label='V8b')
for i in range(len(years)):
    ax8.text(i, v7_yr_avgs[i]+1, f'{v7_yr_avgs[i]:+.0f}%', ha='center', va='bottom', color=colors_v7, fontsize=9)
    ax8.text(i, v8b_yr_avgs[i]-3, f'{v8b_yr_avgs[i]:+.0f}%', ha='center', va='top', color=colors_v8b, fontsize=9)
ax8.set_xticks(x); ax8.set_xticklabels(years)
ax8.set_ylabel('Avg PnL per Trade (%)')
ax8.set_title('⑧ Performance by Test Year')
ax8.legend()
ax8.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'v7_vs_v8b_analysis.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Chart saved to: {out_path}")
