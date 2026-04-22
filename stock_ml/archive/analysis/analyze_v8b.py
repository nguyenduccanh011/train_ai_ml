"""Deep analysis of V8b trade distribution."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_v8b_compare import backtest_v8b
from run_v7_compare import backtest_v7
from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model

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

for label, trades in [("V7", v7_all), ("V8b", v8b_all)]:
    pnls = sorted([t['pnl_pct'] for t in trades])
    print(f"\n{'='*80}")
    print(f"📊 {label} DEEP ANALYSIS ({len(pnls)} trades)")
    print(f"{'='*80}")
    print(f"Mean: {np.mean(pnls):+.2f}%  Median: {np.median(pnls):+.2f}%  Std: {np.std(pnls):.2f}%")
    print(f"Min: {min(pnls):+.2f}%  Max: {max(pnls):+.2f}%")
    
    # Percentiles
    for p in [10, 25, 50, 75, 90]:
        print(f"  P{p}: {np.percentile(pnls, p):+.1f}%", end="")
    print()

    # Distribution
    buckets = [(-999,-10), (-10,-5), (-5,-2), (-2,0), (0,2), (2,5), (5,10), (10,20), (20,50), (50,100), (100,999)]
    print(f"\nPnL Distribution:")
    for lo,hi in buckets:
        cnt = sum(1 for p in pnls if lo <= p < hi)
        if cnt == 0: continue
        avg = np.mean([p for p in pnls if lo <= p < hi])
        tot = sum(p for p in pnls if lo <= p < hi)
        pct = cnt/len(pnls)*100
        bar = "█" * int(pct)
        print(f"  [{lo:+4d}% to {hi:+4d}%): {cnt:4d} ({pct:5.1f}%) avg={avg:+6.1f}% total={tot:+8.0f}% {bar}")

    # Winners vs Losers
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    print(f"\nWinners: {len(wins)} avg={np.mean(wins):+.1f}% median={np.median(wins):+.1f}%")
    print(f"Losers:  {len(losses)} avg={np.mean(losses):+.1f}% median={np.median(losses):+.1f}%")
    if losses:
        print(f"Win/Loss ratio: {np.mean(wins)/abs(np.mean(losses)):.1f}x")

    # Top 20% contribution
    top20_n = max(len(pnls)//5, 1)
    top20 = sorted(pnls, reverse=True)[:top20_n]
    total = sum(pnls)
    print(f"\nTop 20% ({top20_n} trades): total={sum(top20):+.0f}% ({sum(top20)/total*100:.0f}% of all profit)")
    bottom80 = sorted(pnls, reverse=True)[top20_n:]
    print(f"Bottom 80% ({len(bottom80)} trades): total={sum(bottom80):+.0f}%")

    # By exit reason
    from collections import defaultdict
    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t.get('exit_reason', '?')].append(t['pnl_pct'])
    print(f"\nBy exit reason:")
    for reason, ps in sorted(by_exit.items(), key=lambda x: -len(x[1])):
        print(f"  {reason:15s}: {len(ps):4d} trades, avg={np.mean(ps):+6.1f}%, total={sum(ps):+8.0f}%")

    # By window (year)
    by_win = defaultdict(list)
    for t in trades:
        by_win[t.get('window', '?')].append(t['pnl_pct'])
    print(f"\nBy window/year:")
    for w, ps in sorted(by_win.items()):
        print(f"  {w:20s}: {len(ps):4d} trades, avg={np.mean(ps):+6.1f}%, WR={sum(1 for p in ps if p>0)/len(ps)*100:.0f}%")

    # Trades with max_profit but ended low (missed profits)
    missed = [t for t in trades if t.get('max_profit_pct', 0) > 10 and t['pnl_pct'] < t.get('max_profit_pct', 0) * 0.3]
    if missed:
        print(f"\n⚠️ Missed profit trades (max>10% but exited <30% of max): {len(missed)}")
        for t in sorted(missed, key=lambda x: x.get('max_profit_pct',0)-x['pnl_pct'], reverse=True)[:5]:
            print(f"  {t['symbol']} {t.get('entry_date','?')} max={t.get('max_profit_pct',0):+.1f}% → exit={t['pnl_pct']:+.1f}% ({t['exit_reason']}) hold={t['holding_days']}d")
