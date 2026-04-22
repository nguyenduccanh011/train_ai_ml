"""Debug MBB: compare backtest PnL vs chart PnL."""
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from run_v7_compare import backtest_v7
from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model

data_dir = os.path.join('..', 'portable_data', 'vn_stock_ai_dataset_cleaned')
config = {
    'data': {'data_dir': data_dir},
    'split': {'method': 'walk_forward', 'train_years': 4, 'test_years': 1,
              'gap_days': 0, 'first_test_year': 2020, 'last_test_year': 2025},
    'target': {'type': 'trend_regime', 'trend_method': 'dual_ma',
               'short_window': 10, 'long_window': 40, 'classes': 3},
}
loader = DataLoader(data_dir)
splitter = WalkForwardSplitter.from_config(config)
target_gen = TargetGenerator.from_config(config)
symbols = ['MBB']
raw_df = loader.load_all(symbols=symbols)
engine = FeatureEngine(feature_set='leading')
df = engine.compute_for_all_symbols(raw_df)
df = target_gen.generate_for_all_symbols(df)
feature_cols = engine.get_feature_columns(df)
df = df.dropna(subset=feature_cols + ['target'])

v7_all = []
for window, train_df, test_df in splitter.split(df):
    model = build_model('lightgbm')
    X_train = np.nan_to_num(train_df[feature_cols].values)
    y_train = train_df['target'].values.astype(int)
    model.fit(X_train, y_train)
    for sym in test_df['symbol'].unique():
        sym_test = test_df[test_df['symbol'] == sym].reset_index(drop=True)
        if len(sym_test) < 10:
            continue
        X_sym = np.nan_to_num(sym_test[feature_cols].values)
        y_pred = model.predict(X_sym)
        rets = sym_test['return_1d'].values
        r7 = backtest_v7(y_pred, rets, sym_test, feature_cols)
        for t in r7['trades']:
            t['window'] = window.label
        v7_all.extend(r7['trades'])

print(f"MBB V7 backtest: {len(v7_all)} trades")
for t in v7_all:
    ed = t.get('entry_date', '?')
    xd = t.get('exit_date', '?')
    pp = t['pnl_pct']
    hd = t['holding_days']
    er = t['exit_reason']
    print(f"  {ed} -> {xd}  pnl={pp:+.2f}%  hold={hd}d  exit={er}")

pnls = [t['pnl_pct'] for t in v7_all]
print(f"\nAvg: {np.mean(pnls):+.2f}%  Sum: {sum(pnls):+.1f}%")

# Now compare with chart data
print("\n" + "=" * 60)
print("CHART DATA (visualization/data/MBB.json)")
print("=" * 60)
try:
    d = json.load(open('visualization/data/MBB.json'))
    ohlcv = {c['time']: c for c in d['ohlcv']}
    markers = d['markers']
    chart_trades = []
    for i in range(0, len(markers), 2):
        if i + 1 < len(markers):
            buy = markers[i]
            sell = markers[i + 1]
            bp = ohlcv.get(buy['time'], {}).get('close', 0)
            sp = ohlcv.get(sell['time'], {}).get('close', 0)
            if bp > 0:
                pnl = (sp - bp) / bp * 100
                chart_trades.append(pnl)
                print(f"  {buy['time']} -> {sell['time']}  pnl={pnl:+.1f}%")
    print(f"\nChart Avg: {np.mean(chart_trades):+.2f}%  Sum: {sum(chart_trades):+.1f}%")
except Exception as e:
    print(f"Error: {e}")

# KEY: Check if backtest uses COMPOUNDING equity
print("\n" + "=" * 60)
print("KEY INSIGHT: Equity compounding effect")
print("=" * 60)
# The backtest tracks equity (starts at 100M).
# Each trade's pnl_pct = (exit_equity - entry_equity) / entry_equity * 100
# But equity compounds! After winning trades, equity grows.
# A +1% on 200M is pnl_pct=+1% but absolute is 2M (vs 1M on 100M)
# The pnl_pct per trade should still be correct as price-based %
# Let's verify: simulate manually

print("\nManual verification for first trade:")
if v7_all:
    t0 = v7_all[0]
    print(f"  Entry day: {t0.get('entry_day', '?')}, Exit day: {t0.get('exit_day', '?')}")
    print(f"  Holding: {t0['holding_days']}d")
    print(f"  PnL%: {t0['pnl_pct']:+.2f}%")
    
    # Look at the equity curve
    # The pnl_pct in V7 backtest is computed as:
    # pnl = exit_eq - entry_equity
    # pnl_pct = pnl / entry_equity * 100
    # entry_equity = deploy - commission = equity[i-1] * position_size - commission
    # So this is the % return on invested capital for that trade
    # This should match price-based return!
    
    # Unless... equity is not reset between windows!
    print(f"\n  Window: {t0.get('window', '?')}")
