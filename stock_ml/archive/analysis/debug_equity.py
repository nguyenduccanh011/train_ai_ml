"""Trace equity during MBB trade 3 to find the +104% bug."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '.')
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
raw_df = loader.load_all(symbols=['MBB'])
engine = FeatureEngine(feature_set='leading')
df = engine.compute_for_all_symbols(raw_df)
df = target_gen.generate_for_all_symbols(df)
feature_cols = engine.get_feature_columns(df)
df = df.dropna(subset=feature_cols + ['target'])

# Only first window (test 2020)
for window, train_df, test_df in splitter.split(df):
    sym_test = test_df[test_df['symbol'] == 'MBB'].reset_index(drop=True)
    if len(sym_test) < 10:
        continue
    
    model = build_model('lightgbm')
    X_train = np.nan_to_num(train_df[feature_cols].values)
    y_train = train_df['target'].values.astype(int)
    model.fit(X_train, y_train)
    
    X_sym = np.nan_to_num(sym_test[feature_cols].values)
    y_pred = model.predict(X_sym)
    rets = sym_test['return_1d'].values
    close = sym_test['close'].values
    dates = sym_test['timestamp'].values
    
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = 100_000_000
    position = 0
    entry_equity = 0
    
    print(f"\nWindow: {window.label}")
    unique, counts = np.unique(y_pred.astype(int), return_counts=True)
    print(f"Days: {n}, Predictions: {dict(zip(unique, counts))}")
    print(f"\nTracing position changes and equity:")
    
    for i in range(1, n):
        pred = int(y_pred[i-1])
        ret = rets[i] if not np.isnan(rets[i]) else 0
        new_pos = 1 if pred == 1 else 0
        
        # Simplified - no entry/exit filters for diagnosis
        cost = 0
        if new_pos != position:
            if new_pos == 1:
                cost = equity[i-1] * 0.0015
                entry_equity = equity[i-1] - cost
                print(f"  DAY {i} {str(dates[i])[:10]} BUY  close={close[i]:.2f} equity={equity[i-1]:,.0f} entry_eq={entry_equity:,.0f}")
            else:
                cost = equity[i-1] * 0.0025
                exit_eq = equity[i-1] - cost
                pnl_pct = (exit_eq - entry_equity) / entry_equity * 100 if entry_equity > 0 else 0
                price_entry_close = close[max(0, i - (i - 0))]  # approximate
                print(f"  DAY {i} {str(dates[i])[:10]} SELL close={close[i]:.2f} equity={equity[i-1]:,.0f} exit_eq={exit_eq:,.0f} entry_eq={entry_equity:,.0f} pnl={pnl_pct:+.2f}%")
        
        if position == 1:
            equity[i] = equity[i-1] * (1 + ret) - cost
        else:
            equity[i] = equity[i-1] - cost
        position = new_pos
    
    print(f"\nFinal equity: {equity[-1]:,.0f}")
    print(f"Total return: {(equity[-1]/100_000_000-1)*100:+.2f}%")
    
    # Now run actual V7 backtest for comparison
    from run_v7_compare import backtest_v7
    r7 = backtest_v7(y_pred, rets, sym_test, feature_cols)
    print(f"\nV7 backtest trades ({len(r7['trades'])}):")
    for t in r7['trades']:
        print(f"  {t.get('entry_date','?')} -> {t.get('exit_date','?')} pnl={t['pnl_pct']:+.2f}% hold={t['holding_days']}d")
    print(f"V7 total return: {r7['total_return_pct']:+.2f}%")
    
    break  # Only first window
