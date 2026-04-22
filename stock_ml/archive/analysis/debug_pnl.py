"""Debug: verify PnL calculation is correct by comparing with chart data."""
import sys, os, json, numpy as np
sys.path.insert(0, '.')

# 1. Check what the visualization chart actually shows
print("=" * 70)
print("1. CHART DATA (visualization/data/MBB.json)")
print("=" * 70)

try:
    d = json.load(open('visualization/data/MBB.json'))
    markers = d['markers']
    ohlcv = {c['time']: c for c in d['ohlcv']}
    
    trades = []
    for i in range(0, len(markers), 2):
        if i + 1 < len(markers):
            buy = markers[i]
            sell = markers[i + 1]
            bp = ohlcv.get(buy['time'], {}).get('close', 0)
            sp = ohlcv.get(sell['time'], {}).get('close', 0)
            if bp > 0:
                pnl = (sp - bp) / bp * 100
                trades.append(pnl)
                print(f"  Buy {buy['time']} @{bp:.2f} -> Sell {sell['time']} @{sp:.2f} | PnL: {pnl:+.1f}%")
    
    print(f"\n  Total trades: {len(trades)}")
    print(f"  Avg PnL: {np.mean(trades):+.2f}%")
    print(f"  Sum PnL: {sum(trades):+.1f}%")
except Exception as e:
    print(f"  Error: {e}")

# 2. Check the backtest CSV
print("\n" + "=" * 70)
print("2. BACKTEST CSV (latest)")
print("=" * 70)

import glob
csvs = sorted(glob.glob('results/backtest_*.csv'))
if csvs:
    import pandas as pd
    df = pd.read_csv(csvs[-1])
    mbb = df[df['symbol'] == 'MBB']
    print(f"  File: {csvs[-1]}")
    print(f"  MBB trades: {len(mbb)}")
    if 'pnl_pct' in mbb.columns:
        print(f"  Avg pnl_pct: {mbb['pnl_pct'].mean():+.2f}%")
        print(f"  Sum pnl_pct: {mbb['pnl_pct'].sum():+.1f}%")
        print(f"\n  First 10 trades:")
        for _, row in mbb.head(10).iterrows():
            cols = ['entry_date', 'exit_date', 'pnl_pct', 'exit_reason']
            vals = {c: row.get(c, '?') for c in cols}
            print(f"    {vals}")

# 3. Check backtest function directly
print("\n" + "=" * 70)
print("3. BACKTEST FUNCTION - pnl_pct calculation")
print("=" * 70)

from src.evaluation.backtest import Backtester
import inspect
src = inspect.getsource(Backtester)
# Find pnl_pct calculation
for i, line in enumerate(src.split('\n')):
    if 'pnl' in line.lower() and ('=' in line or 'return' in line.lower()):
        print(f"  Line {i}: {line.strip()}")

# 4. Run a quick manual check
print("\n" + "=" * 70)
print("4. MANUAL VERIFICATION")
print("=" * 70)

from src.data.loader import DataLoader
data_dir = os.path.join('..', 'portable_data', 'vn_stock_ai_dataset_cleaned')
loader = DataLoader(data_dir)
raw_df = loader.load_all(symbols=['MBB'])
mbb_data = raw_df[raw_df['symbol'] == 'MBB'].sort_values('date').reset_index(drop=True)

print(f"  return_1d stats:")
r1d = mbb_data['return_1d']
print(f"    mean: {r1d.mean():.6f} ({r1d.mean()*100:.4f}%)")
print(f"    std:  {r1d.std():.6f}")
print(f"    min:  {r1d.min():.6f}")
print(f"    max:  {r1d.max():.6f}")

# Check: is return_1d in decimal (0.01 = 1%) or percentage (1.0 = 1%)?
close = mbb_data['close'].values
manual_ret = (close[1:] - close[:-1]) / close[:-1]
print(f"\n  Manual return[1]: {manual_ret[0]:.6f}")
print(f"  return_1d[1]:     {mbb_data['return_1d'].iloc[1]:.6f}")
print(f"  Ratio: {mbb_data['return_1d'].iloc[1] / manual_ret[0]:.2f}x")

# If return_1d is in decimal form (0.01), summing them gives ~same as price change
# But if backtest uses sum(return_1d) * 100 for "pnl_pct", that's wrong
# It should use compound: prod(1+r) - 1

# Simulate a 10-day hold
idx = 500
hold_days = 10
price_pnl = (close[idx + hold_days] - close[idx]) / close[idx]
sum_ret = sum(mbb_data['return_1d'].values[idx + 1:idx + hold_days + 1])
compound_ret = np.prod(1 + mbb_data['return_1d'].values[idx + 1:idx + hold_days + 1]) - 1

print(f"\n  10-day hold from day {idx}:")
print(f"    Price-based PnL: {price_pnl * 100:+.2f}%")
print(f"    Sum of return_1d: {sum_ret * 100:+.2f}%")
print(f"    Compound return_1d: {compound_ret * 100:+.2f}%")

print("\n" + "=" * 70)
print("5. KEY QUESTION: How does backtest accumulate PnL?")
print("=" * 70)
# Show the actual backtest trade computation
try:
    from src.evaluation import backtest
    src_lines = inspect.getsource(backtest).split('\n')
    in_trade = False
    for i, line in enumerate(src_lines):
        if 'pnl' in line.lower() or 'trade' in line.lower() or 'position' in line.lower():
            print(f"  {i:4d}: {line.rstrip()}")
except:
    pass
