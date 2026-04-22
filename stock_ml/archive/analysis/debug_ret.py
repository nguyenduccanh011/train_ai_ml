"""Check return_1d scale vs actual price changes."""
import sys, os, numpy as np
sys.path.insert(0, '.')
from src.data.loader import DataLoader
from src.features.engine import FeatureEngine

data_dir = os.path.join('..', 'portable_data', 'vn_stock_ai_dataset_cleaned')
loader = DataLoader(data_dir)
raw_df = loader.load_all(symbols=['MBB'])
engine = FeatureEngine(feature_set='leading')
df = engine.compute_for_all_symbols(raw_df)
print("Columns:", list(df.columns)[:20])
date_col = 'date' if 'date' in df.columns else 'timestamp'
mbb = df[df['symbol'] == 'MBB'].sort_values(date_col).reset_index(drop=True)

close = mbb['close'].values
ret1d = mbb['return_1d'].values

# Manual return
manual = (close[1:] - close[:-1]) / close[:-1]

print("return_1d vs manual close-to-close return:")
for i in range(1, 11):
    print(f"  Day {i}: close={close[i]:.2f} return_1d={ret1d[i]:.6f} manual={(close[i]-close[i-1])/close[i-1]:.6f} ratio={ret1d[i]/manual[i-1]:.2f}x" if manual[i-1] != 0 else f"  Day {i}: close={close[i]:.2f} return_1d={ret1d[i]:.6f} manual=0")

# Check around 2020-09-29 to 2020-10-28 (trade 3)
mask_sept = (mbb[date_col] >= '2020-09-29') & (mbb[date_col] <= '2020-10-28')
sub = mbb[mask_sept]
print(f"\n2020-09-29 to 2020-10-28 ({len(sub)} days):")
print(f"  Close: {sub['close'].iloc[0]:.2f} -> {sub['close'].iloc[-1]:.2f} = {(sub['close'].iloc[-1]/sub['close'].iloc[0]-1)*100:+.2f}%")
print(f"  Sum return_1d: {sub['return_1d'].sum()*100:.2f}%")
print(f"  Compound return_1d: {(np.prod(1 + sub['return_1d'].values[1:]) - 1)*100:.2f}%")
print(f"  return_1d values:")
for _, row in sub.iterrows():
    print(f"    {str(row[date_col])[:10]} close={row['close']:.2f} ret={row['return_1d']:.6f}")

# Check all columns that might be return
print(f"\nAll return-like columns: {[c for c in mbb.columns if 'return' in c.lower() or 'ret' in c.lower()]}")
