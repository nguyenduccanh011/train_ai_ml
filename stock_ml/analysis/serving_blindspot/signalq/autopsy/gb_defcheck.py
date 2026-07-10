"""Verify which giveback definition the autopsy n=12/-0.46u split used."""
import pandas as pd
from pathlib import Path

D = pd.read_csv(Path(__file__).parent / "deferred_trades.csv", parse_dates=["entry_date"])
D["gb_entry"] = D.giveback_at_defer  # (peak-close)/entry  (ax_join.py:89)
# fraction-of-peak variant: 1 - close/peak_high = gb_entry / (1+peak_c)
D["gb_peakfrac"] = D.gb_entry / (1.0 + D.peak_c)

for col in ["gb_entry", "gb_peakfrac"]:
    m = D[col] <= 0.10
    sub, rest = D[m], D[~m]
    print(f"{col:12s} <=0.10: n={len(sub)} worse%={100*(sub.dpnl<-1e-9).mean():.1f} "
          f"sum={sub.dpnl.sum():+.3f} | rest n={len(rest)} "
          f"worse%={100*(rest.dpnl<-1e-9).mean():.1f} sum={rest.dpnl.sum():+.3f}")

for s in ["VCI", "VIC"]:
    r = D[(D.symbol == s) & (D.dpnl > 1.0)]
    if len(r):
        r = r.iloc[0]
        print(f"{s}: gb_entry={r.gb_entry:.3f} gb_peakfrac={r.gb_peakfrac:.3f} "
              f"peak_c={r.peak_c:.3f} dpnl={r.dpnl:+.3f}")

# sweep grid preview under both defs (first-order: drop defers with gb < X)
print("\nfirst-order counterfactual (drop defers with gb < X -> their dpnl removed):")
for col in ["gb_entry", "gb_peakfrac"]:
    for x in [0.05, 0.08, 0.10, 0.12]:
        cut = D[D[col] < x]
        keep = D[D[col] >= x]
        print(f"  {col} X={x:.2f}: cut n={len(cut)} dpnl_removed={cut.dpnl.sum():+.3f} "
              f"kept dpnl={keep.dpnl.sum():+.3f} megas_kept="
              f"{int(((keep.symbol.isin(['VCI','VIC'])) & (keep.dpnl>1.0)).sum())}")
