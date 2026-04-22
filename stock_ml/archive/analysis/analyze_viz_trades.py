"""Analyze all trades from visualization data (close-price based PnL)."""
import json, os, numpy as np
from collections import defaultdict

viz_dir = os.path.join(os.path.dirname(__file__), "visualization", "data")
all_trades = []

for f in sorted(os.listdir(viz_dir)):
    if not f.endswith(".json") or f == "index.json":
        continue
    path = os.path.join(viz_dir, f)
    d = json.load(open(path))
    if "ohlcv" not in d or "markers" not in d:
        continue
    
    sym = f.replace(".json", "")
    close_map = {c["time"]: c["close"] for c in d["ohlcv"]}
    markers = d["markers"]
    
    for i in range(0, len(markers) - 1, 2):
        buy, sell = markers[i], markers[i + 1]
        bp = close_map.get(buy["time"], 0)
        sp = close_map.get(sell["time"], 0)
        if bp > 0 and sp > 0:
            pnl = (sp / bp - 1) * 100
            all_trades.append({"sym": sym, "pnl": pnl, "entry": buy["time"], "exit": sell["time"]})

pnls = [t["pnl"] for t in all_trades]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]

print(f"{'='*60}")
print(f"📊 V7 BACKTEST SUMMARY (from visualization charts)")
print(f"{'='*60}")
print(f"Total trades: {len(all_trades)}")
print(f"Symbols: {len(set(t['sym'] for t in all_trades))}")
print(f"Win rate: {len(wins)/len(all_trades)*100:.1f}%")
print(f"Avg PnL: {np.mean(pnls):+.2f}%")
print(f"Median PnL: {np.median(pnls):+.2f}%")
print(f"Avg win: {np.mean(wins):+.2f}%")
print(f"Avg loss: {np.mean(losses):+.2f}%")
print(f"R:R ratio: {abs(np.mean(wins)/np.mean(losses)):.2f}")
print(f"Gross wins: {sum(wins):+.1f}%")
print(f"Gross losses: {sum(losses):+.1f}%")
print(f"PF: {sum(wins)/abs(sum(losses)):.2f}")
print()
print(f"Tiny wins (0-3%): {sum(1 for p in wins if p<3)}")
print(f"Medium wins (3-10%): {sum(1 for p in wins if 3<=p<10)}")
print(f"Big wins (>=10%): {sum(1 for p in wins if p>=10)}")
print(f"Small losses (0 to -3%): {sum(1 for p in losses if p>-3)}")
print(f"Medium losses (-3 to -5%): {sum(1 for p in losses if -5<=p<-3)}")
print(f"Big losses (<=-5%): {sum(1 for p in losses if p<=-5)}")

# Per-symbol
print(f"\n{'='*60}")
print(f"PER-SYMBOL BREAKDOWN")
print(f"{'='*60}")
by_sym = defaultdict(list)
for t in all_trades:
    by_sym[t["sym"]].append(t["pnl"])

print(f"{'Symbol':<8} {'#':>4} {'WR':>6} {'AvgPnL':>8} {'TotPnL':>9}")
for sym in sorted(by_sym.keys()):
    ps = by_sym[sym]
    wr = sum(1 for p in ps if p > 0) / len(ps) * 100
    print(f"{sym:<8} {len(ps):>4} {wr:>5.1f}% {np.mean(ps):>+7.2f}% {sum(ps):>+8.1f}%")
