"""Deep analysis of why avg PnL per trade is so low."""
import json, os, re
import numpy as np
from collections import defaultdict

viz_dir = 'visualization/data'
all_pnls = []
all_holds = []
all_details = []
sym_stats = []

for f in sorted(os.listdir(viz_dir)):
    if not f.endswith('.json') or f == 'index.json':
        continue
    sym = f.replace('.json', '')
    d = json.load(open(os.path.join(viz_dir, f)))
    markers = d.get('markers', [])
    
    pnls = []
    holds = []
    wins_big = 0  # >5%
    wins_small = 0  # 0-5%
    losses_small = 0  # 0 to -5%
    losses_big = 0  # < -5%
    early_exits = 0  # hold < 7 days
    
    for i in range(0, len(markers), 2):
        if i + 1 >= len(markers):
            break
        sell = markers[i + 1]
        text = sell.get('text', '')
        
        # Extract PnL
        match = re.search(r'([+-]?\d+\.?\d*)%', text)
        if not match:
            continue
        pnl = float(match.group(1))
        
        # Extract hold days
        hold_match = re.search(r'\((\d+)d\)', text)
        hold = int(hold_match.group(1)) if hold_match else 0
        
        pnls.append(pnl)
        holds.append(hold)
        all_pnls.append(pnl)
        all_holds.append(hold)
        all_details.append({'sym': sym, 'pnl': pnl, 'hold': hold, 'text': text})
        
        if pnl > 5: wins_big += 1
        elif pnl > 0: wins_small += 1
        elif pnl > -5: losses_small += 1
        else: losses_big += 1
        
        if hold <= 7: early_exits += 1
    
    if pnls:
        sym_stats.append({
            'sym': sym,
            'trades': len(pnls),
            'avg_pnl': np.mean(pnls),
            'med_pnl': np.median(pnls),
            'avg_hold': np.mean(holds),
            'wins_big': wins_big,
            'wins_small': wins_small,
            'losses_small': losses_small,
            'losses_big': losses_big,
            'early_exits': early_exits,
            'early_pct': early_exits / len(pnls) * 100,
        })

# Overall summary
print("=" * 90)
print("TRADE QUALITY ANALYSIS — WHY IS AVG PNL SO LOW?")
print("=" * 90)

all_pnls = np.array(all_pnls)
all_holds = np.array(all_holds)

print(f"\nTotal trades: {len(all_pnls)}")
print(f"Avg PnL: {np.mean(all_pnls):+.2f}%")
print(f"Median PnL: {np.median(all_pnls):+.2f}%")
print(f"Avg hold: {np.mean(all_holds):.1f} days")
print(f"Median hold: {np.median(all_holds):.0f} days")

print(f"\n--- PNL DISTRIBUTION ---")
bins = [(-100, -10), (-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5), (5, 10), (10, 20), (20, 100)]
for lo, hi in bins:
    count = np.sum((all_pnls >= lo) & (all_pnls < hi))
    pct = count / len(all_pnls) * 100
    bar = "█" * int(pct)
    print(f"  [{lo:+4d}%, {hi:+4d}%): {count:4d} ({pct:5.1f}%) {bar}")

print(f"\n--- HOLD PERIOD DISTRIBUTION ---")
hold_bins = [(1, 3), (3, 7), (7, 14), (14, 21), (21, 42), (42, 100)]
for lo, hi in hold_bins:
    count = np.sum((all_holds >= lo) & (all_holds < hi))
    pct = count / len(all_holds) * 100
    avg_p = np.mean(all_pnls[(all_holds >= lo) & (all_holds < hi)]) if count > 0 else 0
    print(f"  [{lo:2d}-{hi:2d}d): {count:4d} trades ({pct:5.1f}%), avg PnL: {avg_p:+.2f}%")

print(f"\n--- KEY PROBLEMS ---")
short_trades = all_pnls[all_holds <= 7]
long_trades = all_pnls[all_holds > 14]
print(f"Short trades (≤7d): {len(short_trades)} trades, avg PnL: {np.mean(short_trades):+.2f}%")
print(f"Long trades (>14d): {len(long_trades)} trades, avg PnL: {np.mean(long_trades):+.2f}%")

tiny_wins = all_pnls[(all_pnls > 0) & (all_pnls < 3)]
big_wins = all_pnls[all_pnls >= 10]
big_losses = all_pnls[all_pnls <= -5]
print(f"\nTiny wins (0-3%): {len(tiny_wins)} trades — wasted opportunities")
print(f"Big wins (≥10%): {len(big_wins)} trades, avg: {np.mean(big_wins):+.1f}%")
print(f"Big losses (≤-5%): {len(big_losses)} trades, avg: {np.mean(big_losses):+.1f}%")

print(f"\n--- PER-SYMBOL BREAKDOWN (sorted by avg_pnl) ---")
sym_stats.sort(key=lambda x: x['avg_pnl'])
for s in sym_stats:
    flag = "🔴" if s['avg_pnl'] < 0 else ("🟡" if s['avg_pnl'] < 2 else "🟢")
    print(f"  {flag} {s['sym']:5s}: {s['trades']:3d} trades, avg={s['avg_pnl']:+5.1f}%, "
          f"med={s['med_pnl']:+5.1f}%, hold={s['avg_hold']:4.1f}d, "
          f"early_exit={s['early_pct']:.0f}%, "
          f"big_wins={s['wins_big']}, big_loss={s['losses_big']}")

print(f"\n--- ROOT CAUSE SUMMARY ---")
early_pct = np.sum(all_holds <= 7) / len(all_holds) * 100
print(f"1. EXIT TOO EARLY: {early_pct:.0f}% of trades exit within 7 days")
print(f"   → Short holds don't capture big moves")
tiny_win_pct = len(tiny_wins) / np.sum(all_pnls > 0) * 100
print(f"2. TINY WINS: {tiny_win_pct:.0f}% of winning trades gain < 3%")
print(f"   → Winners are cut short, losers run similar duration")
wr = np.sum(all_pnls > 0) / len(all_pnls) * 100
print(f"3. WIN RATE: {wr:.1f}% — not high enough to compensate small wins")
avg_win = np.mean(all_pnls[all_pnls > 0])
avg_loss = np.mean(all_pnls[all_pnls <= 0])
print(f"4. RISK/REWARD: avg win={avg_win:+.2f}%, avg loss={avg_loss:+.2f}%, ratio={abs(avg_win/avg_loss):.2f}")

print(f"\n--- RECOMMENDATIONS TO ACHIEVE >10% AVG PNL ---")
print("1. HIGHER ENTRY THRESHOLD: Only enter on strongest signals (score ≥ 0.7+)")
print("2. LET WINNERS RUN: Use trailing stop instead of fixed exit after N days")
print("3. WIDER TIME HORIZON: Min hold 10+ days, max hold 60+ days")
print("4. FILTER MARKET REGIME: Only trade in uptrend market conditions")
print("5. REDUCE TRADE FREQUENCY: Fewer but higher-conviction trades")
print("6. DYNAMIC STOP-LOSS: Tighter stops (-3%) to cut losses fast")
