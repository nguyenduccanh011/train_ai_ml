import json, numpy as np

with open("New folder (7)/stock_signals.json", "r") as f:
    data = json.load(f)

acb = data["symbols"]["ACB"]
trades = acb["trades"]

print("=" * 80)
print("ACB TRADES ANALYSIS")
print("=" * 80)
for i, t in enumerate(trades):
    pnl = t["pnl_pct"]
    tag = "WIN" if pnl > 0 else "LOSS"
    print(f"{i+1:2d}. [{tag:4s}] Entry: {t['entry_date']}  Exit: {t['exit_date']}  "
          f"Hold: {t['holding_days']:3d}d  PnL: {pnl:+7.2f}%  Reason: {t['exit_reason']}")

wins = [t for t in trades if t["pnl_pct"] > 0]
losses = [t for t in trades if t["pnl_pct"] <= 0]
print(f"\nWins: {len(wins)}, Avg win: {np.mean([t['pnl_pct'] for t in wins]):+.2f}%")
print(f"Losses: {len(losses)}, Avg loss: {np.mean([t['pnl_pct'] for t in losses]):+.2f}%")
print(f"Big losses (<-5%): {sum(1 for t in trades if t['pnl_pct'] < -5)}")
print(f"Small wins (0-3%): {sum(1 for t in trades if 0 < t['pnl_pct'] < 3)}")
print(f"Big wins (>10%): {sum(1 for t in trades if t['pnl_pct'] > 10)}")

# OHLCV context
ohlcv = {d["time"]: d for d in acb["ohlcv"]}
print("\n--- Key OHLCV prices ---")
dates = ["2020-01-06","2020-02-13","2020-02-14","2020-03-09","2020-03-23",
         "2020-04-01","2020-05-27","2020-06-02","2020-06-16","2020-06-17"]
for d in dates:
    if d in ohlcv:
        r = ohlcv[d]
        print(f"  {d}: O={r['open']} H={r['high']} L={r['low']} C={r['close']}")

# All symbols summary
print("\n" + "=" * 80)
print("ALL SYMBOLS SUMMARY")
print("=" * 80)
for r in data["rankings"]:
    marker = " ***" if r["symbol"] in data["symbols"] else ""
    print(f"  {r['symbol']:6s}  Trades={r['trades']:3d}  WR={r['win_rate']:5.1f}%  "
          f"AvgPnL={r['avg_pnl']:+6.2f}%  TotalPnL={r['total_pnl']:+8.2f}%{marker}")

print(f"\nTotal symbols tested: {len(data['rankings'])}")
print(f"Profitable symbols (total_pnl>0): {sum(1 for r in data['rankings'] if r['total_pnl']>0)}")
print(f"Unprofitable: {sum(1 for r in data['rankings'] if r['total_pnl']<=0)}")
