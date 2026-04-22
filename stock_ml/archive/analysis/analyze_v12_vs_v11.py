"""
Phân tích chi tiết V12 vs V11: tìm giao dịch zombie, nhiễu, điểm mua không hợp lý,
và đề xuất cải thiện.
"""
import json, os, glob
from collections import defaultdict

data_dir = os.path.join(os.path.dirname(__file__), 'visualization', 'data')

total_v12_pnl = 0
total_v11_pnl = 0
all_v12_trades = []
all_v11_trades = []
zombie_trades = []
noise_trades = []  # hold <= 3 days, small pnl
big_loss_trades = []
missed_v11_wins = []  # V11 wins that V12 missed

symbols = []
for f in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
    if 'index.json' in f:
        continue
    with open(f) as fh:
        d = json.load(fh)
    sym = os.path.basename(f).replace('.json','')
    symbols.append(sym)
    
    v12t = d.get('v12_trades', [])
    v11t = d.get('v11_trades', [])
    v12s = d.get('v12_stats', {})
    v11s = d.get('v11_stats', {})
    
    v12_pnl = v12s.get('total_pnl_pct', 0)
    v11_pnl = v11s.get('total_pnl_pct', 0)
    total_v12_pnl += v12_pnl
    total_v11_pnl += v11_pnl
    
    # Collect all trades with symbol
    for t in v12t:
        t['symbol'] = sym
        all_v12_trades.append(t)
        # Zombie trades
        if t.get('exit_reason','') in ('ZE','Zombie') or t.get('holding_days',0) >= 14 and t.get('pnl_pct',0) < -2:
            zombie_trades.append(t)
        # Noise trades
        if t.get('holding_days',0) <= 3 and abs(t.get('pnl_pct',0)) < 1.5:
            noise_trades.append(t)
        # Big losses
        if t.get('pnl_pct',0) < -5:
            big_loss_trades.append(t)
    
    for t in v11t:
        t['symbol'] = sym
        all_v11_trades.append(t)
    
    # Find V11 winning trades that V12 didn't capture
    v12_entries = set(t.get('entry_date','') for t in v12t)
    for t in v11t:
        if t.get('pnl_pct',0) > 5 and t.get('entry_date','') not in v12_entries:
            missed_v11_wins.append(t)

print("="*80)
print("PHÂN TÍCH CHI TIẾT V12 vs V11")
print("="*80)

print(f"\n📊 TỔNG QUAN:")
print(f"  V12 Total PnL: {total_v12_pnl:+.1f}%  ({len(all_v12_trades)} trades)")
print(f"  V11 Total PnL: {total_v11_pnl:+.1f}%  ({len(all_v11_trades)} trades)")
print(f"  Chênh lệch: {total_v12_pnl - total_v11_pnl:+.1f}%")
print(f"  V12 Avg PnL/trade: {total_v12_pnl/max(len(all_v12_trades),1):.2f}%")
print(f"  V11 Avg PnL/trade: {total_v11_pnl/max(len(all_v11_trades),1):.2f}%")

# Exit reason distribution
v12_reasons = defaultdict(int)
v12_reason_pnl = defaultdict(float)
for t in all_v12_trades:
    r = t.get('exit_reason','Unknown')
    v12_reasons[r] += 1
    v12_reason_pnl[r] += t.get('pnl_pct', 0)

print(f"\n📋 V12 EXIT REASON DISTRIBUTION:")
for r, cnt in sorted(v12_reasons.items(), key=lambda x: -x[1]):
    avg = v12_reason_pnl[r] / cnt
    print(f"  {r:20s}: {cnt:3d} trades, Total: {v12_reason_pnl[r]:+7.1f}%, Avg: {avg:+.2f}%")

v11_reasons = defaultdict(int)
v11_reason_pnl = defaultdict(float)
for t in all_v11_trades:
    r = t.get('exit_reason','Unknown')
    v11_reasons[r] += 1
    v11_reason_pnl[r] += t.get('pnl_pct', 0)

print(f"\n📋 V11 EXIT REASON DISTRIBUTION:")
for r, cnt in sorted(v11_reasons.items(), key=lambda x: -x[1]):
    avg = v11_reason_pnl[r] / cnt
    print(f"  {r:20s}: {cnt:3d} trades, Total: {v11_reason_pnl[r]:+7.1f}%, Avg: {avg:+.2f}%")

# Zombie trades
print(f"\n🧟 ZOMBIE & LONG-HOLD LỖ ({len(zombie_trades)} trades):")
for t in sorted(zombie_trades, key=lambda x: x.get('pnl_pct',0)):
    print(f"  {t['symbol']:5s} {t.get('entry_date',''):10s} → {t.get('exit_date',''):10s} "
          f"Hold:{t.get('holding_days',0):3d}d  PnL:{t.get('pnl_pct',0):+.1f}%  "
          f"Reason:{t.get('exit_reason','')} Trend:{t.get('entry_trend','')}")

# Noise trades
print(f"\n📡 NOISE TRADES (hold<=3d, |pnl|<1.5%) ({len(noise_trades)} trades):")
noise_total = sum(t.get('pnl_pct',0) for t in noise_trades)
print(f"  Tổng PnL noise: {noise_total:+.1f}%")
for t in noise_trades[:15]:
    print(f"  {t['symbol']:5s} {t.get('entry_date',''):10s} → {t.get('exit_date',''):10s} "
          f"Hold:{t.get('holding_days',0):3d}d  PnL:{t.get('pnl_pct',0):+.1f}%  "
          f"Reason:{t.get('exit_reason','')}")

# Big losses
print(f"\n💀 BIG LOSS TRADES (pnl < -5%) ({len(big_loss_trades)} trades):")
big_loss_total = sum(t.get('pnl_pct',0) for t in big_loss_trades)
print(f"  Tổng PnL big loss: {big_loss_total:+.1f}%")
for t in sorted(big_loss_trades, key=lambda x: x.get('pnl_pct',0)):
    print(f"  {t['symbol']:5s} {t.get('entry_date',''):10s} → {t.get('exit_date',''):10s} "
          f"Hold:{t.get('holding_days',0):3d}d  PnL:{t.get('pnl_pct',0):+.1f}%  "
          f"Reason:{t.get('exit_reason','')} Trend:{t.get('entry_trend','')}")

# Missed V11 wins
print(f"\n🎯 V11 WINNING TRADES MÀ V12 BỎ LỠ (pnl>5%) ({len(missed_v11_wins)} trades):")
missed_total = sum(t.get('pnl_pct',0) for t in missed_v11_wins)
print(f"  Tổng PnL bỏ lỡ: {missed_total:+.1f}%")
for t in sorted(missed_v11_wins, key=lambda x: -x.get('pnl_pct',0))[:25]:
    print(f"  {t['symbol']:5s} {t.get('entry_date',''):10s} → {t.get('exit_date',''):10s} "
          f"Hold:{t.get('holding_days',0):3d}d  PnL:{t.get('pnl_pct',0):+.1f}%  "
          f"Reason:{t.get('exit_reason','')} Trend:{t.get('entry_trend','')}")

# V12 early exits (V12 exited early vs V11 on same entry)
print(f"\n⏱️ V12 EXIT SỚM HƠN V11 (cùng entry, V12 pnl < V11 pnl):")
early_exit_loss = 0
count = 0
for sym in symbols:
    v12_by_entry = {}
    v11_by_entry = {}
    for t in all_v12_trades:
        if t['symbol'] == sym:
            v12_by_entry[t.get('entry_date','')] = t
    for t in all_v11_trades:
        if t['symbol'] == sym:
            v11_by_entry[t.get('entry_date','')] = t
    for ed in v12_by_entry:
        if ed in v11_by_entry:
            v12p = v12_by_entry[ed].get('pnl_pct',0)
            v11p = v11_by_entry[ed].get('pnl_pct',0)
            diff = v12p - v11p
            if diff < -3:
                t12 = v12_by_entry[ed]
                t11 = v11_by_entry[ed]
                early_exit_loss += diff
                count += 1
                if count <= 20:
                    print(f"  {sym:5s} Entry:{ed:10s} V12:{v12p:+.1f}%(h{t12.get('holding_days',0)}d,{t12.get('exit_reason','')}) "
                          f"V11:{v11p:+.1f}%(h{t11.get('holding_days',0)}d,{t11.get('exit_reason','')}) Diff:{diff:+.1f}%")
print(f"  => Total lost from early exits: {early_exit_loss:+.1f}% ({count} trades)")

# Per-symbol comparison
print(f"\n📈 PER-SYMBOL COMPARISON:")
print(f"  {'Symbol':6s} {'V12 PnL':>8s} {'V11 PnL':>8s} {'Diff':>8s} {'V12 WR':>7s} {'V11 WR':>7s} {'V12#':>5s} {'V11#':>5s}")
for f in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
    if 'index.json' in f:
        continue
    with open(f) as fh:
        d = json.load(fh)
    sym = os.path.basename(f).replace('.json','')
    v12s = d.get('v12_stats', {})
    v11s = d.get('v11_stats', {})
    v12p = v12s.get('total_pnl_pct',0)
    v11p = v11s.get('total_pnl_pct',0)
    print(f"  {sym:6s} {v12p:+8.1f} {v11p:+8.1f} {v12p-v11p:+8.1f} {v12s.get('win_rate',0):6.1f}% {v11s.get('win_rate',0):6.1f}% {v12s.get('total_trades',0):5d} {v11s.get('total_trades',0):5d}")

print(f"\n{'='*80}")
print("💡 ĐỀ XUẤT CẢI THIỆN V12:")
print("="*80)
print("""
1. PROFIT LOCK quá sớm: V12 lock profit tại 12% threshold với floor 6%, 
   khiến nhiều trades bị cắt sớm khi đang uptrend mạnh. 
   → Đề xuất: Tăng PROFIT_LOCK_THRESHOLD=0.18, PROFIT_LOCK_MIN=0.10 trong strong trend.

2. TIME DECAY trailing quá aggressive: Sau MIN_HOLD, trailing tightens quá nhanh
   → Đề xuất: Chỉ apply time-decay khi RSI < 50 hoặc MACD turning negative.

3. ZOMBIE detection cần cải thiện: Một số zombie trades vẫn hold quá lâu
   → Đề xuất: Giảm ZOMBIE_BARS từ 14 xuống 10 khi trend = sideways.

4. ENTRY FILTER quá chặt: V12 bỏ lỡ nhiều V11 winning entries do:
   - RSI slope cap quá thấp → nới từ 0.7 lên 1.0
   - Consolidation breakout filter quá strict
   - Cooldown after big loss quá dài (5 bars) → giảm xuống 3

5. NOISE TRADES: Nhiều trades hold 1-3 ngày cho PnL rất nhỏ, tốn commission
   → Đề xuất: Thêm min expected move filter trước khi entry.

6. MISSED OPPORTUNITIES: Nhiều V11 big wins V12 không bắt được
   → Đề xuất: Thêm momentum reentry signal khi price vượt SMA20 + volume surge
     mà không cần đợi consolidation.
""")
