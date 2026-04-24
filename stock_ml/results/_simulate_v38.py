"""V38 offline simulation tren trades_v37a.csv.
Goal: dung gia tri max_profit/exit features de uoc luong PnL neu ap dung
fix dexuat. KHONG can re-train ML.

Han che: khong re-enter sau exit som; gia tri sau exit som la xap xi tu
price_max_profit_pct va exit_price.
"""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
v = pd.read_csv(os.path.join(ROOT, "trades_v37a.csv"))

print(f"Baseline V37a: n={len(v)}, total_pnl={v['pnl_pct'].sum():.0f}%, "
      f"avg={v['pnl_pct'].mean():.2f}%, WR={(v['pnl_pct']>0).mean()*100:.1f}%, "
      f"max_loss={v['pnl_pct'].min():.1f}%")

PROFILES_BAD = {"balanced"}  # 16/21 losing symbols la balanced

# ============================================================
# V38a: TAT relax flags cho 'balanced' -> simulate bang cach
# loai bo trade cua balanced ma chac chan duoc tao boi relax (entry chat
# proxy: ko co breakout/vshape va co dist_sma20 > 4 hoac entry_ret_5d > 5)
# Day la heuristic gan dung.
# ============================================================
def sim_v38a(v):
    """Loai bo trades 'balanced' giong dau hieu vao do relax flags lam ra:
    - entry chat (khong breakout, khong vshape) + dist_sma20 > 3% hoac entry_ret_5d > 4%
    """
    is_bal = v['entry_profile'] == 'balanced'
    no_special_entry = (v['breakout_entry'] == False) & (v['vshape_entry'] == False)
    is_late = (v['entry_dist_sma20'] > 3) | (v['entry_ret_5d'] > 4)
    drop_mask = is_bal & no_special_entry & is_late
    return v[~drop_mask].copy()

# ============================================================
# V38b: HAP nhay hon (3% trigger, -2% floor) + stall-exit
# Simulate: khong co granular bar data, dung exit_reason hien tai +
# heuristic. Trades thua >=8% co price_max>=3% va pnl<=-2 -> gia su moi
# exit som hon, capture ~ -2% (thay vi pnl hien tai)
# ============================================================
def sim_v38b(v):
    out = v.copy()
    # HAP nhay hon: trades co price_max>=3 nhung roi sau do
    cond_hap = (out['price_max_profit_pct'] >= 3) & (out['pnl_pct'] < -2)
    out.loc[cond_hap, 'pnl_pct'] = -2.0  # Gia su exit nhay tai -2% tu peak
    # Stall exit: trades hold>=10 ngay nhung max profit < 2% -> exit som tai 0
    cond_stall = (out['holding_days'] >= 10) & (out['price_max_profit_pct'] < 2) & (out['pnl_pct'] < -3)
    out.loc[cond_stall, 'pnl_pct'] = -1.5  # Gia su cat o -1.5%
    return out

# ============================================================
# V38c: HA-driven exit
# Khong co per-bar HA features trong CSV (chi co exit_*). Simulate gia dinh
# exit_above_ema8=0 + exit_above_sma20=0 + pnl<0 + hold>=3 -> exit ngay khi
# vua co loss (uoc luong: loss = max(-3, current_pnl))
# ============================================================
def sim_v38c(v):
    out = v.copy()
    # neu trade ket cuc thua va exit dieu kien xau, gia su HA bearish_reversal fire som
    cond = ((out['exit_above_sma20'] == 0) & (out['exit_above_ema8'] == 0) &
            (out['pnl_pct'] < -3) & (out['holding_days'] >= 5))
    # Gia su HA fire luc loss khoang -3%
    out.loc[cond & (out['pnl_pct'] < -3), 'pnl_pct'] = -3.0
    return out

# ============================================================
# V38d: Anti-fomo entry + rule co-pilot exit
# Filter: bo trades vao luc entry_ret_5d>6 hoac dist_sma20>6 (tru breakout/vshape)
# Co-pilot: thay vi giu loi nhuan, bat ky exit_above_sma20=0 + price_max>=5 -> exit nhay
# ============================================================
def sim_v38d(v):
    out = v.copy()
    # Filter entry fomo
    drop = ((out['entry_ret_5d'] > 6) | (out['entry_dist_sma20'] > 6)) & \
           (out['breakout_entry'] == False) & (out['vshape_entry'] == False)
    out = out[~drop].copy()
    # Rule co-pilot: trade lam dau (price_max>=5) ma exit_above_sma20=0 -> exit nhay
    cond = (out['price_max_profit_pct'] >= 5) & (out['exit_above_sma20'] == 0) & (out['pnl_pct'] < 0)
    # Gia su exit som luc dat 50% cua peak (capture half profit)
    out.loc[cond, 'pnl_pct'] = out.loc[cond, 'price_max_profit_pct'] * 0.5 - 1.5  # minus commission
    return out


def report(name, df, base_n=len(v), base_total=v['pnl_pct'].sum()):
    n = len(df)
    pnl = df['pnl_pct']
    total = pnl.sum()
    avg = pnl.mean()
    wr = (pnl > 0).mean() * 100
    pf_g = pnl[pnl > 0].sum()
    pf_l = abs(pnl[pnl < 0].sum())
    pf = pf_g / pf_l if pf_l > 0 else 0
    max_loss = pnl.min()
    print(f"{name:12s} n={n:5d} ({n-base_n:+d})  total={total:+7.0f}% ({total-base_total:+5.0f})  "
          f"avg={avg:+5.2f}%  WR={wr:5.1f}%  PF={pf:.2f}  max_loss={max_loss:+6.1f}%")

print()
print("="*100)
print("V38 SIMULATIONS (offline, heuristic)")
print("="*100)
report("V37a base", v)
report("V38a", sim_v38a(v))
report("V38b", sim_v38b(v))
report("V38c", sim_v38c(v))
report("V38d", sim_v38d(v))
# Combined V38abcd
def sim_combined(v):
    s = sim_v38a(v)
    s = sim_v38b(s)
    s = sim_v38c(s)
    s = sim_v38d(s)
    return s
report("V38abcd", sim_combined(v))

# Per-losing-symbol impact of best variant
print()
print("="*100)
print("PER-LOSING-SYMBOL: V37a vs V38abcd")
print("="*100)
LOSERS = ['PVS','AAS','BSR','PVD','KBC','AAV','GAS','FRT','BCM','PLX','SBT','BID']
combined = sim_combined(v)
r = pd.read_csv(os.path.join(ROOT, "trades_rule.csv"))
rows = []
for sym in LOSERS:
    base = v[v['symbol']==sym]['pnl_pct'].sum()
    fix = combined[combined['symbol']==sym]['pnl_pct'].sum()
    rule = r[r['symbol']==sym]['pnl_pct'].sum()
    rows.append({'sym': sym, 'V37a': base, 'V38abcd': fix, 'Rule': rule,
                 'V38_vs_V37a': fix-base, 'V38_vs_Rule': fix-rule})
print(pd.DataFrame(rows).round(1).to_string(index=False))
