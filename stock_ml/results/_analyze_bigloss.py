"""Phan tich trades thua nang >8% cua V37a + tim 'du dinh song'."""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
v = pd.read_csv(os.path.join(ROOT, "trades_v37a.csv"))
r = pd.read_csv(os.path.join(ROOT, "trades_rule.csv"))
v["entry_date"] = pd.to_datetime(v["entry_date"])
v["exit_date"] = pd.to_datetime(v["exit_date"])
r["entry_date"] = pd.to_datetime(r["entry_date"])
r["exit_date"] = pd.to_datetime(r["exit_date"])

print("="*100)
print("V37a: TRADES THUA NANG >= 8%")
print("="*100)
big_loss = v[v["pnl_pct"] <= -8].sort_values("pnl_pct")
print(f"Tong: {len(big_loss)} trades, total loss: {big_loss['pnl_pct'].sum():.1f}%")
print(f"Mean max_profit: {big_loss['max_profit_pct'].mean():.1f}%")
print(f"Mean holding: {big_loss['holding_days'].mean():.1f} days")

# Group by exit_reason
print("\n--- Exit reason cua trades thua nang ---")
print(big_loss["exit_reason"].value_counts().to_string())

# Group by profile
print("\n--- Profile cua trades thua nang ---")
print(big_loss["entry_profile"].value_counts().to_string())

# How many had GIANT max_profit (du dinh song)
print("\n--- 'DU DINH SONG': max_profit >= 100% nhung exit <= -8% ---")
top_wave = big_loss[big_loss["max_profit_pct"] >= 100].sort_values("max_profit_pct", ascending=False)
print(f"So trades du dinh song nang: {len(top_wave)}/{len(big_loss)}")
print(top_wave[["symbol","entry_date","exit_date","holding_days","pnl_pct","max_profit_pct","exit_reason","entry_profile"]].head(25).to_string())

# 'Bo lo' = max_profit cao nhung pnl <= 0
print("\n--- TRADES BO LO LOI NHUAN (max_profit>=20% nhung pnl<=0) ---")
miss = v[(v["max_profit_pct"] >= 20) & (v["pnl_pct"] <= 0)]
print(f"So trades bo lo loi nhuan: {len(miss)} ({len(miss)/len(v)*100:.1f}% tong)")
print(f"  Total max_profit cua nhom nay: {miss['max_profit_pct'].sum():.1f}%")
print(f"  Total pnl cua nhom nay: {miss['pnl_pct'].sum():.1f}%")
print(f"  -> Bo lo: {miss['max_profit_pct'].sum() - miss['pnl_pct'].sum():.1f}%")

# Comparable rule trades trong cung period
print("\n--- VOI TRADES THUA >=8% CUA V37: RULE LAM GI? ---")
recs = []
for _, vt in big_loss.iterrows():
    sym = vt["symbol"]
    rs = r[(r["symbol"]==sym) &
           (r["entry_date"] >= vt["entry_date"] - pd.Timedelta(days=15)) &
           (r["exit_date"] <= vt["exit_date"] + pd.Timedelta(days=15))]
    for _, rt in rs.iterrows():
        recs.append({
            "symbol": sym,
            "v_entry": vt["entry_date"].date(),
            "v_exit": vt["exit_date"].date(),
            "v_pnl": vt["pnl_pct"],
            "v_maxp": vt["max_profit_pct"],
            "v_reason": vt["exit_reason"],
            "r_entry": rt["entry_date"].date(),
            "r_exit": rt["exit_date"].date(),
            "r_pnl": rt["pnl_pct"],
            "r_reason": rt["exit_reason"],
            "r_exit_before_v_exit_days": (vt["exit_date"] - rt["exit_date"]).days,
        })
df_cmp = pd.DataFrame(recs)
if len(df_cmp):
    rule_first = df_cmp[df_cmp["r_exit_before_v_exit_days"] > 0]
    rule_first_win = rule_first[rule_first["r_pnl"] > 0]
    print(f"Trong {len(df_cmp)} cap V_loss/Rule overlap:")
    print(f"  Rule thoat TRUOC V: {len(rule_first)} ({len(rule_first)/len(df_cmp)*100:.0f}%)")
    print(f"  Rule thoat truoc + an duoc: {len(rule_first_win)} ({len(rule_first_win)/len(rule_first)*100:.0f}% so cai thoat truoc)")
    print(f"  Tong PnL rule cua group rule_first_win: +{rule_first_win['r_pnl'].sum():.1f}%")
    print(f"  Tong PnL V37a cua cac trade tuong ung: {df_cmp[df_cmp['r_exit_before_v_exit_days']>0]['v_pnl'].sum():.1f}%")

df_cmp.to_csv(os.path.join(ROOT, "_v37a_bigloss_vs_rule.csv"), index=False)
