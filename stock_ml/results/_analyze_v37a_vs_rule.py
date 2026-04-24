"""Phan tich V37a vs Rule - khong chinh sua gi, chi doc va in ra."""
import pandas as pd
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
v = pd.read_csv(os.path.join(ROOT, "trades_v37a.csv"))
r = pd.read_csv(os.path.join(ROOT, "trades_rule.csv"))

# Normalize
v = v.rename(columns={"entry_symbol": "symbol_v"})
v["symbol"] = v["symbol"].astype(str)
r["symbol"] = r["symbol"].astype(str)
v["entry_date"] = pd.to_datetime(v["entry_date"])
v["exit_date"] = pd.to_datetime(v["exit_date"])
r["entry_date"] = pd.to_datetime(r["entry_date"])
r["exit_date"] = pd.to_datetime(r["exit_date"])

print("="*80)
print("OVERALL")
print("="*80)
for name, df in [("V37a", v), ("Rule", r)]:
    n = len(df)
    wr = (df["pnl_pct"] > 0).mean() * 100
    tot = df["pnl_pct"].sum()
    avg = df["pnl_pct"].mean()
    med = df["pnl_pct"].median()
    max_loss = df["pnl_pct"].min()
    print(f"{name}: n={n}, WR={wr:.1f}%, total={tot:.1f}%, avg={avg:.2f}%, median={med:.2f}%, max_loss={max_loss:.2f}%")

# Per-symbol PnL
print("\n" + "="*80)
print("PER-SYMBOL: V37a TOTAL PnL vs RULE TOTAL PnL")
print("="*80)
agg_v = v.groupby("symbol").agg(v_n=("pnl_pct","count"), v_tot=("pnl_pct","sum"),
                                v_avg=("pnl_pct","mean"), v_wr=("pnl_pct", lambda s: (s>0).mean()*100))
agg_r = r.groupby("symbol").agg(r_n=("pnl_pct","count"), r_tot=("pnl_pct","sum"),
                                r_avg=("pnl_pct","mean"), r_wr=("pnl_pct", lambda s: (s>0).mean()*100))
m = agg_v.join(agg_r, how="outer").fillna(0)
m["diff"] = m["v_tot"] - m["r_tot"]
m = m.sort_values("diff")

print("\n--- SYMBOLS V37a THUA RULE (diff < 0) ---")
lose = m[m["diff"] < 0].copy()
print(f"So symbol V37a thua Rule: {len(lose)}/{len(m)}")
print(lose.round(2).to_string())

print("\n--- TOP 15 SYMBOLS V37a THUA NHIEU NHAT ---")
print(lose.head(15).round(2).to_string())

# Save for later phases
m.to_csv(os.path.join(ROOT, "_v37a_vs_rule_per_symbol.csv"))
print(f"\nSaved: _v37a_vs_rule_per_symbol.csv")
