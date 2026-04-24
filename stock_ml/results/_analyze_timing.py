"""Phan tich entry/exit timing: V37a vs Rule tren cac symbol thua."""
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

# Focus on top losing symbols
LOSERS = ['PVS','AAS','BSR','PVD','KBC','AAV','GAS','FRT','BCM','PLX','SBT','BID']

# For each losing symbol: find overlapping trades
# Definition "overlap": rule trade period overlaps with v37a trade period
def find_overlaps(vs, rs):
    """Return list of (v_trade, r_trade_nearest) for analysis."""
    out = []
    for _, vt in vs.iterrows():
        ve, vx = vt["entry_date"], vt["exit_date"]
        # rule trades that START within [ve-30d, vx+30d] - nearest
        nearby = rs[(rs["entry_date"] >= ve - pd.Timedelta(days=30)) &
                    (rs["entry_date"] <= vx + pd.Timedelta(days=30))]
        if len(nearby) == 0:
            out.append((vt, None))
            continue
        # pick the rule trade whose entry closest to v entry
        diffs = (nearby["entry_date"] - ve).abs()
        idx = diffs.idxmin()
        rt = nearby.loc[idx]
        out.append((vt, rt))
    return out

print("="*100)
print("ENTRY/EXIT TIMING: V37a vs RULE (per-trade pairing)")
print("="*100)

records = []
for sym in LOSERS:
    vs = v[v["symbol"] == sym].sort_values("entry_date").reset_index(drop=True)
    rs = r[r["symbol"] == sym].sort_values("entry_date").reset_index(drop=True)
    pairs = find_overlaps(vs, rs)
    for vt, rt in pairs:
        rec = {
            "symbol": sym,
            "v_entry": vt["entry_date"].date(),
            "v_exit": vt["exit_date"].date(),
            "v_pnl": vt["pnl_pct"],
            "v_hold": vt["holding_days"],
            "v_exit_reason": vt["exit_reason"],
            "v_max_profit": vt.get("max_profit_pct", np.nan),
            "v_profile": vt.get("entry_profile", ""),
        }
        if rt is not None:
            rec.update({
                "r_entry": rt["entry_date"].date(),
                "r_exit": rt["exit_date"].date(),
                "r_pnl": rt["pnl_pct"],
                "r_hold": rt["holding_days"],
                "r_exit_reason": rt["exit_reason"],
                "entry_delay": (vt["entry_date"] - rt["entry_date"]).days,  # >0: V vao sau rule
                "exit_delay": (vt["exit_date"] - rt["exit_date"]).days,     # >0: V thoat sau rule
            })
        else:
            rec.update({"r_entry":None,"r_exit":None,"r_pnl":np.nan,"r_hold":np.nan,
                        "r_exit_reason":"","entry_delay":np.nan,"exit_delay":np.nan})
        records.append(rec)

df = pd.DataFrame(records)
print(f"Total paired rows: {len(df)}")

# Aggregated stats
paired = df.dropna(subset=["r_entry"])
print(f"\nV37a trades with rule nearby: {len(paired)}/{len(df)}")
print(f"Mean entry_delay (V vs Rule, positive=V late): {paired['entry_delay'].mean():.1f} days")
print(f"Mean exit_delay (V vs Rule, positive=V late): {paired['exit_delay'].mean():.1f} days")
print(f"% V entry AFTER rule: {(paired['entry_delay']>0).mean()*100:.1f}%")
print(f"% V entry BEFORE rule: {(paired['entry_delay']<0).mean()*100:.1f}%")
print(f"% V exit AFTER rule (hold lau hon): {(paired['exit_delay']>0).mean()*100:.1f}%")
print(f"% V exit BEFORE rule (thoat som hon): {(paired['exit_delay']<0).mean()*100:.1f}%")

# When V lose but rule win (same period)
bad = paired[(paired["v_pnl"] < -3) & (paired["r_pnl"] > 0)]
print(f"\nV thua >=3% NHUNG rule lai an same period: {len(bad)}")
print(bad.sort_values("v_pnl").head(20).round(2).to_string())

df.to_csv(os.path.join(ROOT, "_v37a_vs_rule_pairs.csv"), index=False)
print(f"\nSaved: _v37a_vs_rule_pairs.csv ({len(df)} rows)")
