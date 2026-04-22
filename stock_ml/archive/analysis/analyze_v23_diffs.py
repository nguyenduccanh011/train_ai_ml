"""Cross-model trade analysis. Read-only."""
import pandas as pd, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def load(n):
    df = pd.read_csv(os.path.join(OUT, f"trades_{n}.csv"))
    if "entry_symbol" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "entry_symbol"})
    return df

dfs = {n: load(n) for n in ["v16", "v19_1", "v19_3", "v22", "v23", "rule"]}
for n, d in dfs.items():
    print(f"{n}: cols={list(d.columns)[:8]}... rows={len(d)}")

print("\n========== TOP 5 LOSSES PER MODEL ==========")
for n, d in dfs.items():
    if "pnl_pct" not in d.columns: continue
    cols = [c for c in ["entry_symbol", "entry_date", "exit_date", "pnl_pct", "exit_reason", "entry_trend", "holding_days"] if c in d.columns]
    print(f"\n--- {n} ---")
    print(d.nsmallest(5, "pnl_pct")[cols].to_string(index=False))

print("\n========== PER-SYMBOL TOTALS (entry_date 2022 only - bear year) ==========")
for n, d in dfs.items():
    if "entry_symbol" not in d.columns or "entry_date" not in d.columns: continue
    d2 = d[d["entry_date"].astype(str).str[:4] == "2022"]
    if len(d2) == 0: continue
    g = d2.groupby("entry_symbol")["pnl_pct"].agg(["count", "sum", "mean", lambda x: (x>0).mean()*100]).round(1)
    g.columns = ["n", "total", "avg", "wr"]
    print(f"\n--- {n} 2022 ---  total={d2['pnl_pct'].sum():+.1f}% n={len(d2)}")
    print(g.to_string())

print("\n========== ALIGNED TRADES: same symbol+similar entry_date across V19.1 vs V23 ==========")
v191 = dfs["v19_1"]; v23 = dfs["v23"]
v191["dt"] = pd.to_datetime(v191["entry_date"], errors="coerce")
v23["dt"] = pd.to_datetime(v23["entry_date"], errors="coerce")
diffs = []
for _, t1 in v191.iterrows():
    if pd.isna(t1["dt"]): continue
    cand = v23[(v23["entry_symbol"] == t1["entry_symbol"]) &
               (abs((v23["dt"] - t1["dt"]).dt.days) <= 3)]
    if len(cand) == 0:
        diffs.append((t1["entry_symbol"], str(t1["dt"].date()), t1["pnl_pct"], None, "MISSING_in_V23", t1["exit_reason"], None))
        continue
    t2 = cand.iloc[0]
    diffs.append((t1["entry_symbol"], str(t1["dt"].date()), t1["pnl_pct"], t2["pnl_pct"],
                  f"{t1['exit_reason']}->{t2['exit_reason']}", t1["exit_reason"], t2["exit_reason"]))
ddf = pd.DataFrame(diffs, columns=["sym", "date", "pnl_v191", "pnl_v23", "rsn", "r1", "r2"])
ddf["delta"] = ddf["pnl_v23"] - ddf["pnl_v191"]
print("\nTop 15 trades V23 BETTER than V19.1:")
print(ddf.dropna().nlargest(15, "delta")[["sym","date","pnl_v191","pnl_v23","delta","r1","r2"]].to_string(index=False))
print("\nTop 15 trades V23 WORSE than V19.1:")
print(ddf.dropna().nsmallest(15, "delta")[["sym","date","pnl_v191","pnl_v23","delta","r1","r2"]].to_string(index=False))
print(f"\nMissing in V23 (V19.1 has but V23 skipped): {len(ddf[ddf['pnl_v23'].isna()])}")
miss_pos = ddf[ddf['pnl_v23'].isna() & (ddf['pnl_v191'] > 5)]
print(f"  Of which V19.1 winners (>+5%) skipped: {len(miss_pos)}, sum lost = {miss_pos['pnl_v191'].sum():+.1f}%")
print(miss_pos.nlargest(10, "pnl_v191")[["sym","date","pnl_v191","r1"]].to_string(index=False))

# Same comparison for V23 vs Rule
rule = dfs["rule"]
rule["dt"] = pd.to_datetime(rule["entry_date"], errors="coerce") if "entry_date" in rule.columns else pd.NaT
print("\n========== RULE has but V23 missed (>+10% rule wins) ==========")
if "entry_symbol" in rule.columns:
    rmiss = []
    for _, tr in rule.iterrows():
        if pd.isna(tr.get("dt")): continue
        if tr["pnl_pct"] < 10: continue
        cand = v23[(v23["entry_symbol"] == tr["entry_symbol"]) &
                   (abs((v23["dt"] - tr["dt"]).dt.days) <= 5)]
        if len(cand) == 0:
            rmiss.append((tr["entry_symbol"], str(tr["dt"].date()), tr["pnl_pct"], tr["exit_reason"]))
    rmdf = pd.DataFrame(rmiss, columns=["sym","date","rule_pnl","rule_rsn"])
    print(f"Total rule wins missed by V23: {len(rmdf)}, sum={rmdf['rule_pnl'].sum():+.1f}%")
    print(rmdf.nlargest(20, "rule_pnl").to_string(index=False))

# signal_hard_cap deep dive (V23's biggest issue)
print("\n========== SIGNAL_HARD_CAP DEEP DIVE (V23) ==========")
shc = dfs["v23"][dfs["v23"]["exit_reason"] == "signal_hard_cap"]
print(f"V23 signal_hard_cap: n={len(shc)}, total={shc['pnl_pct'].sum():+.1f}%, avg={shc['pnl_pct'].mean():+.2f}%")
print(shc.groupby("entry_trend").agg(n=("pnl_pct","size"), tot=("pnl_pct","sum"), avg=("pnl_pct","mean")).round(2).to_string())
print("\nFor each V23 signal_hard_cap, what did V19.1 do?")
match = []
for _, t in shc.iterrows():
    cand = v191[(v191["entry_symbol"] == t["entry_symbol"]) &
                (abs((v191["dt"] - pd.to_datetime(t["entry_date"])).dt.days) <= 3)]
    if len(cand): match.append((t["entry_symbol"], t["entry_date"], t["pnl_pct"], cand.iloc[0]["pnl_pct"], cand.iloc[0]["exit_reason"]))
mdf = pd.DataFrame(match, columns=["sym","date","v23_pnl","v191_pnl","v191_rsn"])
mdf["delta"] = mdf["v191_pnl"] - mdf["v23_pnl"]
print(f"\nMatched: {len(mdf)}. V19.1 sum on these = {mdf['v191_pnl'].sum():+.1f}%, V23 sum = {mdf['v23_pnl'].sum():+.1f}%, diff = {mdf['v191_pnl'].sum() - mdf['v23_pnl'].sum():+.1f}%")
print("Top cases V19.1 saved better:")
print(mdf.nlargest(15, "delta").to_string(index=False))

# REE deep dive (V23 -11.4 vs V22 +28, biggest regression)
print("\n========== REE DEEP DIVE (V23 -11.4% vs V22 +28%) ==========")
for n, d in dfs.items():
    sub = d[d.get("entry_symbol", "") == "REE"] if "entry_symbol" in d.columns else pd.DataFrame()
    if len(sub) == 0: continue
    print(f"\n--- {n} REE  total={sub['pnl_pct'].sum():+.1f}% n={len(sub)} ---")
    cols = [c for c in ["entry_date","exit_date","pnl_pct","exit_reason","entry_trend","holding_days"] if c in sub.columns]
    print(sub[cols].sort_values("entry_date").to_string(index=False))

print("\nDONE")
