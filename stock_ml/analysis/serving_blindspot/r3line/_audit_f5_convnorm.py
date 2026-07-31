# -*- coding: utf-8 -*-
"""AUDIT finding #5: full-sample z-scoring of conviction weights in hb_deploy_* harnesses.
Rebuilds cs5_ma50 conv panel identically to hb_deploy_gt.py lines 186-204, attaches conv to
base-run trades (template/x2_struct_to-69338138), replicates _prep() full-sample weight formula
(lines 42-48), and compares against a causal expanding-window implementation."""
import statistics, sys
import psycopg2, pandas as pd, numpy as np, duckdb

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
KCONV, SKIP = 2.0, 0.40

# ---- panel (verbatim logic from hb_deploy_gt.py) ----
cx = duckdb.connect(MARKET, read_only=True)
px = cx.execute("SELECT symbol,date,low,close,high FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol,date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); parts = []
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").copy(); c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1; g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9)); g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c; g["dist_ma50"] = c / c.rolling(50).mean() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
CSm = {(r.symbol, str(r.date.date())): (r.cs5_ma50 if pd.notna(r.cs5_ma50) else 0.5) for r in P.itertuples()}

# ---- base run trades (= sim.trades population; NavSim2 date_lo=2020-01-01) ----
con = psycopg2.connect(**PG)
tr = pd.read_sql("""select symbol, entry_date, exit_date, entry_signal_date, pnl_pct
                    from run_trades where run_id='template/x2_struct_to-69338138'
                    and exit_date is not null order by entry_date, symbol""", con)
tr["entry_date"] = pd.to_datetime(tr["entry_date"]); tr["sigd"] = pd.to_datetime(tr["entry_signal_date"])
tr = tr[tr["entry_date"] >= "2020-01-01"].reset_index(drop=True)
tr["conv"] = [CSm.get((r.symbol, str(r.sigd.date())), 0.5) for r in tr.itertuples()]
n = len(tr); print(f"base-run trades >=2020: {n}; conv from panel: non-default {sum(1 for r in tr.itertuples() if (r.symbol, str(r.sigd.date())) in CSm)}/{n}")

# ---- 1. yearly conv drift (non-stationarity) ----
tr["year"] = tr["entry_date"].dt.year
print("\nYearly conv mean/std (trade-level):")
for y, g in tr.groupby("year"):
    print(f"  {y}: mean={g.conv.mean():.3f} std={g.conv.std():.3f} n={len(g)}")
mu_full = statistics.mean(tr["conv"]); sd_full = statistics.pstdev(tr["conv"]) or 1.0
print(f"FULL-SAMPLE mu={mu_full:.4f} sd={sd_full:.4f}")

# ---- 2. full-sample weights (exact harness formula, lines 42-48) ----
def wfull(cv_all):
    mu = statistics.mean(cv_all); sd = statistics.pstdev(cv_all) or 1.0
    raw = [min(max(1.0 + KCONV * (c - mu) / sd, 0.4), 1.8) for c in cv_all]
    off = 1.0 - statistics.mean(raw)
    return [max(0.3, w + off) for w in raw]

tr["w_full"] = wfull(tr["conv"].tolist())

# ---- 3. causal expanding-window weights ----
# at each trade, stats over trades with entry_date <= this trade's entry_date (incl. same-day; strictly-prior is even more divergent)
convs = tr["conv"].values; edays = tr["entry_date"].values
w_exp = np.empty(n)
csum = 0.0; csq = 0.0; wsum = 0.0; cnt = 0
# group by day so all same-day trades use the same expanding stats (through that day)
i = 0
run_conv = []
run_rawmean_num = 0.0; run_rawmean_den = 0
while i < n:
    j = i
    while j < n and edays[j] == edays[i]:
        j += 1
    # include current day's trades in stats (info available at close of entry day's decision on signal date - conservative; day's convs are known at signal time)
    day_convs = convs[i:j].tolist(); run_conv.extend(day_convs)
    mu = statistics.mean(run_conv); sd = statistics.pstdev(run_conv) or 1.0
    raw_all = [min(max(1.0 + KCONV * (c - mu) / sd, 0.4), 1.8) for c in run_conv]
    off = 1.0 - statistics.mean(raw_all)
    for k in range(i, j):
        w = min(max(1.0 + KCONV * (convs[k] - mu) / sd, 0.4), 1.8)
        w_exp[k] = max(0.3, w + off)
    i = j
tr["w_exp"] = w_exp

# ---- 4. quantify divergence ----
tr["dw"] = tr["w_full"] - tr["w_exp"]; tr["adw"] = tr["dw"].abs()
print(f"\n|w_full - w_expanding|: mean={tr.adw.mean():.4f} median={tr.adw.median():.4f} max={tr.adw.max():.4f}")
print(f"frac trades |dw|>0.05: {(tr.adw > 0.05).mean():.3f}  >0.10: {(tr.adw > 0.10).mean():.3f}  >0.20: {(tr.adw > 0.20).mean():.3f}")
print(f"rank corr (w_full vs w_exp): {tr[['w_full','w_exp']].corr(method='spearman').iloc[0,1]:.4f}")
print("\nYearly mean|dw| and mean dw (sign = full-sample bias direction):")
for y, g in tr.groupby("year"):
    print(f"  {y}: mean|dw|={g.adw.mean():.4f}  mean dw={g.dw.mean():+.4f}  max|dw|={g.adw.max():.3f}")

# ---- 5. direction: does look-ahead normalization inflate? ----
tr["pnl"] = tr["pnl_pct"].astype(float)
m = tr["pnl"].notna()
print(f"\ncorr(dw, pnl) = {np.corrcoef(tr.loc[m,'dw'], tr.loc[m,'pnl'])[0,1]:+.4f}")
print(f"sum(w_full*pnl) = {(tr.loc[m,'w_full']*tr.loc[m,'pnl']).sum():.2f}  vs sum(w_exp*pnl) = {(tr.loc[m,'w_exp']*tr.loc[m,'pnl']).sum():.2f}  vs flat = {tr.loc[m,'pnl'].sum():.2f}")

# ---- 6. SKIP gate interaction: does full-sample vs expanding change which trades pass conv>=0.40? ----
# (SKIP applies to raw conv, not w — verify from code: _entries uses t['conv'] < SKIP, causal. Just confirm.)
print(f"\nSKIP gate uses raw conv (causal): trades conv<0.40: {(tr.conv < SKIP).sum()}")

# ---- 7. early-window magnification: first 60 trades ----
head = tr.head(60)
print(f"first-60-trades (early 2020): mean|dw|={head.adw.mean():.4f} max={head.adw.max():.3f}")
print("sample first 8 trades:")
print(head[["symbol", "entry_date", "conv", "w_full", "w_exp", "dw"]].head(8).to_string(index=False))
con.close()
print("\nAUDIT_F5_DONE")
