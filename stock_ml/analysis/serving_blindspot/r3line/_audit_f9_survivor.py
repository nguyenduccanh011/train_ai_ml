# _audit_f9_survivor.py — verify finding #9: survivor-filtered DuckDB panel poisons
# breadth gate (exit_force_gate_lowbreadth pct_above_ma50<0.25), breadth features
# (entry_ensemble3), and the deploy-harness cs5_ma50 conviction panel (skip<0.40).
import duckdb, psycopg2, pandas as pd, numpy as np

MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MISSING = ["FLC", "ROS", "HAI", "AMD", "KLF", "GAB", "PVX"]

con = duckdb.connect(MARKET, read_only=True)
oh = con.execute("SELECT symbol, date, low, close, high FROM ohlcv WHERE timeframe='1D' AND date>='2018-06-01' ORDER BY symbol, date").fetchdf()
con.close()
oh["date"] = pd.to_datetime(oh["date"])
assert not set(MISSING) & set(oh["symbol"].unique()), "missing cluster unexpectedly present"

# ---- 1) breadth pct_above_ma50 exactly as _load_market_breadth ----
piv = oh.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
ma = piv.rolling(50, min_periods=50).mean()
ind = piv > ma
n_active = ind.notna().sum(axis=1).clip(lower=1)
breadth = ind.sum(axis=1) / n_active
# counterfactual: 7 missing names present and BELOW MA50 (2022 collapse: limit-down chains)
b_cf_low = ind.sum(axis=1) / (n_active + len(MISSING))
delta_low = breadth - b_cf_low
THR = 0.25
gate_now = breadth < THR
gate_cf = b_cf_low < THR
flip = gate_now != gate_cf
print("[breadth] dates:", len(breadth), " gate<0.25 days now:", int(gate_now.sum()),
      " counterfactual(7 below-MA50):", int(gate_cf.sum()), " FLIP days:", int(flip.sum()))
print("[breadth] flip dates:", [str(d.date()) for d in breadth.index[flip]][:20])
print("[breadth] mean delta near-threshold band (b in .22-.30):",
      round(float(delta_low[(breadth > .22) & (breadth < .30)].mean()), 5))
print("[breadth] 2022 min breadth:", round(float(breadth["2022"].min()), 4),
      "days<0.25 in 2022:", int(gate_now["2022"].sum()))

# ---- 2) cs5_ma50 conviction panel exactly as hb_deploy_gt.py ----
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
parts = []
for s, g in oh.groupby("symbol"):
    g = g.sort_values("date").copy()
    c, l, h = g["close"], g["low"], g["high"]
    dd = c.diff(); up = dd.clip(lower=0).rolling(14).mean(); dn = (-dd.clip(upper=0)).rolling(14).mean()
    g["dist20low"] = c / l.rolling(20).min() - 1
    g["dist_ma20"] = c / c.rolling(20).mean() - 1
    g["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    g["ret20"] = c / c.shift(20) - 1
    tr_ = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    g["atrpct"] = tr_.rolling(14).mean() / c
    g["dist_ma50"] = c / c.rolling(50).mean() - 1
    parts.append(g[["symbol", "date"] + CS4 + ["atrpct", "dist_ma50"]])
P = pd.concat(parts, ignore_index=True)
for col in CS4 + ["atrpct", "dist_ma50"]:
    P[col + "_r"] = P.groupby("date")[col].rank(pct=True)
P["cs5_ma50"] = P[[c + "_r" for c in CS4] + ["atrpct_r", "dist_ma50_r"]].mean(axis=1)
P["n_date"] = P.groupby("date")["symbol"].transform("size")

# per-date rank shift bound if 7 extreme names were added: 7/(n+7) per rank column
P["rank_shift_max"] = 7.0 / (P["n_date"] + 7.0)

# ---- 3) join to actual audited run entries ----
pg = psycopg2.connect(**PG)
RIDS = [
    "template/x2_struct_to_k10_cs5ma50_r7ec_gtos-69338138",
    "template/x2_struct_to_k10_cs5ma50_r7ec_osdef-69338138",
    "template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138",
    "template/x2_struct_to_k10_cs5ma50_r7earlycut-69338138",
]
Pk = P.set_index(["symbol", "date"])
for rid in RIDS:
    tr = pd.read_sql("SELECT symbol, entry_signal_date, entry_date, exit_date, pnl_pct FROM run_trades WHERE run_id=%s", pg, params=(rid,))
    tr["sd"] = pd.to_datetime(tr["entry_signal_date"])
    idx = list(zip(tr["symbol"], tr["sd"]))
    conv = [Pk["cs5_ma50"].get(k, np.nan) for k in idx]
    shift = [Pk["rank_shift_max"].get(k, np.nan) for k in idx]
    tr["conv"] = conv; tr["shift"] = shift
    ok = tr.dropna(subset=["conv"])
    near = ok[(ok["conv"] >= 0.40 - ok["shift"]) & (ok["conv"] <= 0.40 + ok["shift"])]
    y0022 = ok[pd.to_datetime(ok["entry_date"]).dt.year <= 2022]
    # holding-days overlapping breadth flip dates
    flipset = set(breadth.index[flip])
    def overlaps(r):
        try:
            days = pd.date_range(r["entry_date"], r["exit_date"] or r["entry_date"])
        except Exception:
            return False
        return any(d in flipset for d in days)
    n_flip_hold = int(ok.apply(overlaps, axis=1).sum()) if flip.sum() else 0
    print(f"[{rid.split('/')[-1]}] trades={len(tr)} conv-matched={len(ok)} "
          f"share2020-22={len(y0022)/max(len(ok),1):.2f} "
          f"entries within rank-shift of skip-0.40: {len(near)} ({len(near)/max(len(ok),1):.1%}) "
          f"trades holding across breadth-flip days: {n_flip_hold}")
print("[panel] per-date symbol count 2021 median:", int(P[P['date'].dt.year == 2021]['n_date'].median()),
      " 2022:", int(P[P['date'].dt.year == 2022]['n_date'].median()),
      " rank_shift_max 2021 median:", round(float(P[P['date'].dt.year == 2021]['rank_shift_max'].median()), 4))
