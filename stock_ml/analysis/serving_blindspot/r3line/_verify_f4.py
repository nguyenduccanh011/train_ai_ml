# -*- coding: utf-8 -*-
"""Verify finding #4: meta-priority features as-of entry-day close vs intraday fill."""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as np
from scipy.stats import spearmanr
import hb_112_meta_target as M
from nh_nav2 import NavSim2, FEE

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
RID = "template/x2_struct_to-69338138"

con = psycopg2.connect(**PG)
tr_raw = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,entry_signal_date from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
print(f"trades in base run: {len(tr_raw)}")
tr_raw["entry_date"] = pd.to_datetime(tr_raw["entry_date"])
tr_raw["entry_signal_date"] = pd.to_datetime(tr_raw["entry_signal_date"])
same_day = (tr_raw["entry_date"] == tr_raw["entry_signal_date"]).mean()
print(f"frac entry_date == signal_date: {same_day:.3f}")

# --- 1. intraday-fill check ---
d = duckdb.connect(DUCK, read_only=True)
syms = tr_raw.symbol.unique().tolist()
ph = ",".join("?" * len(syms))
px = d.execute(f"select symbol,date,open,high,low,close from ohlcv where timeframe='1D' and symbol in ({ph})", syms).fetchdf()
d.close()
px["date"] = pd.to_datetime(px["date"])
j = tr_raw.merge(px.rename(columns={"date": "entry_date"}), on=["symbol", "entry_date"], how="left")
ok = j.dropna(subset=["open", "close"])
tol = 1e-4
at_close = (abs(ok.entry_price / ok.close - 1) < tol).mean()
at_open = (abs(ok.entry_price / ok.open - 1) < tol).mean()
in_range = ((ok.entry_price >= ok.low * (1 - tol)) & (ok.entry_price <= ok.high * (1 + tol))).mean()
below_close = (ok.entry_price < ok.close * (1 - tol)).mean()
print(f"fill==close: {at_close:.3f}  fill==open: {at_open:.3f}  fill within [low,high]: {in_range:.3f}  fill<close: {below_close:.3f}")
ok = ok.copy(); ok["bounce"] = ok.close / ok.entry_price - 1.0
print(f"bounce (close/fill-1): mean {ok.bounce.mean():.4f}  median {ok.bounce.median():.4f}  p90 {ok.bounce.quantile(0.9):.4f}")

# --- 2. build shipped vs causal feature sets ---
feat = M.features(syms)
tr = M.build_tr(con, RID, feat)
con.close()
# bounce merge into tr
bmap = {(r.symbol, r.entry_date): r.bounce for r in ok.itertuples()}
tr["bounce"] = [bmap.get((r.symbol, r.entry_date), np.nan) for r in tr.itertuples()]
m = tr.dropna(subset=["bounce", "t_pnl"])
print(f"\nSpearman(bounce, t_pnl) = {spearmanr(m.bounce, m.t_pnl)[0]:.3f}  (n={len(m)})")

# causal: lag features one trading day per symbol (feature row dated D becomes usable on next trading day)
feat_lag = feat.copy().sort_values(["symbol", "date"])
feat_lag["date"] = feat_lag.groupby("symbol")["date"].shift(-1)
feat_lag = feat_lag.dropna(subset=["date"])
con = psycopg2.connect(**PG)
tr_c = M.build_tr(con, RID, feat_lag)
con.close()

pm_s = M.meta_preds(tr, tgt="t_pnl")
pm_c = M.meta_preds(tr_c, tgt="t_pnl")
tr["p_ship"] = [pm_s.get((r.symbol, r.edkey), np.nan) for r in tr.itertuples()]
tr["p_caus"] = [pm_c.get((r.symbol, r.edkey), np.nan) for r in tr.itertuples()]
sc = tr.dropna(subset=["p_ship", "p_caus", "t_pnl"])
print(f"scored trades: {len(sc)}")
ic_s = spearmanr(sc.p_ship, sc.t_pnl)[0]; ic_c = spearmanr(sc.p_caus, sc.t_pnl)[0]
print(f"overall IC vs t_pnl: shipped {ic_s:.4f}  causal {ic_c:.4f}")
sb = sc.dropna(subset=["bounce"])
print(f"corr(pred, bounce): shipped {spearmanr(sb.p_ship, sb.bounce)[0]:.3f}  causal {spearmanr(sb.p_caus, sb.bounce)[0]:.3f}")
for y in range(2021, 2027):
    g = sc[sc.yr == y]
    if len(g) < 30: continue
    print(f"  {y}: n={len(g):4d} IC shipped {spearmanr(g.p_ship, g.t_pnl)[0]:+.3f}  causal {spearmanr(g.p_caus, g.t_pnl)[0]:+.3f}")
# top-pick divergence on multi-candidate days
div = tot = 0
for dkey, g in sc.groupby("edkey"):
    if len(g) < 2: continue
    tot += 1
    if g.p_ship.idxmax() != g.p_caus.idxmax(): div += 1
print(f"top-priority pick differs on {div}/{tot} multi-candidate entry days")

# --- 3. NAV replay K16 (hb_112 style) shipped vs causal ---
cv = HERE / "_vf4_trades.csv"
tr_raw[["symbol", "entry_date", "exit_date", "entry_price", "exit_price"]].to_csv(cv, index=False)
n_s, cg_s2, dd_s2 = M.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm_s)
n_c, cg_c2, dd_c2 = M.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm_c)
print(f"\nK16 replay: shipped NAV x{n_s:.2f} CAGR {cg_s2*100:.1f}% DD {dd_s2*100:.1f}% | causal NAV x{n_c:.2f} CAGR {cg_c2*100:.1f}% DD {dd_c2*100:.1f}%  delta {100*(cg_s2-cg_c2):+.2f}pp")
print("VF4_PART1_DONE")
