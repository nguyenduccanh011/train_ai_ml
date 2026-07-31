# -*- coding: utf-8 -*-
"""AUDIT finding #4: meta-priority features as-of entry-day CLOSE vs intraday fill.
Shipped arm = exact hb_112 pipeline (features merged on entry_date).
Causal arm = same pipeline but price features lagged 1 session (known before fill).
Measure: bounce corr, walk-forward IC vs t_pnl, top-pick flip rate, NAV replay K16.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import psycopg2, duckdb, pandas as pd, numpy as np
from scipy.stats import spearmanr
import hb_112_meta_target as M
from nh_nav2 import NavSim2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
RID = "template/x2_struct_to-69338138"   # base template 3185 run (seed 123) in DB
FCOLS = M.FCOLS
PRICE_FEATS = [c for c in FCOLS if c not in ("score", "exit_score")]

con = psycopg2.connect(**PG)
tr0 = pd.read_sql("select symbol from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
syms = tr0.symbol.unique().tolist()
feat = M.features(syms)

# causal variant: lag ALL price features by one session per symbol
feat_lag = feat.sort_values(["symbol", "date"]).copy()
feat_lag[PRICE_FEATS] = feat_lag.groupby("symbol")[PRICE_FEATS].shift(1)

tr_ship = M.build_tr(con, RID, feat)
M._FEAT = None  # not used again, safe
tr_caus = M.build_tr(con, RID, feat_lag)
con.close()

# bounce = entry-day close / fill price - 1 (post-fill info inside entry-day features)
d = duckdb.connect(M.DUCK, read_only=True)
px = d.execute("select symbol, date, close from ohlcv where timeframe='1D'").fetchdf(); d.close()
px["date"] = pd.to_datetime(px["date"])
for t in (tr_ship, tr_caus):
    m = t.merge(px.rename(columns={"date": "entry_date", "close": "eod_close"}), on=["symbol", "entry_date"], how="left")
    t["bounce"] = (m["eod_close"] / m["entry_price"] - 1).values

print("== scored trades:", len(tr_ship), "years", sorted(tr_ship.yr.unique()))
sb = tr_ship.dropna(subset=["bounce", "t_pnl"])
print("Spearman(bounce, t_pnl) = %.3f (n=%d)" % (spearmanr(sb.bounce, sb.t_pnl)[0], len(sb)))
# sanity: shipped entry-day features literally contain the bounce? dist_h20/ma20r at close of fill day
print("Spearman(bounce, shipped ret5@entry_close) = %.3f" % spearmanr(sb.bounce, sb.ret5, nan_policy="omit")[0])

pm_ship = M.meta_preds(tr_ship, tgt="t_pnl")
pm_caus = M.meta_preds(tr_caus, tgt="t_pnl")

for t, pm, tag in ((tr_ship, pm_ship, "ship"), (tr_caus, pm_caus, "caus")):
    t["pred"] = [pm.get((r.symbol, r.edkey), np.nan) for r in t.itertuples()]

both = tr_ship[["symbol", "edkey", "t_pnl", "bounce", "yr", "pred"]].rename(columns={"pred": "p_ship"}).merge(
    tr_caus[["symbol", "edkey", "pred"]].rename(columns={"pred": "p_caus"}), on=["symbol", "edkey"])
b = both.dropna(subset=["p_ship", "p_caus", "t_pnl"])
print("\n== walk-forward predictions on %d trades (2021-2026)" % len(b))
print("corr(pred, bounce):  shipped %.3f | causal %.3f" %
      (spearmanr(b.p_ship, b.bounce, nan_policy="omit")[0], spearmanr(b.p_caus, b.bounce, nan_policy="omit")[0]))
ic_s = spearmanr(b.p_ship, b.t_pnl)[0]; ic_c = spearmanr(b.p_caus, b.t_pnl)[0]
print("IC vs t_pnl overall: shipped %.4f | causal %.4f | post-fill share %.1f%%" % (ic_s, ic_c, 100 * (1 - ic_c / ic_s)))
print("per-year IC (shipped | causal | n):")
for y, g in b.groupby("yr"):
    print("  %d: %+.4f | %+.4f | %d" % (y, spearmanr(g.p_ship, g.t_pnl)[0], spearmanr(g.p_caus, g.t_pnl)[0], len(g)))

# top-priority flip on multi-candidate entry days (the decision the deploy harness makes)
flips = tot = 0
for ed, g in b.groupby("edkey"):
    if len(g) < 2:
        continue
    tot += 1
    if g.p_ship.idxmax() != g.p_caus.idxmax():
        flips += 1
print("\ntop-priority pick differs on %d/%d = %.1f%% of multi-candidate entry days" % (flips, tot, 100 * flips / tot))

# NAV replay K=16 (M.prun = priority-fill sim used by hb_112 registration path)
cv = HERE / "_audit_f4_trades.csv"
con = psycopg2.connect(**PG)
cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(RID,))
con.close()
cvtr.to_csv(cv, index=False)
n_s, cg_s, dd_s = M.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm_ship)
n_c, cg_c, dd_c = M.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm_caus)
print("\nNAV replay K=%d: shipped x%.2f CAGR %.1f%% DD %.1f%% | causal x%.2f CAGR %.1f%% DD %.1f%% | delta %.2fpp CAGR"
      % (M.K, n_s, cg_s * 100, dd_s * 100, n_c, cg_c * 100, dd_c * 100, (cg_s - cg_c) * 100))
print("AUDIT_F4_DONE")
