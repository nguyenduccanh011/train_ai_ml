# -*- coding: utf-8 -*-
"""EXIT-ERROR forensic (continuation of selection forensic): champion sells at 29% of peak (capture leak).
Which exits were PREMATURE (price kept running UP after exit) vs CORRECT (flat/down after)? Characterize
EXIT-TIME features (exit_reason, exit_score, momentum/dist-MA at exit, days_held, MFE-so-far, regime) to
find a separable 'should-have-held' signal -> a defer-exit lever. Read-only. AUC = separability."""
from __future__ import annotations
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2, duckdb, numpy as np, pandas as pd

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
MARKET = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
CHAMP = "template/x2_struct_to_k16preempt_cssize-69338138"

con = psycopg2.connect(**PG)
tr = pd.read_sql("SELECT symbol,entry_date,exit_date,entry_price,exit_price,pnl_pct,exit_reason FROM run_trades "
                 "WHERE run_id=%s AND exit_date IS NOT NULL AND exit_reason<>'preempt'", con, params=(CHAMP,))
sg = pd.read_sql("SELECT symbol,date,exit_score FROM run_signals WHERE run_id=%s", con, params=(CHAMP,))
con.close()
tr["entry_date"] = pd.to_datetime(tr["entry_date"]); tr["exit_date"] = pd.to_datetime(tr["exit_date"])
sg["date"] = pd.to_datetime(sg["date"])
tr = tr.merge(sg, left_on=["symbol", "exit_date"], right_on=["symbol", "date"], how="left")

cx = duckdb.connect(MARKET, read_only=True)
syms = ",".join(repr(s) for s in tr.symbol.unique())
px = cx.execute(f"SELECT symbol,date,high,close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({syms}) AND date>='2018-06-01' ORDER BY symbol,date").fetchdf()
vni = cx.execute("SELECT date,close FROM ohlcv WHERE timeframe='1D' AND symbol='VN30F1M' ORDER BY date").fetchdf(); cx.close()
px["date"] = pd.to_datetime(px["date"]); vni["date"] = pd.to_datetime(vni["date"])
vni["mkt50"] = vni["close"] / vni["close"].rolling(50).mean() - 1
vmap = dict(zip(vni.date, vni.mkt50))
CL, HI, BO, MA20, R5 = {}, {}, {}, {}, {}
for s, g in px.groupby("symbol"):
    g = g.sort_values("date").reset_index(drop=True)
    CL[s] = g["close"].to_numpy(); HI[s] = g["high"].to_numpy(); BO[s] = {d: i for i, d in enumerate(g["date"])}
    MA20[s] = g["close"].rolling(20).mean().to_numpy(); R5[s] = g["close"].pct_change(5).to_numpy()

cont20, momexit, distma_exit, mfe_sofar, days_held = [], [], [], [], []
for r in tr.itertuples():
    be = BO.get(r.symbol, {}).get(r.entry_date); bx = BO.get(r.symbol, {}).get(r.exit_date)
    if be is None or bx is None or r.exit_price <= 0:
        for L in (cont20, momexit, distma_exit, mfe_sofar, days_held):
            L.append(np.nan)
        continue
    ca = CL[r.symbol]
    cont20.append(ca[bx + 20] / r.exit_price - 1.0 if bx + 20 < len(ca) else np.nan)
    momexit.append(R5[r.symbol][bx] if bx < len(R5[r.symbol]) else np.nan)
    m = MA20[r.symbol][bx]; distma_exit.append(ca[bx] / m - 1.0 if m and not np.isnan(m) else np.nan)
    peak = HI[r.symbol][be:bx + 1].max(); mfe_sofar.append(peak / r.entry_price - 1.0 if r.entry_price else np.nan)
    days_held.append(bx - be)
tr["cont20"] = cont20; tr["momexit"] = momexit; tr["distma_exit"] = distma_exit
tr["mfe_sofar"] = mfe_sofar; tr["days_held"] = days_held; tr["mkt50"] = tr.exit_date.map(vmap)

print(f"exits: {len(tr)}  avg post-exit+20 = {100*tr.cont20.mean():+.2f}%  (>0 = left money on table)", flush=True)
print("\n=== post-exit continuation by exit_reason ===", flush=True)
for rs, g in tr.groupby("exit_reason"):
    if len(g) < 20:
        continue
    print(f"  {rs:14s}: n={len(g):4d}  avg_pnl={100*g.pnl_pct.mean():+5.1f}%  post+20={100*g.cont20.mean():+5.1f}%  "
          f"(premature%={100*(g.cont20>0.03).mean():.0f})", flush=True)

# cohort: PREMATURE (kept running >+5% after exit) vs CORRECT (<=0 after)
prem = tr[tr.cont20 > 0.05]; corr = tr[tr.cont20 <= 0.0]
print(f"\n=== PREMATURE exit (post+20>+5%, n={len(prem)}) vs CORRECT (post+20<=0, n={len(corr)}) — exit-time features ===", flush=True)
print("  feature      | premature | correct | |AUC-.5|", flush=True)


def auc(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    if len(a) < 20 or len(b) < 20:
        return np.nan
    allv = pd.concat([a, b]); rk = allv.rank(); n1 = len(a)
    return (rk.iloc[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(b))


rows = []
for f in ["exit_score", "momexit", "distma_exit", "mfe_sofar", "days_held", "mkt50", "pnl_pct"]:
    rows.append((f, prem[f].mean(), corr[f].mean(), abs((auc(prem[f], corr[f]) or 0.5) - 0.5)))
for f, pm_, cm_, s in sorted(rows, key=lambda x: -(x[3] if x[3] == x[3] else -1)):
    print(f"  {f:12s} | {pm_:+9.3f} | {cm_:+9.3f} | {s:.3f}", flush=True)
print("\n(|AUC-.5|>0.10 = separable 'hold-longer' signal at exit -> defer-exit lever exists)")
print("EXIT_FORENSIC_DONE", flush=True)
