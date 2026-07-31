"""INTERNAL verification (user challenge): did the 'null' experiments actually CHANGE the internal
selection/ranking, or did z-scoring + top-K rank + the loose buy-threshold (z>-1.9) ABSORB the change
so 'null' is trivial (nothing changed downstream)? Compare frontier vs a TARGET-swap (cst_h10) and a
FEATURE-add (cap_asym) at seed42:
  (a) trade overlap: Jaccard of {symbol, entry_signal_date}. ~1.0 => change absorbed (trivial null).
  (b) raw-score rank correlation on common (symbol,date). ~1.0 => ranking unchanged => absorbed.
Low overlap / low rank-corr => the change DID alter selection and NAV genuinely didn't improve (real null).
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd, psycopg2
from stock_ml.scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
TPLS = {"frontier": 3185, "cst_h10(target)": 3232, "cap_asym(feature)": 3261}

rids = {}
for nm, tid in TPLS.items():
    r = run_template_experiment(template_id=tid, seed=42)
    rids[nm] = r.get("run_id")
    print(f"ran {nm} t{tid} -> {rids[nm]}", flush=True)

con = psycopg2.connect(**PG)


def trades(rid):
    return pd.read_sql(
        "SELECT symbol, entry_signal_date, pnl_pct FROM run_trades WHERE run_id=%s",
        con,
        params=(rid,),
    )


def signals(rid):
    return pd.read_sql(
        "SELECT symbol, date, score FROM run_signals WHERE run_id=%s", con, params=(rid,)
    )


ft = trades(rids["frontier"])
fs = signals(rids["frontier"])
fset = set(zip(ft["symbol"], ft["entry_signal_date"]))
print(f"\nfrontier: {len(ft)} trades, {len(fs)} signal rows")
print("=== INTERNAL CHANGE vs frontier (seed42) ===")
for nm in TPLS:
    if nm == "frontier":
        continue
    vt = trades(rids[nm])
    vs = signals(rids[nm])
    vset = set(zip(vt["symbol"], vt["entry_signal_date"]))
    inter = len(fset & vset)
    union = len(fset | vset)
    jac = inter / union if union else 0.0
    only_f = len(fset - vset)
    only_v = len(vset - fset)
    # raw-score rank correlation on common (symbol,date)
    m = fs.merge(vs, on=["symbol", "date"], suffixes=("_f", "_v"))
    rc = m["score_f"].corr(m["score_v"], method="spearman") if len(m) > 100 else float("nan")
    print(f"\n{nm}: {len(vt)} trades")
    print(
        f"  trade Jaccard overlap = {jac:.3f}  (shared {inter}, only-frontier {only_f}, only-variant {only_v})"
    )
    print(f"  raw-score Spearman rank-corr = {rc:.4f}  (n_common={len(m)})")
    verdict = (
        "ABSORBED (trivial null — selection ~unchanged)"
        if jac > 0.95 and rc > 0.98
        else "REAL CHANGE (selection genuinely differs; NAV null = truly not better)"
    )
    print(f"  => {verdict}")
con.close()
print("VERIFY_DONE")
