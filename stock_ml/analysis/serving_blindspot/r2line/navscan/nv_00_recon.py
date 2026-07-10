# -*- coding: utf-8 -*-
"""nv_00_recon: kham pha kho leaderboard_runs (moi row ke ca superseded)
de chuan bi proxy screening NAV-potential.

In: dem theo strategy, prefix ten, phan phoi trades/hold/pf.
"""
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

con = psycopg2.connect(**PG)
df = pd.read_sql(
    """
    select lr.run_name, lr.run_seed, lr.template_id, lr.composite_score, lr.total_pnl,
           lr.pf, lr.mdd_per_symbol, lr.trades, lr.wr, lr.avg_hold, lr.superseded,
           lr.generated_at, lr.run_id,
           st.name as tpl_name, st.strategy, st.model_mode, st.created_at as tpl_created
    from leaderboard_runs lr
    left join strategy_templates st on st.id = lr.template_id
    where lr.market = 'vn_stock' and lr.composite_score is not null
    """, con)
con.close()
print(f"rows total: {len(df)}  (superseded={df.superseded.sum()})")
print(f"templates distinct: {df.template_id.nunique()}")

df["prefix"] = df.tpl_name.fillna(df.run_name).str.extract(r"^([a-zA-Z]+[0-9]*)")[0]
print("\n== dem theo strategy ==")
print(df.groupby("strategy").agg(n=("run_name", "size"), tpl=("template_id", "nunique"),
                                 best=("composite_score", "max")).sort_values("n", ascending=False).to_string())

print("\n== dem theo prefix ten (top 40) ==")
g = df.groupby("prefix").agg(n=("run_name", "size"), tpl=("template_id", "nunique"),
                             best=("composite_score", "max"),
                             tr_max=("trades", "max")).sort_values("tpl", ascending=False)
print(g.head(40).to_string())

print("\n== phan phoi trades / hold / pf (per best-run-per-template) ==")
best = df.sort_values("composite_score", ascending=False).drop_duplicates("template_id")
print(best[["trades", "avg_hold", "pf", "mdd_per_symbol"]].describe().round(2).to_string())

print("\n== templates trades>=1200 & pf>=2.0 (dem theo prefix) ==")
hi = best[(best.trades >= 1200) & (best.pf >= 2.0)]
print(f"n = {len(hi)}")
print(hi.groupby("prefix").agg(n=("tpl_name", "size"), best=("composite_score", "max"),
                               tr=("trades", "max"), hold=("avg_hold", "mean")).sort_values("n", ascending=False).to_string())
print("NV00_DONE")
