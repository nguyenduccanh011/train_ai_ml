# -*- coding: utf-8 -*-
"""nv_02_families: soi cac nhanh/era chua tung NAV-hoa + mo ta cluster la.

(1) SMAC single_ml_action_classifier: top theo velocity/pnl
(2) v19* names, zigzag_dual_ml, regression_dual_ml (cont era)
(3) mo ta description cua: n2_velov_univ150c, n2_2429_*, sw_ox08_pb03, qQ_rsdrop_*,
    n2_am20_oxt03, w0_pyr_u10_r02, n2_pb_xrule_bear3, n2_1187_age4
"""
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
con = psycopg2.connect(**PG)
df = pd.read_sql(
    """
    select lr.run_name, lr.run_seed, lr.template_id, lr.composite_score, lr.total_pnl,
           lr.pf, lr.mdd_per_symbol, lr.trades, lr.wr, lr.avg_hold, lr.superseded,
           st.name as tpl_name, st.strategy, st.description
    from leaderboard_runs lr
    left join strategy_templates st on st.id = lr.template_id
    where lr.market = 'vn_stock' and lr.composite_score is not null
      and lr.trades is not null and lr.trades > 0
    """, con)

pd.set_option("display.width", 260)
pd.set_option("display.max_colwidth", 60)
df["name"] = df.tpl_name.fillna(df.run_name)
df["is42"] = (df.run_seed == 42).astype(int)
best = df.sort_values(["is42", "composite_score"], ascending=False).drop_duplicates("template_id").copy()
best["vel"] = best.total_pnl / best.avg_hold.replace(0, pd.NA)
C = ["template_id", "name", "strategy", "composite_score", "total_pnl", "pf",
     "mdd_per_symbol", "trades", "avg_hold", "vel"]

print("== SMAC (single_ml_action_classifier) top-12 velocity (pf>=1.5) ==")
s = best[best.strategy.str.startswith("single_ml", na=False) & (best.pf >= 1.5)]
print(s.nlargest(12, "vel")[C].round(2).to_string(index=False))

print("\n== v19 names ==")
v = best[best.name.str.contains("v19", case=False, na=False)]
print(v[C].round(2).to_string(index=False))

print("\n== zigzag_dual_ml top-8 velocity (pf>=1.5) ==")
z = best[best.strategy.str.startswith("zigzag", na=False) & (best.pf >= 1.5)]
print(z.nlargest(8, "vel")[C].round(2).to_string(index=False))

print("\n== regression_dual_ml (cont era) top-8 velocity (pf>=1.5, trades>=800) ==")
c = best[(best.strategy == "regression_dual_ml") & (best.pf >= 1.5) & (best.trades >= 800)]
print(c.nlargest(8, "vel")[C].round(2).to_string(index=False))

print("\n== mo ta cac cluster dang chu y ==")
names = ["n2_velov_univ150c", "n2_2429_dyncsr88", "n2_2429_maxhold20_cagr",
         "n2_2429_exit_vol_market_downleg", "sw_ox08_pb03", "qQ_rsdrop_d15",
         "n2_am20_oxt03", "n2_oxtrail_04", "w0_pyr_u10_r02", "n2_pb_xrule_bear3",
         "n2_1187_age4", "n2_mgz13_emgz09", "n2_prot_a05_m15",
         "n2_dz_act05_m10_fl03_cap08", "n2_1137_poplock_a05_x03_t04",
         "n2_1204_ox50_12", "n2_velov_base"]
tpl = pd.read_sql("select id, name, description, engine_config from strategy_templates where name = any(%s)",
                  con, params=(names,))
con.close()
for _, r in tpl.iterrows():
    print(f"\n--- {r['name']} (t{r['id']}) ---")
    print((r["description"] or "")[:400])
print("NV02_DONE")
