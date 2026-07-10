# -*- coding: utf-8 -*-
"""FAMILY_CHAMPIONS step 1: taxonomy mo rong tren toan bo strategy_templates.

Hai ho moi (bo sung pm01):
  RULE-ONLY : khong slot ML head nao tham gia tin hieu.
              = model_mode/strategy 'rule_only' hoac engine_config.rule_only_no_ml,
              VA khong slot nao co ml_component_id (strict). Neu co ML slot
              (vd hybrid_rule_entry_ml_downside) -> ghi chu 'rule+ml_head', khong strict.
  NO-PULLBACK: fill KHONG phai pullback-limit -> entry_pullback_pct falsy/absent
              (mua at-market theo entry_bar_fill_type: open_next/close_next/close_same).

Output: fc01_classified.csv + bang dem + top-3 moi ho.
"""
import json

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"

con = psycopg2.connect(**PG)
tpl = pd.read_sql(
    "select id, name, strategy, model_mode, signal_threshold, entry_threshold, exit_threshold, "
    "engine_config, created_at, description from strategy_templates", con)
slots = pd.read_sql(
    "select template_id, slot_type, ml_component_id, rule_component_id from component_slots", con)
best = pd.read_sql(
    """
    select distinct on (template_id)
        template_id, run_name, run_seed, composite_score, total_pnl, pf,
        mdd_per_symbol, trades, wr, avg_hold, generated_at, run_id
    from leaderboard_runs
    where template_id is not null and composite_score is not null
      and superseded = false and market = 'vn_stock'
    order by template_id, composite_score desc
    """, con)
con.close()
best = best.set_index("template_id")

ml_slots = slots[slots.ml_component_id.notna()].groupby("template_id")["slot_type"].apply(list)
rule_slots = slots[slots.rule_component_id.notna()].groupby("template_id")["slot_type"].apply(list)


def truthy(v):
    return v is not None and v is not False and v != 0 and v != "" and v != {}


rows = []
for _, r in tpl.iterrows():
    ec = r["engine_config"]
    if isinstance(ec, str):
        ec = json.loads(ec) if ec else {}
    ec = ec or {}
    tid = r["id"]

    mls = ml_slots.get(tid, [])
    rls = rule_slots.get(tid, [])
    no_ml_flag = truthy(ec.get("rule_only_no_ml"))
    mode_rule = (r["model_mode"] == "rule_only") or (r["strategy"] == "rule_only")

    if (mode_rule or no_ml_flag) and not mls:
        fam_rule = "rule_only_strict"
    elif (mode_rule or no_ml_flag) and mls:
        fam_rule = "rule+ml_head"          # vd hybrid_rule_entry_ml_downside
    elif r["strategy"].startswith("hybrid_rule_entry"):
        fam_rule = "rule_entry_ml_exit"
    else:
        fam_rule = "ml_signal"

    pb = ec.get("entry_pullback_pct")
    nopb = not truthy(pb)
    fill = ec.get("entry_bar_fill_type", "open_next")
    b_chan = truthy(ec.get("entry_bchannel_z")) or truthy(ec.get("entry_b_channel_enabled"))

    b = best.loc[tid] if tid in best.index else None
    rows.append(dict(
        template_id=tid, name=r["name"], strategy=r["strategy"], model_mode=r["model_mode"],
        fam_rule=fam_rule, nopb=nopb, pullback_pct=pb, fill_type=fill, b_channel=b_chan,
        ml_slots=",".join(map(str, mls)), rule_slots=",".join(map(str, rls)),
        composite=(float(b["composite_score"]) if b is not None else None),
        run_seed=(int(b["run_seed"]) if b is not None and pd.notna(b["run_seed"]) else None),
        total_pnl=(float(b["total_pnl"]) if b is not None else None),
        pf=(float(b["pf"]) if b is not None and pd.notna(b["pf"]) else None),
        mdd=(float(b["mdd_per_symbol"]) if b is not None and pd.notna(b["mdd_per_symbol"]) else None),
        trades=(int(b["trades"]) if b is not None and pd.notna(b["trades"]) else None),
        wr=(float(b["wr"]) if b is not None and pd.notna(b["wr"]) else None),
        hold=(float(b["avg_hold"]) if b is not None and pd.notna(b["avg_hold"]) else None),
        run_year=(str(b["generated_at"])[:7] if b is not None else None),
        run_id=(b["run_id"] if b is not None else None),
        created=str(r["created_at"])[:10],
    ))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}\\fc01_classified.csv", index=False)

print("== fam_rule x nopb: n / with-run / best composite ==")
g = df.groupby(["fam_rule", "nopb"]).agg(
    n=("template_id", "size"),
    with_run=("composite", lambda s: s.notna().sum()),
    best=("composite", "max")).round(1)
print(g.to_string())

cols = ["template_id", "name", "strategy", "composite", "run_seed", "run_year",
        "total_pnl", "pf", "mdd", "trades", "wr", "hold", "fill_type", "pullback_pct"]

print("\n== HO RULE-ONLY (strict): top 15 ==")
sub = df[(df.fam_rule == "rule_only_strict") & df.composite.notna()].sort_values("composite", ascending=False)
print(sub[cols].head(15).to_string(index=False))

print("\n== rule+ml_head / rule_entry_ml_exit (doi chieu, KHONG thuoc rule-only) ==")
sub2 = df[df.fam_rule.isin(["rule+ml_head", "rule_entry_ml_exit"]) & df.composite.notna()].sort_values(
    "composite", ascending=False)
print(sub2[cols].head(8).to_string(index=False))

print("\n== HO NO-PULLBACK (ML signal, khong pullback fill): top 20 ==")
sub3 = df[(df.fam_rule == "ml_signal") & df.nopb & df.composite.notna()].sort_values(
    "composite", ascending=False)
print(sub3[cols].head(20).to_string(index=False))

print("\n== doi chieu: top-5 toan bang (moi class) ==")
print(df[df.composite.notna()].sort_values("composite", ascending=False)[cols[:8] + ["fam_rule", "nopb"]]
      .head(5).to_string(index=False))
print("FC01_DONE")
