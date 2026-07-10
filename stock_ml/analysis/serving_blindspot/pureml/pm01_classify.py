# -*- coding: utf-8 -*-
"""pureml step 1: phan loai muc-do-rule cua toan bo strategy_templates.

Classes:
  a PURE-ML      : exit chi boi ML signal (+max_hold horizon); khong force-sell rule,
                   khong suppress/protect/hold modifier, khong trailing/hard-stop,
                   khong entry rule-gate (entry_gate/market/breadth/nonbull/bchannel/rs).
  b ML-DOMINANT  : khong force-sell rule / khong regime-suppress exit; cho phep
                   trailing/hard-stop don gian + modifier nhe (protect/hold/snr) +
                   entry gate (ghi chu flag).
  c RULE-HYBRID  : co force-sell rule (exit_force_gate*, overext-sell, shield,
                   structural/downtrend/csr stop, top_reversal...) hoac
                   exit_market_gate (suppress) hoac exit_gate cons (suppress ML sell)
                   hoac signal_exit_enabled=False (exit thuan rule).
"""
import json
from collections import Counter

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"

con = psycopg2.connect(**PG)

tpl = pd.read_sql(
    "select id, name, strategy, model_mode, signal_threshold, entry_threshold, exit_threshold, "
    "engine_config, created_at, description from strategy_templates", con)

# best non-superseded run per template (vn_stock, live score)
best = pd.read_sql(
    """
    select distinct on (template_id)
        template_id, run_name, run_seed, composite_score, total_pnl, pf,
        mdd_per_symbol, trades, wr, avg_hold, generated_at, run_id, superseded
    from leaderboard_runs
    where template_id is not null and composite_score is not null
      and superseded = false and market = 'vn_stock'
    order by template_id, composite_score desc
    """, con)
con.close()

best = best.set_index("template_id")


def val(ec, k, default=None):
    v = ec.get(k, default)
    return v


def truthy(v):
    return v is not None and v is not False and v != 0 and v != "" and v != {}


FORCE_KEYS = [
    "exit_force_gate", "exit_force_gate_nonbull", "exit_force_gate_lowbreadth",
    "exit_force_gate_vn30", "top_reversal_exit", "entry_rollover_exit", "top_exit",
    "exit_rs_drop", "downtrend_hard_stop_pct", "structural_stop_lookback",
]
SUPPRESS_KEYS = ["exit_force_suppress"]
TRAIL_KEYS = [
    "trailing_stop_pct", "trailing_atr_mult", "trailing_struct_donch_win",
    "trailing_tier2_activate_pct", "pop_lock_arm_pct", "dist_arm_neg_thresh",
    "cold_trailing_activate_pct", "overext_trail_pct", "take_profit_k",
    "breakeven_lock_mfe", "vol_spike_z_threshold", "ml_swing_trail_thr",
    "stale_exit_bars", "trailing_score_k", "trailing_combo_k", "mfe_act_k",
    "csr_hard_stop_pct", "tier2_csr_threshold",
]
MOD_KEYS = [  # modifier nhe tren ML exit (suppress mot exit, khong force)
    "signal_exit_protect_lo", "signal_exit_protect_ma", "signal_exit_hold_ext_atr",
    "signal_exit_skip_if_score3_z", "signal_exit_skip_if_entry_z",
    "exit_snr_extend_threshold", "exit_gate", "signal_exit_incubate_floor",
]
ENTRY_RULE_KEYS = [
    "entry_gate", "entry_breadth_gate", "entry_bchannel_z", "entry_rs_gate",
    "entry_head_csrank_gate", "entry_xs_mom_pct", "early_entry_reversal",
    "entry_z_low_threshold",
]

rows = []
for _, r in tpl.iterrows():
    ec = r["engine_config"]
    if isinstance(ec, str):
        ec = json.loads(ec) if ec else {}
    ec = ec or {}

    force = [k for k in FORCE_KEYS if truthy(val(ec, k))]
    if truthy(val(ec, "macd_shield_enabled")):
        force.append("macd_shield")
    if truthy(val(ec, "rsi_shield_enabled")):
        force.append("rsi_shield")
    # overext sell-the-top rule: chi khi overext trong exit_priority va window>0
    ep = ec.get("exit_priority") or ["trailing_stop", "overext", "signal"]  # engine default
    ox_on = ("overext" in ep) and (val(ec, "overext_ma_window", 0) or 0) > 0
    if ox_on:
        force.append("overext_sell")

    suppress = [k for k in SUPPRESS_KEYS if truthy(val(ec, k))]
    if truthy(val(ec, "exit_market_gate_enabled")):
        suppress.append("exit_market_gate")

    hard_stop = val(ec, "hard_stop_pct") is not None and "hard_stop" in ep
    if val(ec, "hard_stop_atr_mult") is not None and "hard_stop" in ep:
        hard_stop = True
    trail = [k for k in TRAIL_KEYS if truthy(val(ec, k))]
    # trailing_stop trong exit_priority nhung pct None => inert
    if "trailing_stop_pct" in trail and "trailing_stop" not in ep and not any(
            t in trail for t in TRAIL_KEYS[1:]):
        pass  # van tinh (atr/struct trail chay doc lap exit_priority)
    mods = [k for k in MOD_KEYS if truthy(val(ec, k))]
    entry_rules = [k for k in ENTRY_RULE_KEYS if truthy(val(ec, k))]
    if (val(ec, "entry_skip_nonbull_persist", 0) or 0) > 0:
        entry_rules.append("entry_skip_nonbull")
    if truthy(val(ec, "entry_market_gate_enabled")):
        entry_rules.append("entry_market_gate")
    if truthy(val(ec, "entry_market_chop_enabled")):
        entry_rules.append("entry_market_chop")

    sig_exit_on = val(ec, "signal_exit_enabled", True)
    ml_exit = bool(sig_exit_on) and ("signal" in ep)
    rule_only = r["strategy"] == "rule_only" or truthy(val(ec, "rule_only_no_ml"))

    if rule_only or not ml_exit:
        cls = "c"
        why = "no_ml_exit" if not ml_exit else "rule_only"
    elif force or suppress:
        cls = "c"
        why = ",".join(force + suppress)
    elif trail or hard_stop or mods or entry_rules:
        cls = "b"
        why = ",".join(trail + (["hard_stop"] if hard_stop else []) + mods + entry_rules)
    else:
        cls = "a"
        why = ""

    b = best.loc[r["id"]] if r["id"] in best.index else None
    rows.append(dict(
        template_id=r["id"], name=r["name"], strategy=r["strategy"], cls=cls,
        flags=why, force=",".join(force), suppress=",".join(suppress),
        trail=",".join(trail), hard_stop=hard_stop, mods=",".join(mods),
        entry_rules=",".join(entry_rules), ml_exit=ml_exit,
        exit_priority=json.dumps(ep),
        composite=(float(b["composite_score"]) if b is not None else None),
        run_name=(b["run_name"] if b is not None else None),
        run_seed=(int(b["run_seed"]) if b is not None and pd.notna(b["run_seed"]) else None),
        run_year=(b["generated_at"].year if b is not None else None),
        total_pnl=(float(b["total_pnl"]) if b is not None else None),
        pf=(float(b["pf"]) if b is not None and pd.notna(b["pf"]) else None),
        mdd=(float(b["mdd_per_symbol"]) if b is not None and pd.notna(b["mdd_per_symbol"]) else None),
        trades=(int(b["trades"]) if b is not None and pd.notna(b["trades"]) else None),
        run_id=(b["run_id"] if b is not None else None),
        created=str(r["created_at"])[:10],
    ))

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}\\pm01_classified.csv", index=False)

print("== class counts (all / with-run) ==")
print(df.groupby("cls").agg(n=("template_id", "size"),
                            with_run=("composite", lambda s: s.notna().sum()),
                            best=("composite", "max")).to_string())

for c in ("a", "b"):
    sub = df[(df.cls == c) & df.composite.notna()].sort_values("composite", ascending=False)
    print(f"\n== TOP 25 class {c} ==")
    cols = ["template_id", "name", "strategy", "composite", "run_seed", "run_year",
            "total_pnl", "pf", "mdd", "trades", "flags"]
    print(sub[cols].head(25).to_string(index=False))

# de doi chieu: top class c
sub = df[(df.cls == "c") & df.composite.notna()].sort_values("composite", ascending=False)
print("\n== TOP 5 class c (doi chieu) ==")
print(sub[["template_id", "name", "composite", "flags"]].head(5).to_string(index=False))
print("PM01_DONE")
