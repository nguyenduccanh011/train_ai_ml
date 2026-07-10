# -*- coding: utf-8 -*-
"""nv_01_proxy: proxy screening NAV-potential toan kho leaderboard_runs.

Gem profile (theo fc_rule2): trades cao, hold ngan, pf >= 2, mdd_sym vua phai,
pnl/hold-day cao (von quay vong nhanh).

Proxy:
  P1 velocity  = total_pnl / avg_hold          (pnl tren 1 ngay giu von, fc_rule2 6.8 vs gb 4.3)
  P2 pnlsqrtT  = total_pnl * sqrt(trades)      (pnl co trong so vong quay)
  P3 hard gate = trades>=1200 & pf>=2.0 & mdd_sym<=0.40 & hold<=30

Xuat: nv_01_candidates.csv (union top-N moi proxy, kem name/strategy/engine keys de dedupe tay).
"""
import json

import numpy as np
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\r2line\navscan"

con = psycopg2.connect(**PG)
df = pd.read_sql(
    """
    select lr.run_name, lr.run_seed, lr.template_id, lr.composite_score, lr.total_pnl,
           lr.pf, lr.mdd_per_symbol, lr.trades, lr.wr, lr.avg_hold, lr.superseded,
           lr.generated_at, lr.run_id,
           st.name as tpl_name, st.strategy, st.model_mode, st.engine_config,
           st.description
    from leaderboard_runs lr
    left join strategy_templates st on st.id = lr.template_id
    where lr.market = 'vn_stock' and lr.composite_score is not null
      and lr.trades is not null and lr.trades > 0
    """, con)
con.close()

# dai dien per-template: uu tien seed 42, roi composite cao nhat
df["is42"] = (df.run_seed == 42).astype(int)
df = df.sort_values(["is42", "composite_score"], ascending=False)
best = df.drop_duplicates("template_id").copy()
best["name"] = best.tpl_name.fillna(best.run_name)

# ------- LOAI cac gia toc da co an NAV / canonical -------
BAN_PAT = (
    r"^(pm2?_|xr_|r17_|ruleexp_|tune_|fc_|r2_|r2b_|r2c_|n3_|np_atmkt|macd_ma20)"
)
best["banned"] = best.name.str.match(BAN_PAT, na=False)
# nopb / t903 / t1058 ho: nhan dien qua engine khong pullback + hold dai se tu rot o gate hold<=30
# gb/champion clones giu lai 1 dai dien duy nhat de doi chieu (khong can - da co anchor)

b = best[~best.banned].copy()
b = b[b.avg_hold > 0]

b["velocity"] = b.total_pnl / b.avg_hold
b["pnlsqrtT"] = b.total_pnl * np.sqrt(b.trades)
b["gate"] = (b.trades >= 1200) & (b.pf >= 2.0) & (b.mdd_per_symbol <= 0.40) & (b.avg_hold <= 30)

top_vel = b[b.gate].nlargest(60, "velocity")
top_pnl = b[b.gate].nlargest(60, "pnlsqrtT")
# them nhanh trades-cuc-cao pf tot (khong can gate mdd chat)
top_thr = b[(b.trades >= 1800) & (b.pf >= 2.3) & (b.avg_hold <= 22)].nlargest(40, "velocity")

cand = pd.concat([top_vel, top_pnl, top_thr]).drop_duplicates("template_id")


def eng_keys(ec):
    if isinstance(ec, str):
        try:
            ec = json.loads(ec)
        except Exception:
            return ""
    ec = ec or {}
    keys = ["entry_pullback_pct", "entry_pullback_window", "entry_gate",
            "downtrend_hard_stop_pct", "exit_overextension_pct", "trailing_stop_pct"]
    return ";".join(f"{k}={ec.get(k)}" for k in keys if ec.get(k) is not None)


cand["eng"] = cand.engine_config.apply(eng_keys)
cols = ["template_id", "name", "strategy", "run_seed", "superseded", "composite_score",
        "total_pnl", "pf", "mdd_per_symbol", "trades", "wr", "avg_hold",
        "velocity", "pnlsqrtT", "eng", "description"]
cand = cand.sort_values("velocity", ascending=False)
cand[cols].to_csv(f"{OUT}\\nv_01_candidates.csv", index=False)
print(f"union candidates: {len(cand)}")
pd.set_option("display.width", 250)
print(cand[["template_id", "name", "strategy", "composite_score", "total_pnl", "pf",
            "mdd_per_symbol", "trades", "avg_hold", "velocity"]].head(70).round(2).to_string(index=False))

# doi chieu anchor
print("\n== anchor (de so proxy) ==")
for nm in ["gb_x08", "n3_dtstop06", "fc_rule2", "r2_c2_pb40snr", "r2c_oxt04_p42"]:
    row = best[best.name == nm]
    if len(row):
        r = row.iloc[0]
        print(f"{nm}: comp={r.composite_score:.1f} pnl={r.total_pnl:.1f} pf={r.pf:.2f} "
              f"mdd={r.mdd_per_symbol:.3f} tr={r.trades} hold={r.avg_hold:.1f} "
              f"vel={r.total_pnl/r.avg_hold:.2f}")
print("NV01_DONE")
