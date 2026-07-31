# -*- coding: utf-8 -*-
"""hb_137: RE-REGISTER honest FAIR conviction-sizing (exposure-matched to base). hb_135/136: raw 105%
was ~75% EXPOSURE artifact. Fair same-conditions (exposure ~0.576 = base): preempt + conviction a0.6
deploy0.55 = 76.6%/DD-10.8 vs base 64.7%/-13.7 (SAME exposure) = +allocation-skill (higher return
LOWER DD). Update x2_struct_to_k25size to honest number+desc; supersede k16size (exposure-inflated)."""
from __future__ import annotations
import os, sys, warnings, hashlib
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
import hb_112_meta_target as M
import hb_136_exposure_matched as E

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent
RUNID = "template/x2_struct_to_k25size-69338138"
DESC = ("[K25 + preempt + CONVICTION-SIZING, EXPOSURE-MATCHED to base (fair same-conditions) — TONG VON"
        "<=1] Cung signal double-RS+struct-trail. Meta-prio conviction-size (a0.6) SCALED (deploy0.55) de "
        "avg-exposure=0.57 KHOP base equal-weight (0.576) -> co lap ALLOCATION-SKILL, khong tinh beta/"
        "exposure. Fair: CAGR 76.6%/DD-10.8/expo0.57 vs base 64.7%/-13.7/expo0.58 = +allocation-skill "
        "same-conditions (+return VA -DD, Calmar 7.1 vs 5.2 — meta chon return-cao-risk-thap). Raw un-"
        "matched la 105% nhung ~75% do EXPOSURE (deploy nhieu hon, khong phai skill) -> da loai de fair. "
        "3-seed robust. Preemption + sizing deu exposure-neutral sau khi match.")


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k137.csv"; cvtr.to_csv(cv, index=False)
    feat = M.features(cvtr.symbol.unique().tolist()); pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    navf, cagr, dd, ex = E.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm, K=25, mode="conviction", alpha=0.6, deploy=0.55)
    nav22, _, _, _ = E.prun(NavSim2(str(cv), date_lo="2022-01-01"), pm, K=25, mode="conviction", alpha=0.6, deploy=0.55)
    print(f"FAIR exposure-matched seed42: NAV=x{navf:.2f} CAGR={cagr*100:.1f}% DD={dd*100:.1f}% expo={ex:.3f} f22=x{nav22:.2f}", flush=True)
    # rename the k25size row to reflect fair; update leaderboard_nav with honest numbers
    cur.execute("update strategy_templates set description=%s where name='x2_struct_to_k25size'", (DESC,))
    ch = hashlib.md5(f"{RUNID}_fair_expomatched".encode()).hexdigest()[:16]
    cur.execute("""update leaderboard_nav set nav_adv=%s, nav_noadv=%s, cagr_adv=%s, cagr_noadv=%s, maxdd_nav=%s,
                     nav_f22_adv=%s, config_hash=%s, computed_at=now() where run_id=%s""",
                (navf, navf, cagr, cagr, dd, nav22, ch, RUNID))
    # supersede the exposure-inflated aggressive k16size
    cur.execute("update leaderboard_runs set superseded=true where run_name='x2_struct_to_k16size'")
    con.commit()
    print("UPDATED x2_struct_to_k25size to FAIR exposure-matched numbers; superseded k16size (inflated)", flush=True)
    # show final board
    cur.execute("""select lr.run_name, ln.cagr_adv, ln.maxdd_nav from leaderboard_runs lr
                   join leaderboard_nav ln on lr.run_id=ln.run_id
                   where lr.run_name like 'x2_struct_to%' and lr.superseded=false order by ln.cagr_adv desc""")
    print("FINAL fair board:", flush=True)
    for r in cur.fetchall(): print(f"  {r[0]:24s} {r[1]*100:5.1f}%  DD {r[2]*100:6.1f}%", flush=True)
    con.close(); print("HB_137_DONE", flush=True)


if __name__ == "__main__":
    main()
