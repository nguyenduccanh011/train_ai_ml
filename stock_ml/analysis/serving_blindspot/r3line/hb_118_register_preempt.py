# -*- coding: utf-8 -*-
"""hb_118: REGISTER x2_struct_to_k16preempt = struct_to t3185 @ K16 + meta-fill + SLOT-PREEMPTION
(R2 prio-swap margin 0.01). hb_116/117: 3/3 seed win, base meta 73%->89.3% CAGR, DD phang. Label RO
operating-point TAN CONG (!= board K25-shuffle). Supersede t3333 k16meta."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy, hashlib
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
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
import hb_112_meta_target as M
import hb_115_preempt_causal as P

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent
NAME = "x2_struct_to_k16preempt"
MARGIN = 0.01
DESC = ("[K16 + META-FILL + SLOT-PREEMPTION operating-point cua struct_to t3185 — KHONG cung thuoc do "
        "board K25-shuffle] Cung signal double-RS+struct-trail. Execution nang cap: K=16 concentration + "
        "fill uu tien META-model + SLOT-PREEMPTION (khi book day + tin hieu conviction-cao moi den, duoi "
        "held-leg co entry-meta-prio thap nhat neu new_prio-held_prio>0.01, realize @mark, nap lenh moi; "
        "van single-book equal-weight, causal). Multi-seed 42/21/123: CAGR 89.3%/DD-16.3%/NAV x64.0 (3/3 "
        "seed win, +15pp vs meta-fill K16 73%; DD phang; prio=score cung win). Vuot k16meta 74%. TAN CONG.")


async def make_tmpl():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(NAME)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=NAME, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=DESC, hypothesis="K16+meta+slot-preemption",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    tid = asyncio.run(make_tmpl()); asyncio.run(async_engine.dispose())
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("update strategy_templates set description=%s where id=%s", (DESC, tid)); con.commit()
    r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k118.csv"; cvtr.to_csv(cv, index=False)
    feat = M.features(cvtr.symbol.unique().tolist())
    tr = M.build_tr(con, rid, feat); pm = M.meta_preds(tr, tgt='t_pnl')
    navf, cagr, dd, ev = P.prun_causal(NavSim2(str(cv), date_lo="2020-01-01"), pm, rule="R2", margin=MARGIN)
    nav22, _, _, _ = P.prun_causal(NavSim2(str(cv), date_lo="2022-01-01"), pm, rule="R2", margin=MARGIN)
    print(f"K16+meta+preempt seed42: NAV=x{navf:.2f} CAGR={cagr*100:.1f}% DD={dd*100:.1f}% f22=x{nav22:.2f} evict={ev}", flush=True)
    ch = hashlib.md5(f"{rid}_k16preempt_m{MARGIN}".encode()).hexdigest()[:16]
    cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav, nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                   on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv, maxdd_nav=excluded.maxdd_nav,
                     nav_f22_adv=excluded.nav_f22_adv, years=excluded.years, n_trades_sim=excluded.n_trades_sim, config_hash=excluded.config_hash, computed_at=now()""",
                (rid, navf, navf, cagr, cagr, dd, nav22, 6.51, len(cvtr), ch))
    cur.execute("update leaderboard_runs set superseded=true where run_id='template/x2_struct_to_k16meta-69338138'")
    con.commit()
    print(f"REGISTERED {NAME} (t{tid}) @K16+meta+preempt m{MARGIN}; superseded k16meta", flush=True)
    con.close(); print("HB_118_DONE", flush=True)


if __name__ == "__main__":
    main()
