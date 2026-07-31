# -*- coding: utf-8 -*-
"""hb_122: REGISTER x2_struct_to_k25preempt = struct_to t3185 @ board-K K25 + meta-fill + SLOT-
PREEMPTION (R2 m0.01). hb_121: K25 base 66.7% -> preempt 71.2%, DD -13.8 (= board -13.7), 3/3 seed.
FAIR-anchor: cung K25, cung DD, chi execution thong minh hon -> 'model manh hon' tren metric CONG
BANG. Giu k16preempt (aggressive) song song."""
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
NAME = "x2_struct_to_k25preempt"
MARGIN = 0.01
DESC = ("[K25 board-K + META-FILL + SLOT-PREEMPTION — FAIR-anchor cung thuoc do board (K25, DD-matched)] "
        "Cung signal double-RS+struct-trail. Execution: K=25 (board standard) + fill uu tien META + SLOT-"
        "PREEMPTION (full-book + tin hieu conviction moi -> duoi held entry-meta-prio thap nhat neu new-"
        "held>0.01, realize @mark; single-book equal-weight, causal). Multi-seed 42/21/123: CAGR 71.2%/"
        "DD-13.8%/NAV x30.6 (3/3 seed, +4.5pp vs board base 66.7% CUNG DD -13.7). Preemption fire 125x "
        "ngay o K25 -> cai thien tren metric CONG BANG (khong chi operating-point aggressive). Con k16preempt "
        "89.3% la profile TAN CONG song song.")


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
            validation_config=base.validation_config, seed=42, description=DESC, hypothesis="K25 board + slot-preemption fair-anchor",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    tid = asyncio.run(make_tmpl()); asyncio.run(async_engine.dispose())
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("update strategy_templates set description=%s where id=%s", (DESC, tid)); con.commit()
    r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k122.csv"; cvtr.to_csv(cv, index=False)
    feat = M.features(cvtr.symbol.unique().tolist())
    tr = M.build_tr(con, rid, feat); pm = M.meta_preds(tr, tgt='t_pnl')
    P.K = 25
    navf, cagr, dd, ev = P.prun_causal(NavSim2(str(cv), date_lo="2020-01-01"), pm, rule="R2", margin=MARGIN)
    nav22, _, _, _ = P.prun_causal(NavSim2(str(cv), date_lo="2022-01-01"), pm, rule="R2", margin=MARGIN)
    print(f"K25+preempt seed42: NAV=x{navf:.2f} CAGR={cagr*100:.1f}% DD={dd*100:.1f}% f22=x{nav22:.2f} evict={ev}", flush=True)
    ch = hashlib.md5(f"{rid}_k25preempt_m{MARGIN}".encode()).hexdigest()[:16]
    cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav, nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                   on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv, maxdd_nav=excluded.maxdd_nav,
                     nav_f22_adv=excluded.nav_f22_adv, years=excluded.years, n_trades_sim=excluded.n_trades_sim, config_hash=excluded.config_hash, computed_at=now()""",
                (rid, navf, navf, cagr, cagr, dd, nav22, 6.51, len(cvtr), ch))
    con.commit()
    print(f"REGISTERED {NAME} (t{tid}) @K25+preempt fair-anchor (k16preempt kept)", flush=True)
    con.close(); print("HB_122_DONE", flush=True)


if __name__ == "__main__":
    main()
