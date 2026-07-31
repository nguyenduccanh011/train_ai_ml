# -*- coding: utf-8 -*-
"""hb_77: co-optimize EXIT rules tren ENTRY-RS base (ft_rs t3102). Exit optimum cua tôi
(mh14/oxt03/skip300) toi uu tren entry CU (ab_noT) — tren entry-RS moi co the KHAC. Cache-reuse
(ft_rs predictions tmpl_3102_9327ff2ec8). Tim to hop exit vuot ft_rs x26.53."""
from __future__ import annotations
import asyncio, copy, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE, FP = 3102, "9327ff2ec8"
RESULTS = REPO / "results"

COMBOS = {
    "fr_skip250": {"signal_exit_skip_if_mkt_above_ma": 250},
    "fr_skip350": {"signal_exit_skip_if_mkt_above_ma": 350},
    "fr_skip400": {"signal_exit_skip_if_mkt_above_ma": 400},
    "fr_mh12": {"max_hold_bars": 12},
    "fr_mh16": {"max_hold_bars": 16},
    "fr_mh18": {"max_hold_bars": 18},
    "fr_mh20": {"max_hold_bars": 20},
    "fr_oxt025": {"overext_trail_pct": 0.025},
    "fr_oxt04": {"overext_trail_pct": 0.04},
    "fr_mh18_skip400": {"max_hold_bars": 18, "signal_exit_skip_if_mkt_above_ma": 400},
    "fr_mh16_skip350": {"max_hold_bars": 16, "signal_exit_skip_if_mkt_above_ma": 350},
    "fr_holdmkt50": {"signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "ma50"},
}


async def make(name, ov):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            return ex.id
        base = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config); ec.update(ov)
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec,
            validation_config=base.validation_config, seed=42,
            description=f"ft_rs co-optimize exit: {ov}", hypothesis="exit optimum shift on entry-RS base",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def seed_cache(tid):
    src = RESULTS / f"tmpl_{BASE}_{FP}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{FP}" / "folds"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.parquet"):
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    for name, ov in COMBOS.items():
        tid = asyncio.run(make(name, ov)); asyncio.run(async_engine.dispose())
        seed_cache(tid)
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        cur.execute("select composite_score,total_pnl,pf,mdd_per_symbol,trades from leaderboard_runs where run_id=%s", (rid,))
        row = cur.fetchone()
        if row:
            print(f"  {name:16s} t{tid} comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} {ov}", flush=True)
    con.close(); print("HB_77_DONE", flush=True)


if __name__ == "__main__":
    main()
