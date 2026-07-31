# -*- coding: utf-8 -*-
"""hb_99: MULTI-HEAD exit (user #4 "nhieu head bat nhieu thong tin exit"). exit_ensemble wired
(experiment.py:2518 union-sell khi z(exit_score2)>thr). Main head=velocity (downside-timing, STRONG
nhung regime-FLIP 2024); head2=amplitude-exhaustion (regime-ROBUST) UNION -> bat tops ma velocity lo
o dead-year. Sweep z_threshold head2. So base struct_to 27.14. NAV@K25 + per-year pnl."""
from __future__ import annotations
import asyncio, copy, os, sys, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 3185; HERE = Path(__file__).parent
EXH = {"type": "exit_amplitude_exhaustion", "horizon": 10}
CS = {"type": "cross_sectional_exit", "horizon": 20}
VARIANTS = {
    "xe_exh_z25": {"target": EXH, "z_threshold": 2.5},
    "xe_exh_z30": {"target": EXH, "z_threshold": 3.0},
    "xe_exh_z20": {"target": EXH, "z_threshold": 2.0},
    "xe_cs_z25":  {"target": CS, "z_threshold": 2.5},
}


async def make(name, xens):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config); ec["exit_ensemble"] = xens
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec, validation_config=base.validation_config,
            seed=42, description=f"multi-head exit {xens}", hypothesis="2nd robust exit head unions to catch dead-year tops",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def score(rid, con):
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,pnl_pct,holding_days from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    tr['exit_date'] = pd.to_datetime(tr['exit_date']); tr['yr'] = tr.exit_date.dt.year
    cv = HERE / f"_k99_{rid.replace('/','_')}.csv"; tr[['symbol','entry_date','exit_date','entry_price','exit_price']].to_csv(cv, index=False)
    nav = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    yr = tr.groupby('yr').pnl_pct.sum().round(1).to_dict()
    return len(tr), tr.holding_days.mean(), nav, yr


def main():
    con = psycopg2.connect(**PG); yrs = [2021, 2022, 2023, 2024, 2025, 2026]
    r = run_template_experiment(template_id=3185, seed=42); ntr, hd, nav, yr = score(r.get("run_id"), con)
    print(f"BASE struct_to: ntr={ntr} hold={hd:.1f} NAV=x{nav:.2f} | " + " ".join(f"{y}={yr.get(y,0):+.0f}" for y in yrs), flush=True)
    for name, xens in VARIANTS.items():
        tid = asyncio.run(make(name, xens)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=42); ntr, hd, nav, yr = score(r.get("run_id"), con)
        print(f"  {name:11s}(t{tid}) ntr={ntr} hold={hd:.1f} NAV=x{nav:.2f} | " + " ".join(f"{y}={yr.get(y,0):+.0f}" for y in yrs), flush=True)
    con.close(); print("HB_99_DONE", flush=True)


if __name__ == "__main__":
    main()
