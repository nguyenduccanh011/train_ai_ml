# -*- coding: utf-8 -*-
"""hb_90: multi-seed (42/21/123) NAV @ K25 de xac lap TRUE ranking (board seed-42 co the noise,
xr_liq da bi bac). So base 3102 (exit_vol_downpress) vs 3185 x2_struct_to (dblRS exit_vol_rs +
struct-trail) vs NEW combo x2_struct_rgskip (3185 + regime-skip MA300 winner_only). run_id khong
chua seed -> dump trades ngay sau moi run. NAV full(2020)+f22(2022), mean+/-std."""
from __future__ import annotations
import asyncio, copy, os, sys, statistics, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
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
HERE = Path(__file__).parent
SEEDS = [42, 21, 123]
RGSKIP = {"signal_exit_skip_if_mkt_above_ma": 300, "signal_exit_skip_if_mkt_winner_only": True,
          "signal_exit_skip_if_score3_z": 1.6, "overext_skip_bull_enabled": True}


async def make_combo():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name("x2_struct_rgskip")
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config); ec.update(RGSKIP)
        t = await repo.create(name="x2_struct_rgskip", market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec,
            validation_config=base.validation_config, seed=42, description="dblRS + struct-trail + regime-skip MA300 winner_only",
            hypothesis="struct-trail + regime-skip stack", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def dump(rid, out):
    con = psycopg2.connect(**PG)
    t = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                    "where run_id=%s and exit_date is not null and entry_price is not null and exit_price is not null",
                    con, params=(rid,)); con.close()
    t.to_csv(out, index=False); return len(t)


def nav(csv, lo, K=25):
    return shuffle_stats(NavSim2(str(csv), date_lo=lo), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]


def main():
    combo = asyncio.run(make_combo()); asyncio.run(async_engine.dispose())
    cands = [(3102, "base_ftrs"), (3185, "struct_to"), (combo, "struct_rgskip")]
    print(f"combo tid={combo}. CANDS={cands} SEEDS={SEEDS}", flush=True)
    csvs = {}
    for tid, lab in cands:
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd); rid = r.get("run_id")
            out = HERE / f"_k90_{lab}_s{sd}.csv"; n = dump(rid, out) if rid else 0
            csvs[(lab, sd)] = out
            print(f"RAN {lab} s{sd} -> {rid} ({n} tr)", flush=True)
    print("\n== NAV per seed (adv, K=25) ==", flush=True)
    agg = {}
    for _, lab in cands:
        full = [nav(csvs[(lab, sd)], "2020-01-01") for sd in SEEDS]
        f22 = [nav(csvs[(lab, sd)], "2022-01-01") for sd in SEEDS]
        for i, sd in enumerate(SEEDS):
            print(f"  {lab:15s} s{sd:<4d} full x{full[i]:6.2f}  f22 x{f22[i]:5.2f}", flush=True)
        agg[lab] = (statistics.mean(full), statistics.pstdev(full), statistics.mean(f22), statistics.pstdev(f22))
    print("\n== MEAN over seeds (vs base_ftrs) ==", flush=True)
    bf, _, bff, _ = agg["base_ftrs"]
    for _, lab in cands:
        mf, sf, mff, sff = agg[lab]
        print(f"  {lab:15s} full x{mf:6.2f}+/-{sf:4.2f} ({(mf/bf-1)*100:+5.1f}%)  f22 x{mff:5.2f}+/-{sff:4.2f} ({(mff/bff-1)*100:+5.1f}%)", flush=True)
    print("HB_90_DONE", flush=True)


if __name__ == "__main__":
    main()
