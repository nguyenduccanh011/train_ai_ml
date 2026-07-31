# -*- coding: utf-8 -*-
"""hb_93: DECOMPOSE entry ensemble union (line 2547: buy|rev|cont|mfe|fwd). Ngach A buoc 1: head nao
gay bay 2024? Disable tung head (z_threshold=99, cache-reusable) tren struct_to 3185, do NAV@K25 full
+ pnl_pct sum THEO NAM + ntr. Neu tat mfe-head (ens3=score4) cai thien 2024 (pnl bot am) nhung hai
nam-tot -> xac nhan amplitude-only-union = bay -> dung agreement-gate (mfe AND direction)."""
from __future__ import annotations
import asyncio, copy, os, sys, shutil, logging, warnings
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
RESULTS = REPO / "results"; HERE = Path(__file__).parent; BASE = 3185
HI = 99.0  # disable a head
# key -> which ensemble to disable (None=base all-on)
VARIANTS = {
    "base":        [],
    "no_h1_rev":   ["entry_ensemble"],
    "no_h2_cont":  ["entry_ensemble2"],
    "no_h3_mfe":   ["entry_ensemble3"],
    "no_h4_fwd":   ["entry_ensemble4"],
    "only_primary": ["entry_ensemble", "entry_ensemble2", "entry_ensemble3", "entry_ensemble4"],
}


async def make(name, disable):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config)
        for key in disable:
            d = dict(ec.get(key) or {}); d["z_threshold"] = HI; ec[key] = d
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec,
            validation_config=base.validation_config, seed=42, description=f"head-decomp disable {disable}",
            hypothesis="which head causes 2024 trap", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def reuse_folds(tid):
    for src in RESULTS.glob("tmpl_3185_*"):
        fp = src.name.split("tmpl_3185_")[1]; dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"; dst.mkdir(parents=True, exist_ok=True)
        for p in (src / "folds").glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)


def score(rid, con):
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,pnl_pct from run_trades "
                     "where run_id=%s and exit_date is not null", con, params=(rid,))
    tr['exit_date'] = pd.to_datetime(tr['exit_date']); tr['yr'] = tr.exit_date.dt.year
    cv = HERE / f"_k93_{rid.replace('/','_')}.csv"; tr[['symbol','entry_date','exit_date','entry_price','exit_price']].to_csv(cv, index=False)
    nav = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    yr = tr.groupby('yr').pnl_pct.sum().round(1).to_dict()
    return len(tr), nav, yr


def main():
    con = psycopg2.connect(**PG)
    print("=== head decompose (disable each union head) — NAV@K25 + sum pnl_pct/year ===", flush=True)
    yrs = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    for name, disable in VARIANTS.items():
        tid = asyncio.run(make(name, disable)); asyncio.run(async_engine.dispose()); reuse_folds(tid)
        r = run_template_experiment(template_id=tid, seed=42); ntr, nav, yr = score(r.get("run_id"), con)
        ys = " ".join(f"{y}={yr.get(y,0):+.0f}" for y in yrs)
        print(f"  {name:13s}(t{tid}) ntr={ntr:4d} NAV=x{nav:5.2f} | pnl/yr: {ys}", flush=True)
    con.close(); print("HB_93_DONE", flush=True)


if __name__ == "__main__":
    main()
