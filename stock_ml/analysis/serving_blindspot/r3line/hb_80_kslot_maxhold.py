# -*- coding: utf-8 -*-
"""hb_80: PHA TRAN slot-lockup. composite⟂NAV ep max_hold ngan (giu lau -> lockup K=25 slot ->
NAV giam). Lever CHUA THU: K (so slot). Nhieu slot hon -> het lockup -> giu lau CO THE thang.
Tao bien the max_hold tren ft_rs (cache-reuse) roi NavSim o K=25/35/50/70. Tim (max_hold,K) toi da NAV."""
from __future__ import annotations
import asyncio, copy, shutil, sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE, FP = 3102, "9327ff2ec8"
RESULTS = REPO / "results"; HERE = Path(__file__).parent
VARIANTS = {"k_mh14": {}, "k_mh20": {"max_hold_bars": 20}, "k_mh30": {"max_hold_bars": 30},
            "k_mh50": {"max_hold_bars": 50}, "k_uncap": {"max_hold_bars": 10000}}
KS = [25, 35, 50, 70]


async def make(name, ov):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
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
            validation_config=base.validation_config, seed=42, description=f"K-sweep {ov}",
            hypothesis="higher K breaks slot-lockup", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    trades = {}
    for name, ov in VARIANTS.items():
        tid = asyncio.run(make(name, ov)); asyncio.run(async_engine.dispose())
        src = RESULTS / f"tmpl_{BASE}_{FP}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{FP}" / "folds"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        t = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades "
                        "where run_id=%s and exit_date is not null", con, params=(rid,))
        csv = HERE / f"_k80_{name}.csv"; t.to_csv(csv, index=False); trades[name] = str(csv)
        print(f"  built {name}: {len(t)} tr", flush=True)
    print(f"\n=== NAV (adv) theo (max_hold x K) — pha tran slot-lockup? ===", flush=True)
    print("  " + "variant".ljust(10) + "".join(f"K={k}".rjust(9) for k in KS), flush=True)
    for name in VARIANTS:
        row = ""
        for k in KS:
            m = shuffle_stats(NavSim2(trades[name], date_lo="2020-01-01"), K=k, roundtrip=0.006,
                              settle_lag=2, advance_fee=0.0008, n=20)["mean"]
            row += f"x{m:.2f}".rjust(9)
        print(f"  {name:10s}{row}", flush=True)
    con.close(); print("HB_80_DONE", flush=True)


if __name__ == "__main__":
    main()
