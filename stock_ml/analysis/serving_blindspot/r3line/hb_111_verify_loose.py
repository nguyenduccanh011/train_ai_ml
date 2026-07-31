# -*- coding: utf-8 -*-
"""hb_111: VERIFY loose_-2.3 +13% (hb_110 seed42 suspicious: ~same trade count, NAV jump). Check
(1) trade overlap base vs loose, (2) MULTI-SEED loose vs base meta@K16. Neu collapse multi-seed ->
overfit/noise; neu hold + trades genuinely differ -> real."""
from __future__ import annotations
import os, sys, warnings, statistics, asyncio, copy, shutil
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from nh_nav2 import NavSim2, shuffle_stats, FEE
from scripts.run_template import run_template_experiment
import hb_109_meta_kdepth as M  # reuse features/build_tr/meta_preds/prun

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; RESULTS = REPO / "results"; SEEDS = [42, 21, 123]


async def make_loose(name, thr):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=thr, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=f"loosen {thr}", hypothesis="verify",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def reuse_folds(tid):
    for src in RESULTS.glob("tmpl_3185_*"):
        fp = src.name.split("tmpl_3185_")[1]; dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"; dst.mkdir(parents=True, exist_ok=True)
        for p in (src / "folds").glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)


def get_tr(con, rid):
    return pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))


def meta_nav(con, rid, cv, feat, Kv=16):
    tr = M.build_tr(con, rid, feat); pm = M.meta_preds(tr)
    return M.prun(NavSim2(str(cv), date_lo="2020-01-01"), pm, Kv=Kv)


def main():
    con = psycopg2.connect(**PG)
    tid = asyncio.run(make_loose("loose_-2.3", -2.3)); asyncio.run(async_engine.dispose()); reuse_folds(tid)
    feat = None; base_navs, loose_navs = [], []
    for sd in SEEDS:
        rb = run_template_experiment(template_id=3185, seed=sd).get("run_id")
        rl = run_template_experiment(template_id=tid, seed=sd).get("run_id")
        tb = get_tr(con, rb); tl = get_tr(con, rl)
        if feat is None: feat = M.features(tb.symbol.unique().tolist())
        cvb = HERE / f"_k111_base_s{sd}.csv"; tb.to_csv(cvb, index=False)
        cvl = HERE / f"_k111_loose_s{sd}.csv"; tl.to_csv(cvl, index=False)
        # overlap (only seed 42)
        if sd == 42:
            mb = set(zip(tb.symbol, tb.entry_date.astype(str))); ml = set(zip(tl.symbol, tl.entry_date.astype(str)))
            print(f"seed42 overlap: base={len(mb)} loose={len(ml)} shared={len(mb & ml)} loose-only={len(ml-mb)} base-only={len(mb-ml)}", flush=True)
        nb, cgb, ddb = meta_nav(con, rb, cvb, feat)
        nl, cgl, ddl = meta_nav(con, rl, cvl, feat)
        base_navs.append(nb); loose_navs.append(nl)
        print(f"seed {sd}: base meta x{nb:.2f} (CAGR {cgb*100:.1f}%) | loose meta x{nl:.2f} (CAGR {cgl*100:.1f}%)", flush=True)
    print(f"\nMEAN: base x{statistics.mean(base_navs):.2f} | loose x{statistics.mean(loose_navs):.2f} ({(statistics.mean(loose_navs)/statistics.mean(base_navs)-1)*100:+.1f}%)", flush=True)
    con.close(); print("HB_111_DONE", flush=True)


if __name__ == "__main__":
    main()
