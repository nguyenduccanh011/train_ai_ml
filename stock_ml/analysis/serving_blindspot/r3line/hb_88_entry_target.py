# -*- coding: utf-8 -*-
"""hb_88: ROOT entry target swap. hb_87 -> entry head biet AMP (mfe IC +0.21 moi nam) khong biet
DIR (fwd IC lat dau 2024 -0.10, DIR-CS 2024 -0.125 = bat direction bang market-beta). Thu
cross_sectional_entry target (market-neutral: buy names outperform PEERS) -> ep head hoc direction
regime-robust. Do entry `score` IC by year (DIR fwd10 + AMP mfe20) + full-model NAV @ K18/K25.
So voi base 3102 (DIR 2024=-0.098, AMP +0.206)."""
from __future__ import annotations
import asyncio, copy, sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, duckdb, pandas as pd, numpy as np, scipy.stats as ss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"
BASE = 3102; HERE = Path(__file__).parent

ENTRY_TARGETS = {
    "en_cs10": {"type": "cross_sectional_entry", "horizon": 10},
    "en_cs20": {"type": "cross_sectional_entry", "horizon": 20},
    "en_fwd20": {"type": "forward_return_regression", "horizon": 20},
}


async def make(name, tgt):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = []
        for sl in base.component_slots:
            tc = copy.deepcopy(tgt) if sl.slot_type == "entry" else copy.deepcopy(sl.target_config)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                          "target_config": tc})
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=f"root entry target {tgt}",
            hypothesis="cross-sectional entry -> regime-robust direction", universe_slug=base.universe_slug,
            model_mode="ml_only")
        await s.commit(); return t.id


def fwd_frames(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close,high from ohlcv where timeframe='1D' and symbol in ({ph}) "
                   f"order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date'])
    out = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index()
        g['fwd10'] = g['close'].shift(-10) / g['close'] - 1
        fmax = pd.concat([g['high'].shift(-k) for k in range(1, 21)], axis=1).max(axis=1)
        g['mfe20'] = fmax / g['close'] - 1
        out.append(g[['fwd10', 'mfe20']].assign(symbol=s).reset_index())
    return pd.concat(out)


def entry_ic(rid, con, fw):
    sig = pd.read_sql("select symbol,date,signal,score from run_signals where run_id=%s and score is not null",
                      con, params=(rid,))
    sig['date'] = pd.to_datetime(sig['date']); ent = sig[sig.signal == 1]
    m = ent.merge(fw, on=['symbol', 'date']); m['year'] = m.date.dt.year
    def ic(tcol):
        mm = m.dropna(subset=['score', tcol])
        ov = ss.spearmanr(mm.score, mm[tcol]).correlation
        by = {int(y): round(ss.spearmanr(g.score, g[tcol]).correlation, 3) for y, g in mm.groupby('year') if len(g) > 30}
        return ov, by
    return ic('fwd10'), ic('mfe20')


def nav(csv, K):
    return shuffle_stats(NavSim2(csv, date_lo="2020-01-01"), K=K, roundtrip=0.006,
                         settle_lag=2, advance_fee=0.0008, n=20)["mean"]


def main():
    con = psycopg2.connect(**PG)
    # fwd frames for full universe (from base signals symbols)
    syms = pd.read_sql("select distinct symbol from run_signals where run_id='template/ft_rs-8bc8ec4e'", con).symbol.tolist()
    fw = fwd_frames(syms)
    print("=== ROOT entry-target swap — entry `score` IC + full NAV. base: DIR2024=-0.098 AMP=+0.206 ===", flush=True)
    for name, tgt in ENTRY_TARGETS.items():
        tid = asyncio.run(make(name, tgt)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        (dov, dby), (aov, aby) = entry_ic(rid, con, fw)
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days "
                         "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k88_{name}.csv"; tr.to_csv(cv, index=False)
        n18, n25 = nav(str(cv), 18), nav(str(cv), 25)
        print(f"\n  {name} (tid={tid}) ntr={len(tr)} NAV@K18=x{n18:.1f} K25=x{n25:.1f}", flush=True)
        print(f"    DIR fwd10 IC={dov:+.3f} 2022={dby.get(2022)} 2023={dby.get(2023)} 2024={dby.get(2024)} 2026={dby.get(2026)}", flush=True)
        print(f"    AMP mfe20 IC={aov:+.3f} 2022={aby.get(2022)} 2024={aby.get(2024)} 2026={aby.get(2026)}", flush=True)
    con.close(); print("\nHB_88_DONE", flush=True)


if __name__ == "__main__":
    main()
