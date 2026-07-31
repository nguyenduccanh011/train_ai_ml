# -*- coding: utf-8 -*-
"""hb_95: EXIT TARGET upgrade (user intuition: entry biet AMPLITUDE robust -> amplitude-exhaustion
LA tin hieu exit). Swap exit target tren struct_to 3185. Do raw exit IC vs fwd10-ret theo nam
(exit tot: high exit_score -> ban -> fwd_ret THAP -> IC AM & AM MOI NAM = regime-robust) + NAV@K25.
So base velocity_exit (regime-FLIP: 2022 -0.14 dung, 2024 +0.11 dao)."""
from __future__ import annotations
import asyncio, copy, os, sys, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
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
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; BASE = 3185; HERE = Path(__file__).parent

EXIT_TARGETS = {
    "xt_exh10":   {"type": "exit_amplitude_exhaustion", "horizon": 10},
    "xt_exh20":   {"type": "exit_amplitude_exhaustion", "horizon": 20},
    "xt_exh10vn": {"type": "exit_amplitude_exhaustion", "horizon": 10, "vol_normalize": True, "vol_window": 40},
    "xt_zigzag":  {"type": "zigzag_pivot", "tau": 5.0},
    "xt_cs20":    {"type": "cross_sectional_exit", "horizon": 20},
}


async def make(name, tgt):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = []
        for sl in base.component_slots:
            tc = copy.deepcopy(tgt) if sl.slot_type == "exit" else copy.deepcopy(sl.target_config)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                          "target_config": tc})
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=f"exit target {tgt}",
            hypothesis="amplitude-exhaustion / structural exit = regime-robust", universe_slug=base.universe_slug,
            model_mode="ml_only")
        await s.commit(); return t.id


def fwd_frame(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date']); out = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index(); g['fwd10'] = g['close'].shift(-10)/g['close']-1
        out.append(g[['fwd10']].assign(symbol=s).reset_index())
    return pd.concat(out)


def exit_ic(rid, con, fw):
    sig = pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and exit_score is not null", con, params=(rid,))
    sig['date'] = pd.to_datetime(sig['date']); m = sig.merge(fw, on=['symbol', 'date']).dropna(subset=['exit_score', 'fwd10'])
    m['year'] = m.date.dt.year
    ov = ss.spearmanr(m.exit_score, m.fwd10).correlation
    by = {int(y): round(ss.spearmanr(g.exit_score, g.fwd10).correlation, 3) for y, g in m.groupby('year') if len(g) > 30}
    return ov, by


def nav(csv, K=25):
    return shuffle_stats(NavSim2(str(csv), date_lo="2020-01-01"), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]


def main():
    con = psycopg2.connect(**PG)
    syms = pd.read_sql("select distinct symbol from run_signals where run_id='template/ft_rs-8bc8ec4e'", con).symbol.tolist()
    fw = fwd_frame(syms)
    # base 3185 reference
    r = run_template_experiment(template_id=3185, seed=42); rid = r.get("run_id")
    bov, bby = exit_ic(rid, con, fw)
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k95_base.csv"; tr.to_csv(cv, index=False)
    print(f"BASE velocity: NAV=x{nav(str(cv)):.2f} exit-IC={bov:+.3f} 2022={bby.get(2022)} 2023={bby.get(2023)} 2024={bby.get(2024)} 2026={bby.get(2026)} (robust=AM moi nam)", flush=True)
    for name, tgt in EXIT_TARGETS.items():
        tid = asyncio.run(make(name, tgt)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        ov, by = exit_ic(rid, con, fw)
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k95_{name}.csv"; tr.to_csv(cv, index=False)
        print(f"  {name:11s}(t{tid}) NAV=x{nav(str(cv)):.2f} exit-IC={ov:+.3f} 2022={by.get(2022)} 2023={by.get(2023)} 2024={by.get(2024)} 2026={by.get(2026)} ntr={len(tr)}", flush=True)
    con.close(); print("HB_95_DONE", flush=True)


if __name__ == "__main__":
    main()
