# -*- coding: utf-8 -*-
"""hb_83: CAI THIEN GOC exit signal. Raw exit IC~0 & lat dau regime (2022 -0.14 dung, 2024 +0.11
nguoc) vi velocity_exit (downside cua so 20) conflate dip-tam vs dinh-that. Thu target REAL-TOP
(downside window DAI = dinh that ben vung) -> head hoc sell regime-robust. Danh gia tren ban da
STRIP MASK (exit chi raw signal) de cai thien KHONG bi rule che. Do v_sigonly NAV + exit IC by year.
"""
from __future__ import annotations
import asyncio, copy, sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
import psycopg2, duckdb, pandas as pd, numpy as np, scipy.stats as ss
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
from scripts.run_template import run_template_experiment
from nh_nav2 import NavSim2, shuffle_stats

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 3102; HERE = Path(__file__).parent
# STRIP config: exit chi raw signal (giong v_sigonly hb_82)
STRIP = {"exit_priority": ["signal"], "max_hold_bars": 10000,
         "signal_exit_hold_ext_atr": None, "signal_exit_hold_rs_scale": 0.0, "signal_exit_hold_mkt_scale": 0.0,
         "signal_exit_hold_legage_scale": 0.0, "signal_exit_hold_legamp_scale": 0.0, "signal_exit_hold_min_score3_z": None,
         "signal_exit_protect_lo": None, "signal_exit_protect_hi": None, "signal_exit_protect_release_drop_k": None,
         "signal_exit_skip_if_score3_z": None, "exit_snr_extend_threshold": None, "signal_exit_skip_if_mkt_above_ma": None}
# exit target candidates (real-top = downside horizon DAI)
TARGETS = {
    "e_base": {"type": "velocity_exit_regression", "horizon": 20, "upside_horizon": 8, "vol_normalize": True, "vol_window": 40},
    "ec_cs5": {"type": "cross_sectional_exit", "horizon": 5},
    "ec_cs10": {"type": "cross_sectional_exit", "horizon": 10},
    "ec_cs20": {"type": "cross_sectional_exit", "horizon": 20},
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
        ec = copy.deepcopy(base.engine_config); ec.update(STRIP)
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec,
            validation_config=base.validation_config, seed=42, description=f"root exit target {tgt} on STRIPPED",
            hypothesis="real-top target -> regime-robust raw exit", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def exit_ic(rid, con):
    sig = pd.read_sql("select symbol,date,exit_score from run_signals where run_id=%s and exit_score is not null", con, params=(rid,))
    sig['date'] = pd.to_datetime(sig['date']); syms = sig.symbol.unique().tolist()
    d = duckdb.connect(r"market_data/market.duckdb", read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date'])
    fr = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index(); g['fwd10'] = g['close'].shift(-10) / g['close'] - 1
        fr.append(g[['fwd10']].assign(symbol=s).reset_index())
    m = sig.merge(pd.concat(fr), on=['symbol', 'date']).dropna(subset=['exit_score', 'fwd10']); m['year'] = m.date.dt.year
    ov = ss.spearmanr(m.exit_score, m.fwd10).correlation
    by = {int(y): round(ss.spearmanr(g.exit_score, g.fwd10).correlation, 3) for y, g in m.groupby('year')}
    return ov, by


def main():
    con = psycopg2.connect(**PG)
    print("=== ROOT exit-target on STRIPPED bench — v_sigonly NAV + exit IC (nen AM) ===", flush=True)
    for name, tgt in TARGETS.items():
        tid = asyncio.run(make(name, tgt)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days "
                         "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k83_{name}.csv"; tr.to_csv(cv, index=False)
        nav = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=18, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
        ov, by = exit_ic(rid, con)
        print(f"  {name:9s} ntr={len(tr):4d} hold={tr.holding_days.mean():4.1f} NAV_sig@K18=x{nav:5.1f} "
              f"| IC={ov:+.3f} 2022={by.get(2022)} 2024={by.get(2024)} 2023={by.get(2023)}", flush=True)
    con.close(); print("HB_83_DONE", flush=True)


if __name__ == "__main__":
    main()
