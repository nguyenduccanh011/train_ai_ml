# -*- coding: utf-8 -*-
"""hb_96: PATH-AWARE exit features (root reasoning: exit path-dependent, ML head stateless -> rules
mask ~2x). Swap exit feature set exit_vol_rs -> exit_vol_path (+aroon age, up_days, dist_63d_low leg,
dist_20d_high giveback) tren struct_to 3185. Optionally pair voi exit target tot nhat tu hb_95 (BEST_TGT).
Do raw exit IC theo nam (robust=AM moi nam) + NAV@K25. Neu path-feat manh raw signal -> co the thao rule."""
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
# hb_95: amplitude-exhaustion h10 = most regime-robust raw exit (2024 flip +0.119->+0.037)
BEST_TGT = {"type": "exit_amplitude_exhaustion", "horizon": 10}
VARIANTS = {
    "xf_path":    {"exit_fs": "exit_vol_path", "tgt": None},      # path features, velocity target (isolate path effect)
    "xf_path_exh": {"exit_fs": "exit_vol_path", "tgt": BEST_TGT}, # path features + amplitude-exhaustion (both upgrades)
    "xf_rs_exh":  {"exit_fs": "exit_vol_rs", "tgt": BEST_TGT},    # amplitude-exhaustion on base features (isolate target effect on NAV)
}


async def make(name, exit_fs, tgt):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = []
        for sl in base.component_slots:
            fs = exit_fs if (sl.slot_type == "exit" and exit_fs) else sl.feature_set_name
            tc = copy.deepcopy(tgt) if (sl.slot_type == "exit" and tgt) else copy.deepcopy(sl.target_config)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": fs, "target_config": tc})
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=f"exit path-feat {exit_fs} tgt={tgt}",
            hypothesis="path-proxy features give stateless exit head path-awareness", universe_slug=base.universe_slug,
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
    r = run_template_experiment(template_id=3185, seed=42); rid = r.get("run_id")
    bov, bby = exit_ic(rid, con, fw)
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k96_base.csv"; tr.to_csv(cv, index=False)
    print(f"BASE exit_vol_rs: NAV=x{nav(str(cv)):.2f} exit-IC={bov:+.3f} 2022={bby.get(2022)} 2023={bby.get(2023)} 2024={bby.get(2024)} 2026={bby.get(2026)}", flush=True)
    for name, cfg in VARIANTS.items():
        tid = asyncio.run(make(name, cfg["exit_fs"], cfg["tgt"])); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        ov, by = exit_ic(rid, con, fw)
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
        cv = HERE / f"_k96_{name}.csv"; tr.to_csv(cv, index=False)
        print(f"  {name:11s}(t{tid}) NAV=x{nav(str(cv)):.2f} exit-IC={ov:+.3f} 2022={by.get(2022)} 2023={by.get(2023)} 2024={by.get(2024)} 2026={by.get(2026)} ntr={len(tr)}", flush=True)
    con.close(); print("HB_96_DONE", flush=True)


if __name__ == "__main__":
    main()
