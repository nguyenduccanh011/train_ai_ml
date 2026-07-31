# -*- coding: utf-8 -*-
"""hb_92: entry ensemble norm zscore->csrank (CACHE-REUSABLE, post-predict). Engine comment: per-symbol
z FLIPS outcome signal of momentum/mfe heads (IC +0.05 -> z -0.03); csrank PRESERVES (WR 22->37%).
Khop chan doan entry regime-flip (hb_87: DIR IC 2024 -0.098). Doi direction-heads (rev/cont/fwd_pen)
sang csrank, giu mfe=zscore. Reuse 3185 folds (norm KHONG trong _fp). NAV@K25 + blended DIR/AMP IC/year.
z->pct: Phi(0.9)=0.816, Phi(0.7)=0.758 (giu firing-rate)."""
from __future__ import annotations
import asyncio, copy, os, sys, shutil, logging, warnings
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
DUCK = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"; RESULTS = REPO / "results"; HERE = Path(__file__).parent
BASE = 3185
# (ensemble_key, csrank_pct) per variant -> patch norm+z_threshold
VARIANTS = {
    "cr_rev":    [("entry_ensemble", 0.816)],
    "cr_cont":   [("entry_ensemble2", 0.758)],
    "cr_fwd":    [("entry_ensemble4", 0.758)],
    "cr_alldir": [("entry_ensemble", 0.816), ("entry_ensemble2", 0.758), ("entry_ensemble4", 0.758)],
    "cr_all":    [("entry_ensemble", 0.816), ("entry_ensemble2", 0.758), ("entry_ensemble3", 0.758), ("entry_ensemble4", 0.758)],
}


async def make(name, patches):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config)
        for key, pct in patches:
            d = dict(ec.get(key) or {}); d["norm"] = "csrank"; d["z_threshold"] = pct; ec[key] = d
        t = await repo.create(name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id, component_slots=slots,
            direction=base.direction, signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec,
            validation_config=base.validation_config, seed=42, description=f"csrank norm {patches}",
            hypothesis="csrank preserves regime signal where zscore flips", universe_slug=base.universe_slug,
            model_mode="ml_only")
        await s.commit(); return t.id


def reuse_folds(tid):
    for src in RESULTS.glob("tmpl_3185_*"):
        fp = src.name.split("tmpl_3185_")[1]
        dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"; dst.mkdir(parents=True, exist_ok=True)
        for p in (src / "folds").glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)


def fwd_frames(syms):
    d = duckdb.connect(DUCK, read_only=True); ph = ",".join("?" * len(syms))
    px = d.execute(f"select symbol,date,close,high from ohlcv where timeframe='1D' and symbol in ({ph}) order by symbol,date", syms).fetchdf()
    d.close(); px['date'] = pd.to_datetime(px['date']); out = []
    for s, g in px.groupby('symbol'):
        g = g.set_index('date').sort_index(); g['fwd10'] = g['close'].shift(-10)/g['close']-1
        fmax = pd.concat([g['high'].shift(-k) for k in range(1, 21)], axis=1).max(axis=1); g['mfe20'] = fmax/g['close']-1
        out.append(g[['fwd10', 'mfe20']].assign(symbol=s).reset_index())
    return pd.concat(out)


def score(rid, con, fw):
    sig = pd.read_sql("select symbol,date,signal,score from run_signals where run_id=%s and score is not null", con, params=(rid,))
    sig['date'] = pd.to_datetime(sig['date']); ent = sig[sig.signal == 1]
    m = ent.merge(fw, on=['symbol', 'date']); m['year'] = m.date.dt.year
    def ic(t):
        mm = m.dropna(subset=['score', t]); ov = ss.spearmanr(mm.score, mm[t]).correlation
        by = {int(y): round(ss.spearmanr(g.score, g[t]).correlation, 3) for y, g in mm.groupby('year') if len(g) > 30}
        return ov, by
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / f"_k92_{rid.replace('/','_')}.csv"; tr.to_csv(cv, index=False)
    nav = shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    return ic('fwd10'), ic('mfe20'), len(tr), nav


def main():
    con = psycopg2.connect(**PG)
    syms = pd.read_sql("select distinct symbol from run_signals where run_id='template/ft_rs-8bc8ec4e'", con).symbol.tolist()
    fw = fwd_frames(syms)
    # base 3185 in-harness reference
    r = run_template_experiment(template_id=3185, seed=42); (bd, bdy), (ba, bay), bn, bnav = score(r.get("run_id"), con, fw)
    print(f"BASE 3185: ntr={bn} NAV@K25=x{bnav:.2f} DIR IC={bd:+.3f} 2022={bdy.get(2022)} 2024={bdy.get(2024)} 2026={bdy.get(2026)} | AMP={ba:+.3f}", flush=True)
    for name, patches in VARIANTS.items():
        tid = asyncio.run(make(name, patches)); asyncio.run(async_engine.dispose()); reuse_folds(tid)
        r = run_template_experiment(template_id=tid, seed=42); (dov, dby), (aov, aby), ntr, nav = score(r.get("run_id"), con, fw)
        print(f"  {name:9s}(t{tid}) ntr={ntr} NAV=x{nav:.2f} ({(nav/bnav-1)*100:+.1f}%) DIR={dov:+.3f} 2022={dby.get(2022)} 2024={dby.get(2024)} 2026={dby.get(2026)} | AMP={aov:+.3f}", flush=True)
    con.close(); print("HB_92_DONE", flush=True)


if __name__ == "__main__":
    main()
