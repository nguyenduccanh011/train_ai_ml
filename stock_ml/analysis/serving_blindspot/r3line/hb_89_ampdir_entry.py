# -*- coding: utf-8 -*-
"""hb_89: entry target = amplitude_direction (mfe signed by cross-sectional direction). Synthesis of
hb_88: cs_entry fixed 2024 DIR (-0.098->+0.02) but killed AMP (0.21->0.06) -> NAV down. This target
keeps AMP magnitude but signs it by peer-relative direction -> aim: AMP high AND DIR 2024 positive
AND NAV >= base x31.8. Measure entry `score` IC (DIR fwd10 + AMP mfe20) + full NAV @ K18/K25."""
from __future__ import annotations
import asyncio, copy, sys, os, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
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

TARGETS = {
    "ad_h20": {"type": "amplitude_direction_entry", "horizon": 20, "scale": 0.05},
    "ad_h20s10": {"type": "amplitude_direction_entry", "horizon": 20, "scale": 0.10},
    "ad_h10": {"type": "amplitude_direction_entry", "horizon": 10, "scale": 0.05},
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
            validation_config=base.validation_config, seed=42, description=f"root entry amplitude-direction {tgt}",
            hypothesis="amplitude signed by cross-sectional direction -> keep AMP + fix DIR",
            universe_slug=base.universe_slug, model_mode="ml_only")
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


def score(rid, con, fw):
    sig = pd.read_sql("select symbol,date,signal,score from run_signals where run_id=%s and score is not null",
                      con, params=(rid,))
    sig['date'] = pd.to_datetime(sig['date']); ent = sig[sig.signal == 1]
    m = ent.merge(fw, on=['symbol', 'date']); m['year'] = m.date.dt.year
    def ic(tcol):
        mm = m.dropna(subset=['score', tcol])
        ov = ss.spearmanr(mm.score, mm[tcol]).correlation
        by = {int(y): round(ss.spearmanr(g.score, g[tcol]).correlation, 3) for y, g in mm.groupby('year') if len(g) > 30}
        return ov, by
    tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days "
                     "from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / f"_k89_{rid.replace('/','_')}.csv"; tr.to_csv(cv, index=False)
    def nav(K): return shuffle_stats(NavSim2(str(cv), date_lo="2020-01-01"), K=K, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    return ic('fwd10'), ic('mfe20'), len(tr), nav(18), nav(25)


def main():
    con = psycopg2.connect(**PG)
    syms = pd.read_sql("select distinct symbol from run_signals where run_id='template/ft_rs-8bc8ec4e'", con).symbol.tolist()
    fw = fwd_frames(syms)
    print("=== ampdir entry — base: DIR2024=-0.098 AMP=+0.206 NAV=x31.8 ===", flush=True)
    for name, tgt in TARGETS.items():
        tid = asyncio.run(make(name, tgt)); asyncio.run(async_engine.dispose())
        r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
        (dov, dby), (aov, aby), ntr, n18, n25 = score(rid, con, fw)
        print(f"\n>> {name} (tid={tid}) ntr={ntr} NAV@K18=x{n18:.1f} K25=x{n25:.1f}", flush=True)
        print(f"   DIR fwd10 IC={dov:+.3f} 2021={dby.get(2021)} 2022={dby.get(2022)} 2023={dby.get(2023)} 2024={dby.get(2024)} 2025={dby.get(2025)} 2026={dby.get(2026)}", flush=True)
        print(f"   AMP mfe20 IC={aov:+.3f} 2022={aby.get(2022)} 2023={aby.get(2023)} 2024={aby.get(2024)} 2026={aby.get(2026)}", flush=True)
    con.close(); print("\nHB_89_DONE", flush=True)


if __name__ == "__main__":
    main()
