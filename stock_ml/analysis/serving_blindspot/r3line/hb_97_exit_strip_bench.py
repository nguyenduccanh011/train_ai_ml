# -*- coding: utf-8 -*-
"""hb_97: UNMASK test. hb_95/96: amplitude-exhaustion exit target = raw signal regime-robust hon
velocity NHUNG full-NAV masked by rules. Decisive: do velocity vs amplitude-exhaustion tren ban
STRIP RULE (exit = raw signal only). Neu exh-strip NAV > velocity-strip -> raw exit exh GANH exit
tot hon MOT MINH = internal upgrade THAT (rule co the don gian hoa quanh no). Cache-reuse folds tu
3185 (velocity) + 3263 (exh10). Do NAV@K25 full + per-year, + exit-IC."""
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
RESULTS = REPO / "results"; HERE = Path(__file__).parent
# STRIP = exit raw-signal only (hb_82 v_sigonly)
STRIP = {"exit_priority": ["signal"], "max_hold_bars": 10000, "signal_exit_hold_ext_atr": None,
         "signal_exit_hold_rs_scale": 0.0, "signal_exit_hold_mkt_scale": 0.0, "signal_exit_hold_legage_scale": 0.0,
         "signal_exit_hold_legamp_scale": 0.0, "signal_exit_hold_min_score3_z": None, "signal_exit_protect_lo": None,
         "signal_exit_protect_hi": None, "signal_exit_protect_release_drop_k": None, "signal_exit_skip_if_score3_z": None,
         "exit_snr_extend_threshold": None, "signal_exit_skip_if_mkt_above_ma": None,
         "trailing_struct_donch_win": None, "trailing_struct_trend_only": False}
# also a LIGHT strip: keep max_hold+trailing but drop the regime/hold/protect masks
LIGHT = {"signal_exit_hold_ext_atr": None, "signal_exit_hold_rs_scale": 0.0, "signal_exit_hold_mkt_scale": 0.0,
         "signal_exit_hold_legage_scale": 0.0, "signal_exit_hold_legamp_scale": 0.0, "signal_exit_hold_min_score3_z": None,
         "signal_exit_skip_if_mkt_above_ma": None}
SRC = {"velocity": 3185, "exh10": 3263}
# LIGHT only: full-strip (signal-only, no max_hold) makes amplitude-exhaustion hold forever -> NavSim
# IndexError. LIGHT keeps max_hold+trailing (safe), drops regime/hold/protect masks -> tests if the
# better raw signal needs FEWER masks.
BENCH = {"light": LIGHT}


async def make(name, src_tid, ov):
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(src_tid)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config); ec.update(ov)
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec, validation_config=base.validation_config,
            seed=42, description=f"unmask bench {name}", hypothesis="raw exit carries alone?",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def reuse(tid, src_tid):
    for src in RESULTS.glob(f"tmpl_{src_tid}_*"):
        fp = src.name.split(f"tmpl_{src_tid}_")[1]; dst = RESULTS / f"tmpl_{tid}_{fp}" / "folds"; dst.mkdir(parents=True, exist_ok=True)
        for p in (src / "folds").glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)


def navyr(csv):
    out = {}
    for lo, lab in [("2020-01-01", "full"), ("2022-01-01", "f22"), ("2024-01-01", "f24")]:
        out[lab] = shuffle_stats(NavSim2(str(csv), date_lo=lo), K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)["mean"]
    return out


def main():
    con = psycopg2.connect(**PG)
    print("=== UNMASK: raw exit carries alone? velocity vs amplitude-exhaustion, stripped rules ===", flush=True)
    # also FULL (no strip) reference per source to see mask-gap
    for bench, ov in list(BENCH.items()) + [("full", {})]:
        for lab, src in SRC.items():
            name = f"um_{lab}_{bench}"
            try:
                tid = asyncio.run(make(name, src, ov)); asyncio.run(async_engine.dispose()); reuse(tid, src)
                r = run_template_experiment(template_id=tid, seed=42); rid = r.get("run_id")
                tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
                if len(tr) < 20:
                    print(f"  {bench:5s} {lab:9s}(t{tid}) DEGENERATE ntr={len(tr)}", flush=True); continue
                cv = HERE / f"_k97_{name}.csv"; tr.to_csv(cv, index=False)
                ny = navyr(str(cv))
                print(f"  {bench:5s} {lab:9s}(t{tid}) ntr={len(tr):4d} hold={tr.holding_days.mean():5.1f} "
                      f"full=x{ny['full']:5.2f} f22=x{ny['f22']:4.2f} f24=x{ny['f24']:4.2f}", flush=True)
            except Exception as e:
                print(f"  {bench:5s} {lab:9s} ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
    con.close(); print("HB_97_DONE", flush=True)


if __name__ == "__main__":
    main()
