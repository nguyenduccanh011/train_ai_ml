# -*- coding: utf-8 -*-
"""hb_82: THAO DO MASK exit. Rule stack (max_hold/overext/trailing/regime-skip/protect/snr/hold)
ROBUSTIFY = MASK raw exit ML signal. Strip dan tren ft_rs (cache-reuse) -> do raw-signal-only
con lai bao nhieu. Gap full-vs-stripped = muc mask. Neu sig-only sup nhieu -> rule ganh nang,
raw signal yeu -> can cai thien GOC (target/feature exit head), do tren ban da strip (khong mask)."""
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

# strip knobs (None/uncap = tat mask). exit_priority chi con 'signal' o sig-only.
OFF_HOLD = {"signal_exit_hold_ext_atr": None, "signal_exit_hold_rs_scale": 0.0,
            "signal_exit_hold_mkt_scale": 0.0, "signal_exit_hold_legage_scale": 0.0,
            "signal_exit_hold_legamp_scale": 0.0, "signal_exit_hold_min_score3_z": None}
OFF_PROTECT = {"signal_exit_protect_lo": None, "signal_exit_protect_hi": None,
               "signal_exit_protect_release_drop_k": None, "signal_exit_skip_if_score3_z": None}
OFF_SNR = {"exit_snr_extend_threshold": None}
OFF_SKIP = {"signal_exit_skip_if_mkt_above_ma": None}

VARIANTS = {
    "v_full": {},
    "v_no_skip": {**OFF_SKIP},
    "v_no_hold": {**OFF_HOLD},
    "v_no_protect": {**OFF_PROTECT},
    "v_no_maxhold": {"max_hold_bars": 10000},
    "v_no_overext": {"exit_priority": ["max_hold", "trailing_stop", "signal"]},
    "v_no_trail": {"exit_priority": ["max_hold", "overext", "signal"]},
    # RAW SIGNAL ONLY: exit chi tren z(exit)>thr, tat het rule mask
    "v_sigonly": {"exit_priority": ["signal"], "max_hold_bars": 10000,
                  **OFF_HOLD, **OFF_PROTECT, **OFF_SNR, **OFF_SKIP},
    # sig-only NHUNG giu max_hold (chan buy&hold)
    "v_sig_mh": {"exit_priority": ["max_hold", "signal"],
                 **OFF_HOLD, **OFF_PROTECT, **OFF_SNR, **OFF_SKIP},
}


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
            validation_config=base.validation_config, seed=42, description=f"strip mask: {ov}",
            hypothesis="reveal masking", universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def nav(csv, K):
    return shuffle_stats(NavSim2(csv, date_lo="2020-01-01"), K=K, roundtrip=0.006,
                         settle_lag=2, advance_fee=0.0008, n=20)


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    print("=== THAO DO MASK exit (ft_rs) — NAV @ K=18 / K=25, so raw-signal ===", flush=True)
    print(f"  {'variant':13s}{'ntr':>6s}{'hold':>6s}{'K18':>9s}{'K25':>9s}{'DDk18':>7s}", flush=True)
    for name, ov in VARIANTS.items():
        tid = asyncio.run(make(name, ov)); asyncio.run(async_engine.dispose())
        src = RESULTS / f"tmpl_{BASE}_{FP}" / "folds"; dst = RESULTS / f"tmpl_{tid}_{FP}" / "folds"
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*.parquet"):
            if not (dst / p.name).exists(): shutil.copy2(p, dst / p.name)
        r = run_template_experiment(template_id=tid, seed=42)
        tr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price,holding_days "
                         "from run_trades where run_id=%s and exit_date is not null", con, params=(r.get("run_id"),))
        cv = HERE / f"_k82_{name}.csv"; tr.to_csv(cv, index=False)
        s18 = nav(str(cv), 18); s25 = nav(str(cv), 25)
        print(f"  {name:13s}{len(tr):6d}{tr.holding_days.mean():6.1f}"
              f"x{s18['mean']:6.1f} x{s25['mean']:6.1f}{s18['dd_worst']*100:6.0f}%", flush=True)
    con.close(); print("HB_82_DONE", flush=True)


if __name__ == "__main__":
    main()
