"""SIGNAL-RECOMBINE probe: clone champion 2646, override ONLY the probed keys, run seed 42.
Usage: python stock_ml/analysis/serving_blindspot/signalq/rq_probe.py <probe_id>
Cache-hit config-only changes (no head retrain). One probe per process (kill-guard outside).
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2646  # champion n2_2643_wavestruct_la05_lamp02 (composite seed-42 729.6) — read-only

# probe_id -> (engine-dict deep updates, top-level overrides)
PROBES = {
    # 1. main-head entry_threshold micro-grid (champion -1.9)
    "rq_et17": ({}, {"entry_threshold": -1.7}),
    "rq_et21": ({}, {"entry_threshold": -2.1}),
    # 2. ensemble z loosening one-at-a-time (champion 0.7 each)
    "rq_e2z05": ({"entry_ensemble2": {"z_threshold": 0.5}}, {}),
    "rq_e3z05": ({"entry_ensemble3": {"z_threshold": 0.5}}, {}),
    "rq_e4z05": ({"entry_ensemble4": {"z_threshold": 0.5}}, {}),
    # 3. norm switch on ens4 (fwd-return-penalized head); thr becomes a percentile for csrank.
    #    0.75 ~ matches z>0.7 one-sided firing rate (~24% of bars).
    "rq_e4csr75": ({"entry_ensemble4": {"norm": "csrank", "z_threshold": 0.75}}, {}),
    # 4. smoothing: only no-code lever is the _ema strategy (EMA5 on BOTH entry & exit z)
    "rq_ema5": ({}, {"strategy": "regression_dual_ml_recombine_decoupled_ema"}),
    # 5. signal_threshold (sell z(exit) band; champion 2.0)
    "rq_sz18": ({}, {"signal_threshold": 1.8}),
    "rq_sz22": ({}, {"signal_threshold": 2.2}),
    # follow-up slots (filled later if warranted)
    "rq_e4csr85": ({"entry_ensemble4": {"norm": "csrank", "z_threshold": 0.85}}, {}),
    "rq_e2z06": ({"entry_ensemble2": {"z_threshold": 0.6}}, {}),
    "rq_e3z06": ({"entry_ensemble3": {"z_threshold": 0.6}}, {}),
    "rq_e4z06": ({"entry_ensemble4": {"z_threshold": 0.6}}, {}),
    "rq_sz19": ({}, {"signal_threshold": 1.9}),
    "rq_sz21": ({}, {"signal_threshold": 2.1}),
    "rq_et18": ({}, {"entry_threshold": -1.8}),
    "rq_et16": ({}, {"entry_threshold": -1.6}),
    "rq_et20": ({}, {"entry_threshold": -2.0}),
}


async def make_clone(name: str, eng_updates: dict, top: dict) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        for k, sub in eng_updates.items():
            if isinstance(sub, dict) and isinstance(eng.get(k), dict):
                eng[k] = {**eng[k], **sub}   # merge into nested dict, keep other keys
            else:
                eng[k] = sub
        t = await repo.create(
            name=name, market=base.market,
            strategy=top.get("strategy", base.strategy),
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=top.get("signal_threshold", base.signal_threshold),
            entry_threshold=top.get("entry_threshold", base.entry_threshold),
            exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"signalq recombine probe over champion 2646: eng={eng_updates} top={top}",
            hypothesis="SIGNAL-RECOMBINE knob mapping on champion (selection-wall-aware); "
                       "occupancy-bound fill expected to mute unions.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} eng_updates={eng_updates} top={top}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    eng_updates, top = PROBES[pid]
    tid = asyncio.run(make_clone(pid, eng_updates, top))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=42)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        d = float(row[0]) - 729.6
        print(f"RQ_RESULT {pid} tmpl={tid} comp={row[0]} d={d:+.1f} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]}", flush=True)
    else:
        print(f"RQ_RESULT {pid} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
