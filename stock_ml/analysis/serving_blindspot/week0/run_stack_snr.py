"""Week-0 follow-up TASK 1: STACK snr-extend on top of best pyramid candidate.

Clone w0_pyr_u10_r02_snr10 = base template 2669 (w0_pyr_u10_r02, champion+pyramid)
+ {exit_snr_extend_threshold: 1.0, exit_snr_extend_window: 20, exit_snr_min_gain: 0.27}.
Clones FROM 2669 so the pyramid keys carry over; VERIFIES the cloned engine_config
contains BOTH pyramid_add_units=1.0 AND the snr keys before any run. Idempotent.

Pattern = stock_ml/scripts/ops/deploy_wavestruct.py. Config-only -> cache-hit runs.
Usage: python stock_ml/analysis/serving_blindspot/week0/run_stack_snr.py [seed ...]
       (default: 42)
"""
from __future__ import annotations
import asyncio, copy, json, sys, time
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
BASE_TMPL = 2669  # w0_pyr_u10_r02 — NEVER modified, only cloned
NAME = "w0_pyr_u10_r02_snr10"
ADD = {"exit_snr_extend_threshold": 1.0, "exit_snr_extend_window": 20, "exit_snr_min_gain": 0.27}
PYR_EXPECT = {"pyramid_add_units": 1.0, "pyramid_add_min_ret": 0.02}


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NAME)
        if ex:
            tid = ex.id
            eng = ex.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else eng
            print(f"clone exists: {NAME} id={tid}", flush=True)
        else:
            base = await repo.get_by_id(BASE_TMPL)
            slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                      "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                      "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                        else copy.deepcopy(sl.target_config))}
                     for sl in base.component_slots]
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            eng.update(ADD)  # ADD only the 3 snr keys; pyramid keys already in 2669's config
            t = await repo.create(
                name=NAME, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description="Week-0 STACK: w0_pyr_u10_r02 (2669, champion+pyramid u=1.0 r=0.02) "
                            "+ exit_snr_extend_threshold=1.0 window=20 min_gain=0.27 (w0_snr_10 lever).",
                hypothesis="SNR-extend is the only positive exit lever (+2.2 standalone) and is orthogonal "
                           "to sizing; holding runners longer matters MORE once the position has pyramided.",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            tid = t.id
            print(f"created clone: {NAME} id={tid}  ADD={ADD}", flush=True)
        # VERIFY: cloned config must contain BOTH the pyramid keys AND the snr keys
        want = dict(PYR_EXPECT); want.update(ADD)
        bad = {k: (eng.get(k), v) for k, v in want.items()
               if eng.get(k) is None or abs(float(eng.get(k)) - float(v)) > 1e-9}
        if bad:
            print(f"VERIFY FAIL: {bad}", flush=True)
            raise SystemExit(f"engine_config verify failed for {NAME}: {bad}")
        print("VERIFY OK: " + json.dumps({k: eng[k] for k in want}), flush=True)
        return tid


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [42]
    tid = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    for seed in seeds:
        t0 = time.time()
        r = run_template_experiment(template_id=tid, seed=seed)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {"name": NAME, "template_id": tid, "seed": seed, "runtime_s": round(dt, 1),
               "composite": float(row[0]) if row[0] is not None else None,
               "total_pnl": float(row[1]) if row[1] is not None else None,
               "pf": float(row[2]) if row[2] is not None else None,
               "mdd": float(row[3]) if row[3] is not None else None,
               "trades": int(row[4]) if row[4] is not None else None}
        print("ROW " + json.dumps(rec), flush=True)
    print("STACK_SNR_DONE")


if __name__ == "__main__":
    main()
