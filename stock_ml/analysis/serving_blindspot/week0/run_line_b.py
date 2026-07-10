"""Week-0 LINE B (blindspot report, priority #1): pyramid sizing screen over champion 2646.

Clones champion 2646 (n2_2643_wavestruct_la05_lamp02) six times, adding ONLY
pyramid_add_units x pyramid_add_min_ret (pyramid_add_bars stays default 3), and screens
each at seed 42. Config-only -> reuses cached predictions (engine.py:294-304 semantics:
add <units> extra units at bar 3 post-fill if ret-since-fill >= min_ret).

NOTE: with sizing, pnl_pct semantics change (base+add units) -> composite comparability
is imperfect by design; record composite AND total_pnl/mdd_per_symbol.

Pattern = stock_ml/scripts/deploy_wavestruct.py. Idempotent on clone names.
Usage: python stock_ml/analysis/serving_blindspot/week0/run_line_b.py
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
BASE_TMPL = 2646  # CHAMPION n2_2643_wavestruct_la05_lamp02 — never modified, only cloned
SEED = 42

# (name, pyramid_add_units, pyramid_add_min_ret) — pyramid_add_bars=3 default untouched
GRID = [
    ("w0_pyr_u05_r02", 0.5, 0.02),
    ("w0_pyr_u05_r03", 0.5, 0.03),
    ("w0_pyr_u05_r04", 0.5, 0.04),
    ("w0_pyr_u10_r02", 1.0, 0.02),
    ("w0_pyr_u10_r03", 1.0, 0.03),
    ("w0_pyr_u10_r04", 1.0, 0.04),
]


async def make_clones() -> dict[str, int]:
    """Clone base 2646 once per grid point; ADD only the 2 pyramid keys. Idempotent."""
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    ids: dict[str, int] = {}
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_TMPL)
        base_slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                       "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                       "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                         else copy.deepcopy(sl.target_config))}
                      for sl in base.component_slots]
        base_eng = base.engine_config
        base_eng = json.loads(base_eng) if isinstance(base_eng, str) else copy.deepcopy(base_eng)
        for name, units, min_ret in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"clone exists: {name} id={ex.id}", flush=True)
                ids[name] = ex.id
                continue
            eng = copy.deepcopy(base_eng)
            eng.update({"pyramid_add_units": units, "pyramid_add_min_ret": min_ret})
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(base_slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"Week-0 LINE B pyramid sizing screen over champion 2646: "
                            f"pyramid_add_units={units} pyramid_add_min_ret={min_ret} (add_bars=3 default).",
                hypothesis="Blindspot report P4-5: champion never presses winners; adding units at bar 3 "
                           "when ret-since-fill >= min_ret converts high-PF selection edge into larger PnL "
                           "at some MDD cost. Readout = PnL/MDD tradeoff (composite imperfect under sizing).",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"created clone: {name} id={t.id} (units={units} min_ret={min_ret})", flush=True)
            ids[name] = t.id
    return ids


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    ids = asyncio.run(make_clones())
    asyncio.run(async_engine.dispose())
    results = []
    for name, units, min_ret in GRID:
        tid = ids[name]
        t0 = time.time()
        r = run_template_experiment(template_id=tid, seed=SEED)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        rec = {"name": name, "template_id": tid, "seed": SEED, "runtime_s": round(dt, 1),
               "composite": float(row[0]) if row[0] is not None else None,
               "total_pnl": float(row[1]) if row[1] is not None else None,
               "pf": float(row[2]) if row[2] is not None else None,
               "mdd": float(row[3]) if row[3] is not None else None,
               "trades": int(row[4]) if row[4] is not None else None}
        results.append(rec)
        print("ROW " + json.dumps(rec), flush=True)
    print("LINE_B_DONE")


if __name__ == "__main__":
    main()
