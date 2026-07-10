"""Week-0 OPS levers screen (BLINDSPOT_REPORT Phan 4-5): clone champion 2646 and screen
three config-only ops levers at seed 42. Reuses cached predictions (backtest-only re-run).

  1. w0_cxlma20  : entry_pullback_cancel_below_ma = 20  (cancel pending limits below MA20)
  2. w0_win20    : entry_pullback_window = 20 (champion 40; ops-load lever)
  3. w0_reprem04 : reentry_max_premium_pct = 0.04 (cap re-buy-higher chasing)

Pattern copied from stock_ml/scripts/deploy_wavestruct.py. Sequential, seed 42 only.
Usage: python stock_ml/analysis/serving_blindspot/week0/run_line_ops.py
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
BASE_TMPL = 2646  # champion n2_2643_wavestruct_la05_lamp02 — NEVER modified, only cloned
SEED = 42
CONTROL = {"composite": 729.6, "total_pnl": 127.253, "pf": 5.963, "mdd": 0.17455, "trades": 1384}

LINES = [
    ("w0_cxlma20", {"entry_pullback_cancel_below_ma": 20},
     "OPS lever: cancel pending pullback limits when close drops below MA20 (knife-fills, complaint #3)"),
    ("w0_win20", {"entry_pullback_window": 20},
     "OPS lever: pending-limit window 40->20 bars (-45.6% pending-days for ~-4.6% pnl in static overlay)"),
    ("w0_reprem04", {"reentry_max_premium_pct": 0.04},
     "OPS lever: cap re-buy-higher chasing at +4% over last exit price (selection-wall risk screen)"),
]


async def make_clone(s, name: str, add: dict, desc: str) -> int:
    if True:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id} name={name}", flush=True)
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update(add)  # ADD/override ONLY the lever key; everything else from 2646 untouched
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"Week-0 blindspot screen over champion 2646: {desc}. ADD={add}",
            hypothesis="BLINDSPOT_REPORT Week-0 ops-lever screen (serving/ops quality, not alpha).",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} ADD={add}", flush=True)
        return t.id


async def make_all_clones() -> dict:
    """Create/reuse all clones in ONE event loop, then dispose the engine (a second
    asyncio.run against the module-level async_engine reuses stale-loop connections)."""
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        for name, add, desc in LINES:
            ids[name] = await make_clone(s, name, add, desc)
    await async_engine.dispose()
    return ids


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone()
    con.close()
    return r


def main():
    results = []
    ids = asyncio.run(make_all_clones())
    for name, add, desc in LINES:
        tid = ids[name]
        t0 = time.time()
        r = run_template_experiment(template_id=tid, seed=SEED)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        comp, pnl, pf, mdd, tr = (float(row[0]), float(row[1]), float(row[2]),
                                  float(row[3]), int(row[4]))
        d = {"name": name, "template_id": tid, "seed": SEED, "composite": comp,
             "total_pnl": pnl, "pf": pf, "mdd": mdd, "trades": tr, "runtime_s": round(dt, 1)}
        results.append(d)
        print(f"RESULT {name} tmpl={tid} seed={SEED} runtime={dt:.1f}s: "
              f"comp={comp:.1f} (dC {comp-CONTROL['composite']:+.1f}) "
              f"pnl={pnl:.3f} (dC {pnl-CONTROL['total_pnl']:+.3f}) "
              f"pf={pf:.3f} (dC {pf-CONTROL['pf']:+.3f}) "
              f"mdd={mdd:.5f} (dC {mdd-CONTROL['mdd']:+.5f}) "
              f"tr={tr} (dC {tr-CONTROL['trades']:+d})", flush=True)
    out = Path(__file__).with_name("run_line_ops_results.json")
    out.write_text(json.dumps({"control_seed42": CONTROL, "results": results}, indent=2))
    print(f"wrote {out}")
    print("RUN_LINE_OPS_DONE")


if __name__ == "__main__":
    main()
