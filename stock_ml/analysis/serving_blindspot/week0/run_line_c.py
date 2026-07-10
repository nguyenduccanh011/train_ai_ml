"""Week-0 LINE C step-0: conditional exit levers screen over champion 2646, seed 42.

Six config-only clones of 2646 (copy component_slots + engine_config, eng.update(ADD) with
ONLY the lever keys), each run once at seed 42 via run_template_experiment, row read back
from leaderboard_runs. Sequential. Deltas vs CONTROL seed-42 (comp 729.6, pnl 127.253,
pf 5.963, mdd 0.17455, trades 1384). Flags trades change >20% (non-masking levers).
Usage: python stock_ml/analysis/serving_blindspot/week0/run_line_c.py
"""
from __future__ import annotations
import asyncio, copy, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2646  # champion n2_2643_wavestruct_la05_lamp02 — NEVER modified, only cloned
SEED = 42
CONTROL = dict(composite=729.6, total_pnl=127.253, pf=5.963, mdd=0.17455, trades=1384)

LEVERS = [
    ("w0_relk_30",   {"signal_exit_protect_release_drop_k": 3.0}),
    ("w0_relk_35",   {"signal_exit_protect_release_drop_k": 3.5}),
    ("w0_mfeact_05", {"mfe_act_k": 0.5}),
    ("w0_mfeact_10", {"mfe_act_k": 1.0}),
    ("w0_snr_10",    {"exit_snr_extend_threshold": 1.0, "exit_snr_extend_window": 20,
                      "exit_snr_min_gain": 0.27}),
    ("w0_snr_12",    {"exit_snr_extend_threshold": 1.2, "exit_snr_extend_window": 20,
                      "exit_snr_min_gain": 0.27}),
]


async def make_clones() -> dict[str, int]:
    """Create (or reuse) one clone of 2646 per lever. Returns name -> template_id."""
    ids: dict[str, int] = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        base_eng = base.engine_config
        base_eng = json.loads(base_eng) if isinstance(base_eng, str) else copy.deepcopy(base_eng)
        for name, add in LEVERS:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"clone exists: {name} id={ex.id}", flush=True)
                ids[name] = ex.id
                continue
            eng = copy.deepcopy(base_eng)
            eng.update(add)  # ADD/override ONLY the lever keys; everything else from 2646
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"week0 LINE-C step-0 screen: champion 2646 + {json.dumps(add)}",
                hypothesis="BLINDSPOT week-0 LINE C: conditional exit levers (release-K, MFE "
                           "activation, SNR extend) improve exit timing without masking entries "
                           "(trades must stay within 20% of control).",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"created clone: {name} id={t.id} ADD={add}", flush=True)
            ids[name] = t.id
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
    ids = asyncio.run(make_clones())
    asyncio.run(async_engine.dispose())
    results = []
    for name, add in LEVERS:
        tid = ids[name]
        t0 = time.time()
        print(f"RUN_START {name} tmpl={tid} t={time.strftime('%H:%M:%S')}", flush=True)
        r = run_template_experiment(template_id=tid, seed=SEED)
        dt = round(time.time() - t0, 1)
        row = read_row(r.get("run_id"))
        out = dict(name=name, template_id=tid, seed=SEED, runtime_s=dt,
                   composite=float(row[0]), total_pnl=float(row[1]), pf=float(row[2]),
                   mdd=float(row[3]), trades=int(row[4]))
        out["d_comp"] = round(out["composite"] - CONTROL["composite"], 1)
        out["d_pnl"] = round(out["total_pnl"] - CONTROL["total_pnl"], 3)
        out["d_trades"] = out["trades"] - CONTROL["trades"]
        out["trades_pct"] = round(100.0 * out["d_trades"] / CONTROL["trades"], 1)
        out["TRADES_FLAG"] = abs(out["trades_pct"]) > 20.0
        results.append(out)
        print("ROW " + json.dumps(out), flush=True)
    print("SUMMARY " + json.dumps(results))
    print("LINE_C_DONE")


if __name__ == "__main__":
    main()
