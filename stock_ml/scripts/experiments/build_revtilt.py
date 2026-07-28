"""Rev-tilt: monetize the regime-robust REVERSAL head (score2 = entry_ensemble) found in the deep
internal analysis. score2 separates dead-year winners and is POSITIVE in dead years where the primary
momentum score flips negative, but the ensemble under-weights it (entry2_z_threshold=0.9 = few
reversal entries fire). Heads UNION -> LOWERING entry_ensemble.z_threshold ADDS reversal entries
(volume UP, no velocity loss, unlike every filter tested). Also test lowering score5 (forward_penalized,
also robust-ish). Clone NAV champion xh_skip300w (3064); seed 42; NAV-validate via --run-like rv_.

Usage: venv/Scripts/python.exe stock_ml/scripts/experiments/build_revtilt.py
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 3064  # xh_skip300w (NAV champion)
SEEDS = [42]
# (name, {ensemble_key: new_z_threshold}) — lower = MORE entries from that head (union)
VARIANTS = [
    ("rv_z07", {"entry_ensemble": 0.7}),
    ("rv_z05", {"entry_ensemble": 0.5}),
    ("rv_z03", {"entry_ensemble": 0.3}),
    ("rv_z05_s5z05", {"entry_ensemble": 0.5, "entry_ensemble4": 0.5}),
    ("rv_s5z04", {"entry_ensemble4": 0.4}),
]


async def make_all() -> dict:
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        base_eng = base.engine_config
        base_eng = json.loads(base_eng) if isinstance(base_eng, str) else dict(base_eng)
        for new_name, ov in VARIANTS:
            ex = await repo.get_by_name(new_name)
            if ex:
                print(f"clone exists: id={ex.id} name={new_name}"); ids[new_name] = ex.id; continue
            eng = copy.deepcopy(base_eng)
            for ekey, zt in ov.items():
                d = dict(eng.get(ekey) or {})
                d["z_threshold"] = zt
                eng[ekey] = d
            t = await repo.create(
                name=new_name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"xh_skip300w (3064) + rev-tilt {ov} (lower reversal/fwd head z_threshold "
                            "= MORE reversal entries via union; monetize regime-robust score2, no velocity loss).",
                hypothesis="score2 (reversal head) is regime-robust & dead-year-positive but underweighted "
                           "(z0.9); union means lowering its threshold adds robust reversal entries.",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"created id={t.id} name={new_name} ov={ov}")
            ids[new_name] = t.id
    return ids


def read(run_id):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,avg_pnl,trades,avg_hold FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    ids = asyncio.run(make_all())
    asyncio.run(async_engine.dispose())
    for name, tid in ids.items():
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id"); row = read(rid)
            if row:
                print(f"  {name}(t{tid}) seed={sd}: comp={row[0]} pnl={row[1]:.1f} avg={row[2]:.4f} "
                      f"tr={row[3]} hold={row[4]:.1f} run_id={rid}", flush=True)
            else:
                print(f"  {name} seed={sd}: NO ROW run_id={rid}", flush=True)
    print("BUILD_REVTILT_DONE")


if __name__ == "__main__":
    main()
