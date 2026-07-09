"""LOOP iter-13 (autonomous): the last untested big lever — the main entry-head TARGET. The head's
value is via the head-BLEND (drives conv fill on predicted runners). Maybe a target tuned toward
bigger forward up-runs (longer horizon / higher profit-target) sharpens that prediction. Clone 2417,
swap ONLY the entry target_config, train, blend-test. Seed-42 screen; multi-seed the winner.
Usage: python stock_ml/scripts/screen_headblend_targets.py
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
BASE = 2417
TARGETS = {
    "tb_h40_pt20": {"type": "triple_barrier", "horizon": 40, "pt": 0.20, "sl": 0.08, "direction": "long"},
    "tb_h20_pt12": {"type": "triple_barrier", "horizon": 20, "pt": 0.12, "sl": 0.06, "direction": "long"},
    "tb_h30_pt20": {"type": "triple_barrier", "horizon": 30, "pt": 0.20, "sl": 0.10, "direction": "long"},
    "fwd_h20": {"type": "forward_return_regression", "horizon": 20},
}


async def clone_all():
    out = {}
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        b = await repo.get_by_id(BASE)
        for tag, tcfg in TARGETS.items():
            name = f"hbt_{tag}"
            ex = await repo.get_by_name(name)
            if ex:
                out[tag] = ex.id; continue
            slots = []
            for sl in b.component_slots:
                tc = (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                      else copy.deepcopy(sl.target_config))
                if sl.slot_type == "entry":
                    tc = copy.deepcopy(tcfg)
                slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                              "rule_component_id": sl.rule_component_id,
                              "feature_set_name": sl.feature_set_name, "target_config": tc})
            eng = b.engine_config; eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            t = await repo.create(
                name=name, market=b.market, strategy=b.strategy, feature_set_id=b.feature_set_id,
                target_id=b.target_id, component_slots=copy.deepcopy(slots), direction=b.direction,
                signal_mode=b.signal_mode, signal_threshold=b.signal_threshold,
                entry_threshold=b.entry_threshold, exit_threshold=b.exit_threshold,
                split_config=b.split_config, engine_config=eng, validation_config=b.validation_config,
                seed=b.seed, description=f"head-blend champ + entry target {tag} (iter-13).",
                hypothesis="a bigger-runner-tuned head target sharpens the blend's fill targeting.",
                universe_slug=b.universe_slug, model_mode=b.model_mode)
            out[tag] = t.id
        await s.commit()
    await async_engine.dispose()
    return out


def comp(rid):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,mdd_per_symbol FROM leaderboard_runs WHERE run_id=%s", (rid,))
    r = cur.fetchone(); con.close(); return r


def main():
    print("baseline 2417 (tb_h30_pt15) head-blend = 688.6 multi-seed / seed42 691.3\n")
    ids = asyncio.run(clone_all())
    for tag in TARGETS:
        r = run_template_experiment(template_id=ids[tag], seed=42); c = comp(r.get("run_id"))
        print(f"  {tag:14} seed42 comp={c[0]:.1f} mdd={c[1]:.3f}  (vs 691.3)", flush=True)
    print("TARGET_SCREEN_DONE")


if __name__ == "__main__":
    main()
