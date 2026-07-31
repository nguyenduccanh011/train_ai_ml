"""EXIT feature swap on the STRUCTURE-RIDE champion 2482 (user 2026-06-20: "tốc độ thay đổi hist" =
sell on momentum deceleration). On 2482 the tight %-trail is removed and the SIGNAL-EXIT head carries
~100% of PnL (avg_hold 27.8) — so an exit-feature improvement that was MASKED on 2429 (trail dominated)
has LEVERAGE here. Clone 2482, swap ONLY the EXIT slot feature_set, retrain.

Prints the RAW leaderboard composite per seed; the caller compares to the base 2482 run at the SAME
seed (same scale — do NOT compare to an algo_ab value, see the scale-bug note in retrain_2429_entryfeat).

Usage: python stock_ml/scripts/retrain_exit_feat.py <exit_feat> [seed ...]   default seeds [42]
"""

from __future__ import annotations
import asyncio, copy, json, os, sys
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
BASE_TMPL = int(
    os.environ.get("BASE_TMPL", "2482")
)  # structure-ride champion (exit carries 100% PnL)
EXIT_FEAT = sys.argv[1] if len(sys.argv) > 1 else "exit_dir_dyn"
SEEDS = [int(x) for x in sys.argv[2:]] or [42]
NEW_NAME = f"n2_{BASE_TMPL}_xf_{EXIT_FEAT}"


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            feat = EXIT_FEAT if sl.slot_type == "exit" else sl.feature_set_name
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": feat,
                    "target_config": tc,
                }
            )
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=copy.deepcopy(slots),
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config,
            engine_config=copy.deepcopy(eng),
            validation_config=base.validation_config,
            seed=base.seed,
            description=f"{BASE_TMPL} structure-ride + exit feature swap -> {EXIT_FEAT} "
            "(momentum-deceleration exit; signal-exit carries 100% PnL on the ride base).",
            hypothesis="on the structure-ride base the tight trail is gone and the signal-exit head is "
            "load-bearing, so a directional/velocity exit set (top-deceleration) that was masked "
            "on 2429 can now improve the exit timing.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} exit_feat={EXIT_FEAT}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs "
        "WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        print(
            f"  EXITFEAT {NEW_NAME} seed={sd}: comp={comp} pnl={row[1]:.1f} "
            f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    print("RETRAIN_EXITFEAT_DONE")


if __name__ == "__main__":
    main()
