"""Retrain the CONTINUATION head (score3 = entry_ensemble2) on a DIFFERENT feature set, on the champion
line (2515 + the vol-adaptive selective exit-hold). Thesis: the momentum-DYNAMICS features (RSI/%R/
MACD-hist velocity + EMA-ribbon expansion) were MASKED as entry-SELECTION features, but they are
naturally CONTINUATION predictors — and score3 is only used as the hold's GATE (will-this-winner-keep-
running), so a dynamics-trained score3 may be a BETTER gate (hold the right names, drop the roll-overs)
-> more captured run. Clone 2515, set entry_ensemble2.features=<set>, add the hold config, retrain.
Same-scale: leaderboard composite vs 2612 (champion mean ~723.3).
Usage: python stock_ml/scripts/retrain_score3_feat.py <feature_set> [seed ...]   default entry_recov_dyn, [42]
"""

from __future__ import annotations
import asyncio, copy, json, os, statistics, sys
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
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2515"))
S3_FEAT = sys.argv[1] if len(sys.argv) > 1 else "entry_recov_dyn"
SEEDS = [int(x) for x in sys.argv[2:]] or [42]
HOLD = {
    "signal_exit_hold_ext_atr": 1.3,
    "signal_exit_hold_ma": int(os.environ.get("HOLD_MA", "20")),
    "signal_exit_hold_min_score3_z": 0.5,
    "signal_exit_hold_profit_floor": -0.03,
}
if os.environ.get("CV"):
    HOLD["signal_exit_skip_if_score3_z"] = float(os.environ["CV"])
_ma = os.environ.get("HOLD_MA", "20")
_cv = os.environ.get("CV", "")
NEW_NAME = os.environ.get(
    "NEW_NAME",
    f"n2_2515_s3feat_{S3_FEAT.replace('entry_', '')}_ma{_ma}" + (f"_cv{_cv}" if _cv else ""),
)


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [
            {
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
                "target_config": (
                    json.loads(sl.target_config)
                    if isinstance(sl.target_config, str)
                    else copy.deepcopy(sl.target_config)
                ),
            }
            for sl in base.component_slots
        ]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        # give the continuation head (entry_ensemble2 -> score3) its OWN dynamics feature set
        e2 = dict(eng.get("entry_ensemble2") or {})
        e2["features"] = S3_FEAT
        eng["entry_ensemble2"] = e2
        eng.update(HOLD)
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
            engine_config=eng,
            validation_config=base.validation_config,
            seed=base.seed,
            description=f"2515 + hold + continuation head (score3) retrained on {S3_FEAT} (momentum dynamics) "
            "— a better hold-gate (will-the-winner-run) from velocity/expansion features.",
            hypothesis="dynamics features masked as entry-selection are natural CONTINUATION predictors; "
            "score3 is only the hold's gate, so a dynamics-trained score3 holds the right names.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} score3_feat={S3_FEAT}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    comps = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        print(
            f"  {NEW_NAME} seed={sd}: comp={comps[sd]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} (champ 2612 mean ~723.3)",
            flush=True,
        )
    cv = [v for v in comps.values() if v is not None]
    if cv and len(cv) > 1:
        print(
            f"\n== {NEW_NAME} (tmpl {new_id}): leaderboard MEAN={statistics.mean(cv):.1f} seeds={comps}"
        )
    print("RETRAIN_S3FEAT_DONE")


if __name__ == "__main__":
    main()
