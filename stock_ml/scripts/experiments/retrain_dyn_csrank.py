"""RAW UN-MASK test (user 2026-06-20): the indicator-DYNAMICS (MA/RSI/%R/MACD-hist velocity +
expansion + accel) are masked as shared per-symbol-z entry features (−5.9). This adds them as a
SEPARATE ensemble head (entry_ensemble3 -> score4) trained on the PURE dynamics set, unioned via
norm='csrank' (CROSS-SECTIONAL rank). The thesis: per-symbol z FLATTENS the expansion MAGNITUDE
(strong expansion z'd vs the symbol's own history looks like a weak one); csrank preserves "which
symbol is accelerating/expanding HARDEST right now" cross-sectionally — the raw, un-masked form.

SAME-SCALE: reads the LEADERBOARD composite; 2429 LEADERBOARD baseline = {42:719.1, 555:719.6}
(NOT the algo_ab 703.5 — see the scale-bug note in retrain_2429_entryfeat.py).

Usage: python stock_ml/scripts/retrain_dyn_csrank.py <csrank_pct> [seed ...]   default pct .90, seeds [42]
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
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2429"))
DYN_FEAT = os.environ.get("DYN_FEAT", "entry_dyn_pure")
PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
SEEDS = [int(x) for x in sys.argv[2:]] or [42]
BASE_PERSEED = (
    {42: 719.1, 7: 714.5, 99: 714.5, 555: 719.6}
    if BASE_TMPL == 2429
    else {s: float(os.environ.get("BASE42", "0")) for s in SEEDS}
)
_ftag = "" if DYN_FEAT == "entry_dyn_pure" else f"_{DYN_FEAT.replace('entry_', '')}"
NEW_NAME = f"n2_{BASE_TMPL}_dyncsr{int(PCT * 100)}{_ftag}"
# the dynamics head's target: continuation (does the current accel/expansion predict forward upside)
DYN_TARGET = {
    "type": "continuation_entry_regression",
    "horizon": 10,
    "penalty": 0.5,
    "trend_window": 50,
}


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
        # add the dynamics csrank head as the 4th entry head (entry_ensemble3 -> score4)
        eng["entry_ensemble3"] = {
            "target": copy.deepcopy(DYN_TARGET),
            "features": DYN_FEAT,
            "norm": "csrank",
            "z_threshold": PCT,
        }
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
            description=f"{BASE_TMPL} + PURE-dynamics csrank ensemble head (pct {PCT}). RAW un-mask: "
            "cross-sectional rank of MA/RSI/%R/MACD-hist velocity+expansion (per-symbol z masks it).",
            hypothesis="the dynamics signal is real but per-symbol z flattens its magnitude; a csrank "
            "dynamics head fires on the cross-sectionally strongest accel/expansion = winners the "
            "per-symbol-z champion misses.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} dyn_feat={DYN_FEAT} pct={PCT}")
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
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        base = BASE_PERSEED.get(sd)
        d = f"{comp - base:+.1f}" if (comp is not None and base) else "?"
        seeds[sd] = comp
        print(
            f"  {NEW_NAME} seed={sd}: comp={comp} (vs {BASE_TMPL} {base} Δ{d}) pnl={row[1]:.1f} "
            f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    comps = [v for v in seeds.values() if v is not None]
    if comps and len(comps) > 1:
        mean = statistics.mean(comps)
        bmean = statistics.mean([BASE_PERSEED[s] for s in seeds if s in BASE_PERSEED])
        print(
            f"\n== {NEW_NAME}: MEAN={mean:.1f} seeds={seeds} vs {BASE_TMPL} MEAN={bmean:.1f} (Δ={mean - bmean:+.1f})"
        )
    print("RETRAIN_DYNCSR_DONE")


if __name__ == "__main__":
    main()
