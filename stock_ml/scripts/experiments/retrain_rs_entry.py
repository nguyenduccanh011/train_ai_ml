"""Add RS-vs-MARKET (vs VNINDEX) features to the MAIN entry head on champion 2632 + retrain. RS-vs-market
is the STRONGEST per-trade separator found (corr +0.16; strong-RS dips win 72%% vs weak 52%%) but it is a
SELECTION signal (all-cohorts-net-positive -> gate walled) and the wrong type for the conv-FILL modulator
(flat). The realizable path for a strong selection signal = let the ML ENTRY HEAD use it as a feature
(new engine.entry_xsec_features). Tests whether the strongest + most-orthogonal (market-relative) signal
beats the masking wall on the saturated entry head.
Usage: python stock_ml/scripts/retrain_rs_entry.py [seed ...]   default 42
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
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2632"))
RS = os.environ.get("RS_FEATS", "rs_vsma,rs_sl20,rs_sl5,rs_nh").split(",")
SEEDS = [int(x) for x in sys.argv[1:]] or [42]
NEW_NAME = os.environ.get("NEW_NAME", "n2_2632_rsentry")
# 2632 leaderboard per-seed (champion, for Δ)
BASE_PERSEED = {42: 726.4, 7: 726.8, 99: 722.9, 555: 724.6}


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
        eng[os.environ.get("SLOT", "entry") + "_xsec_features"] = RS
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
            description=f"2632 + RS-vs-market entry features {RS} on the main entry head (strongest +0.16 "
            "separator; leader-dips bounce, laggard-dips knife).",
            hypothesis="the strongest + most-orthogonal (market-relative) signal can beat the entry masking "
            "wall where weaker per-symbol features could not.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} rs={RS}")
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
        b = BASE_PERSEED.get(sd)
        d = f"{comps[sd] - b:+.1f}" if (comps[sd] and b) else "?"
        print(
            f"  {NEW_NAME} seed={sd}: comp={comps[sd]} (vs 2632 {b} Δ{d}) pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    cv = [v for v in comps.values() if v is not None]
    if cv and len(cv) > 1:
        bm = statistics.mean([BASE_PERSEED[s] for s in comps if s in BASE_PERSEED])
        print(
            f"\n== {NEW_NAME} (tmpl {new_id}): MEAN={statistics.mean(cv):.1f} vs 2632 {bm:.1f} (Δ={statistics.mean(cv) - bm:+.1f})"
        )
    print("RETRAIN_RSENTRY_DONE")


if __name__ == "__main__":
    main()
