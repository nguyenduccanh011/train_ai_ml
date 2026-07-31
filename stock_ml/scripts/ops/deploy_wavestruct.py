"""Deploy the WAVE-STRUCTURE exit-hold champion candidate = champion 2643 (n2_2515_volhold_rs8)
+ causal-zigzag leg-AGE hold (young leg = more run -> hold deeper) + leg-AMPLITUDE hold (strong
impulse -> hold deeper). The first multi-wave-structure INPUT into the model. Config-only over 2643
(reuses cached predictions; only the backtest re-runs). Multi-seed LEADERBOARD register.

Probe: leg-age has incremental IC -0.27 vs forward-run AFTER removing momentum (orthogonal, not
subsumed like market-trend); leg-amp adds +0.14. score_summary multi-seed: legage 0.5 = +1.12,
+legamp 0.2 STACK = +1.78 mean ALL-SEED over 2643.

Clones 2643's FULL engine_config and ADDS ONLY the 3 new keys (touches nothing else).
Usage: python stock_ml/scripts/deploy_wavestruct.py [seed ...]   default 42 7 99 555
"""

from __future__ import annotations
import asyncio, copy, json, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import os  # noqa: E402

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = int(os.environ.get("BASE_TMPL", "2643"))
ADD = {
    "signal_exit_hold_legage_scale": float(os.environ.get("LEGAGE", "0.5")),
    "signal_exit_hold_legage_pct": float(os.environ.get("LEGAGE_PCT", "0.06")),
    "signal_exit_hold_legamp_scale": float(os.environ.get("LEGAMP", "0.2")),
}
SEEDS = [int(x) for x in sys.argv[1:]] or [42, 7, 99, 555]
NEW_NAME = os.environ.get("NEW_NAME", "n2_2643_wavestruct_la05_lamp02")


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
        eng.update(ADD)  # ADD ONLY the 3 new keys; everything else from 2643 untouched
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
            description="champion 2643 + WAVE-STRUCTURE exit-hold: scale the signal-exit hold K by the "
            "causal-zigzag current-leg AGE (young leg = more run ahead -> hold deeper) and "
            "AMPLITUDE-so-far vs the stock's typical up-leg (strong impulse -> hold deeper). "
            "First explicit multi-wave-structure input; score_summary multi-seed +1.78 all-seed.",
            hypothesis="loss-streaks cluster in market-wide down-windows and SELECTION is walled; but the "
            "position-in-wave (leg age, incr IC -0.27 vs forward-run orthogonal to momentum) and "
            "impulse strength (leg amplitude, +0.14) predict remaining run and REALIZE via "
            "exit-timing — hold winners through young/strong legs, exit aged/weak ones.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME}  ADD={ADD}")
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


def champ_baseline():
    """Leaderboard composites of champion 2643 per seed for a fair comparison."""
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT run_seed, composite_score FROM leaderboard_runs WHERE run_name='n2_2515_volhold_rs8' AND superseded=false ORDER BY run_seed"
    )
    r = dict(cur.fetchall())
    con.close()
    return r


def main():
    base_lb = champ_baseline()
    print(f"champion 2643 leaderboard per-seed: {base_lb}")
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    comps = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_row(r.get("run_id"))
        comps[sd] = float(row[0]) if row and row[0] is not None else None
        b = base_lb.get(sd)
        dl = f"{comps[sd] - float(b):+.1f}" if (comps[sd] is not None and b is not None) else "n/a"
        print(
            f"  {NEW_NAME} seed={sd}: comp={comps[sd]} (champ {b}, Δ{dl}) "
            f"pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    cv = [v for v in comps.values() if v is not None]
    bv = [
        float(base_lb[sd]) for sd in comps if base_lb.get(sd) is not None and comps[sd] is not None
    ]
    if cv:
        print(
            f"\n== {NEW_NAME} (tmpl {new_id}): leaderboard MEAN={statistics.mean(cv):.2f} seeds={comps}"
        )
        if bv:
            print(
                f"== champion 2643 MEAN={statistics.mean(bv):.2f}  -> Δ MEAN={statistics.mean(cv) - statistics.mean(bv):+.2f}"
            )
    print("DEPLOY_WAVESTRUCT_DONE")


if __name__ == "__main__":
    main()
