"""ENTRY feature swap on champion 2429 (user proposal: money-flow / price-vs-volume-zone / crossovers).
Clone 2429, swap ONLY the ENTRY slot feature_set, retrain. The volume-at-price (dist_vwap, IC_pnl
+0.195) + net-accumulation (net_acc_vol_30) features exist but predate the 2417/2429 champion line
(untested fresh here); entry_recov_flowcross adds the new crossover features (ma/px/vol/macd cross).

Per-seed 2429 baseline: {42:703.5, 7:698.8, 99:698.5, 555:704.0}.
Usage: python stock_ml/scripts/retrain_2429_entryfeat.py <entry_feat> [seed ...]   default seeds [42]
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
BASE_TMPL = int(
    os.environ.get("BASE_TMPL", "2429")
)  # set BASE_TMPL=2355 for the nopullback sandbox
BASE42 = float(os.environ.get("BASE42", "703.5"))  # the base model's seed-42 composite (for Δ)
ENTRY_FEAT = (
    sys.argv[1] if len(sys.argv) > 1 else "entry_recov_flowcross"
)  # "-" = keep base feature
ENTRY_TARGET = os.environ.get("ENTRY_TARGET", "-")  # "-" = keep base entry target
SEEDS = [int(x) for x in sys.argv[2:]] or [42]
# knife-averse / drawdown-aware entry targets (the nopullback MDD lever, ML not hard-rule)
_ENTRY_TARGETS = {
    "tb_sl05": {
        "type": "triple_barrier",
        "horizon": 30,
        "pt": 0.15,
        "sl": 0.05,
        "direction": "long",
    },
    "tb_sl06": {
        "type": "triple_barrier",
        "horizon": 30,
        "pt": 0.15,
        "sl": 0.06,
        "direction": "long",
    },
    "tb_pt10_sl05": {
        "type": "triple_barrier",
        "horizon": 30,
        "pt": 0.10,
        "sl": 0.05,
        "direction": "long",
    },
    "fretpen": {"type": "forward_return_penalized_regression", "horizon": 20, "penalty": 1.0},
    "bottomstruct": {"type": "bottom_structure_entry_regression", "horizon": 8, "park_window": 20},
    "contrecov": {"type": "continuation_recov_entry_regression", "horizon": 10},
}
_ftag = "" if ENTRY_FEAT == "-" else f"_{ENTRY_FEAT}"
_ttag = "" if ENTRY_TARGET == "-" else f"_{ENTRY_TARGET}"
NEW_NAME = f"n2_{BASE_TMPL}_ef{_ftag}{_ttag}"
# ⚠️ SCALE BUG FIX (2026-06-20): this script reads the LEADERBOARD composite (run_template_experiment
# + read_composite), so the baseline MUST also be the LEADERBOARD composite, NOT the algo_ab
# score_summary value. 2429 LEADERBOARD = ~719 (seed42 719.1, seed555 719.6); the old {42:703.5}
# values were algo_ab (~+16 lower) → every 2429-base delta was inflated by ~+16 (a false "+13.6"
# dynamics win was actually −5.9). Pass BASE42 = the base template's LEADERBOARD composite.
BASE_PERSEED = {42: 719.1, 7: 714.5, 99: 714.5, 555: 719.6} if BASE_TMPL == 2429 else {42: BASE42}


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
            if sl.slot_type == "entry" and ENTRY_TARGET in _ENTRY_TARGETS:
                tc = copy.deepcopy(_ENTRY_TARGETS[ENTRY_TARGET])  # override entry target
            feat = (
                sl.feature_set_name
                if (sl.slot_type != "entry" or ENTRY_FEAT == "-")
                else ENTRY_FEAT
            )
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
        eng = json.loads(eng) if isinstance(eng, str) else eng
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
            description=f"2429 champion + entry feature swap -> {ENTRY_FEAT} (money-flow/volume-zone/crossovers).",
            hypothesis="volume-at-price + net-accumulation + crossover state give the entry head WHERE money "
            "traded and which side of key crosses — info the point-features miss.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} entry_feat={ENTRY_FEAT}")
        return t.id


def read_composite(run_id: str):
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
        row = read_composite(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        base = BASE_PERSEED.get(sd)
        d = f"{comp - base:+.1f}" if (comp is not None and base) else "?"
        seeds[sd] = comp
        print(
            f"  {NEW_NAME} seed={sd}: comp={comp} (vs 2429 {base} Δ{d}) pnl={row[1]:.1f} "
            f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    comps = [v for v in seeds.values() if v is not None]
    if comps:
        mean = statistics.mean(comps)
        bmean = statistics.mean([BASE_PERSEED[s] for s in seeds if s in BASE_PERSEED])
        print(
            f"\n== {NEW_NAME}: MEAN={mean:.1f} seeds={seeds}  vs 2429 baseline MEAN={bmean:.1f} (Δ={mean - bmean:+.1f})"
        )
    print("RETRAIN_ENTRYFEAT_DONE")


if __name__ == "__main__":
    main()
