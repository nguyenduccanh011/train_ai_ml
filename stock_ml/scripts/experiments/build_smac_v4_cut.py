"""SMAC v4 — 5-class oracle with a calibrated CUT (loss-cut) action, sweep X = 6/8/12%.

Arc so far: v1 -201 (holds losers 45d, no downside exit), v2 -303 (blanket exit-on-OUT
over-churns winners), v3 -199 (entry features flat — a knife isn't separable from a good
dip at entry time). So the lever is a CALIBRATED loss-cut AFTER entry.

CUT label (action_oracle, cut_drawdown=X): on each peak->next-bottom DOWN-LEG, the bars
that have fallen >= X below the prior peak. A knife the model wrongly bought looks like
this falling-off-a-high pattern -> the model predicts CUT -> the SMAC branch sells (CUT is
an exit, same as EXIT). Unlike v2's blanket OUT, CUT fires only on real declines (not on
minor pullbacks inside an up-leg). Sweep X tells us the cut tightness vs let-it-run trade.

Each variant clones v1 (tmpl 2459), only the entry target's cut_drawdown changes, multi-seed.

Usage: python stock_ml/scripts/build_smac_v4_cut.py
"""
from __future__ import annotations

import asyncio
import copy
import statistics
import sys
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
BASE_TMPL = 2459  # n2_smac_action_v1
SEEDS = [42, 7, 99]
SWEEP = {  # name -> cut_drawdown
    "n2_smac_v4_cut06": 0.06,
    "n2_smac_v4_cut08": 0.08,
    "n2_smac_v4_cut12": 0.12,
}


async def make_template(new_name: str, cut: float) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(new_name)
        if ex:
            print(f"template exists: id={ex.id} ({new_name})")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = copy.deepcopy(sl.target_config)
            if sl.slot_type == "entry":
                tc = dict(tc)
                tc["cut_drawdown"] = cut
            slots.append({
                "slot_type": sl.slot_type,
                "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id,
                "feature_set_name": sl.feature_set_name,
                "target_config": tc,
            })
        t = await repo.create(
            name=new_name,
            market=base.market,
            strategy=base.strategy,  # single_ml_action_classifier (CUT sold via branch)
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=slots,
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=None,
            exit_threshold=None,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config,
            seed=42,
            description=f"SMAC v4: 5-class oracle with CUT loss-cut (cut_drawdown={cut}). CUT "
                        "labels the down-leg bars >= X below the prior peak; the model sells "
                        "there (CUT=exit). Pure-ML calibrated loss-cut, no stop-loss rule.",
            hypothesis="v1 holds losers 45d because EXIT=upside-peak only; v2's blanket OUT "
                       "over-churns. A CUT class on real declines teaches a calibrated exit that "
                       "cuts knives without clipping healthy up-leg pullbacks.",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={new_name} cut_drawdown={cut}")
        return t.id


def read(run_id):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
        "FROM leaderboard_runs WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    for new_name, cut in SWEEP.items():
        tid = asyncio.run(make_template(new_name, cut))
        asyncio.run(async_engine.dispose())
        seeds = {}
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            row = read(r.get("run_id")) if r.get("run_id") else None
            comp = float(row[0]) if row and row[0] is not None else None
            seeds[sd] = comp
            if row:
                print(
                    f"  {new_name} seed={sd}: comp={comp} pnl={row[1]:.1f} pf={row[2]:.2f} "
                    f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}",
                    flush=True,
                )
            else:
                print(f"  {new_name} seed={sd}: NO RESULT (run failed?)", flush=True)
        comps = [v for v in seeds.values() if v is not None]
        if comps:
            mean = statistics.mean(comps)
            std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
            print(f"== {new_name} (cut={cut}): MEAN={mean:.1f} std={std:.1f} seeds={seeds}\n")
    print("== vs SMAC v1 (no cut) MEAN=-201.0 mdd0.518 WR0.62 hold28 ; baselines 197.4/397.8 ; champ 704")
    print("BUILD_SMAC_V4_DONE")


if __name__ == "__main__":
    main()
