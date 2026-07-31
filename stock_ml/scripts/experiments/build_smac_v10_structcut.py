"""SMAC v10 — CONFIRMED structure-break EXIT label (exit via LABEL, not features).

v9 showed structure FEATURES alone are flat (the peak-only exit label ignores them). v10
puts the structure signal into the LABEL: on a down-leg, CUT where close pierces BOTH the
trailing 20-bar support AND MA50 (confluence — a confirmed trend break a TA acts on, not a
shallow pullback). Far rarer/more precise (6.6% of bars) than v4's fixed-% CUT (20-34%,
which over-churned). Paired with the entry_struct feature set so the model can PREDICT the
structure-CUT. Built on v8 (regime ENTER + confirm06, the current best -33.0).

If this beats v8, the exit axis is finally moving (and the position-state engine becomes
worth building); if flat/worse, the exit-realizability wall holds and entry is the axis.

Usage: python stock_ml/scripts/build_smac_v10_structcut.py
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
BASE_TMPL = 2459
SEEDS = [42, 7, 99]
ENTRY_FS = "entry_struct"
SWEEP = {  # name -> cut_struct_ma
    "n2_smac_v10_scut_ma50": 50,
    "n2_smac_v10_scut_ma20": 20,
}
BASE_TARGET = {
    "type": "action_oracle",
    "pct": 0.10,
    "min_fwd_leg": 0.10,
    "min_leg_bars": 3,
    "entry_min_ret_120": -0.10,
    "entry_confirm_pct": 0.06,
    "cut_struct_supp": 20,
}


async def make_template(new_name: str, cut_ma: int) -> int:
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
            if sl.slot_type == "entry":
                tc = dict(BASE_TARGET)
                tc["cut_struct_ma"] = cut_ma
                fs = ENTRY_FS
            else:
                tc = copy.deepcopy(sl.target_config)
                fs = sl.feature_set_name
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": fs,
                    "target_config": tc,
                }
            )
        t = await repo.create(
            name=new_name,
            market=base.market,
            strategy=base.strategy,
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
            description=f"SMAC v10: v8 + confirmed structure-break CUT (close<support20 AND "
            f"close<MA{cut_ma}) + entry_struct features. Exit signal via LABEL.",
            hypothesis="Structure features were flat (v9) because the label ignored them. A "
            "confirmed structure-break CUT label (support+MA confluence) is precise "
            "enough to cut breakdowns without the v2/v4 whipsaw, and the struct features "
            "let the model predict it.",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={new_name} cut_struct_ma={cut_ma}")
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
    for new_name, cut_ma in SWEEP.items():
        tid = asyncio.run(make_template(new_name, cut_ma))
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
                print(f"  {new_name} seed={sd}: NO RESULT", flush=True)
        comps = [v for v in seeds.values() if v is not None]
        if comps:
            mean = statistics.mean(comps)
            std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
            print(
                f"== {new_name} (cut_struct_ma={cut_ma}): MEAN={mean:.1f} std={std:.1f} seeds={seeds}\n"
            )
    print(
        "== vs v8 -33.0 (pnl+22 pf1.58 mdd0.337 WR0.72) / v9 struct-feats -34.9 ; baselines 197.4/397.8"
    )
    print("BUILD_SMAC_V10_DONE")


if __name__ == "__main__":
    main()
