"""SMAC v5 — REGIME-AWARE ENTER oracle (root-cause fix at the LABEL, pure-ML, no rule).

Root-cause forensic of the recurring bad trades: the model re-enters chronic decliners
many times (DIG 22 tr, AAV 35 tr) and the catastrophic losers (<-30%, 65% in 2022, held
68d) are bottoms entered while the stock had FALLEN >10% over the prior 120d (7% catastr
rate vs 1.9% sideways). The ENTER oracle was REGIME-BLIND — it labels every profitable
zigzag bottom ENTER, incl. dips inside a sustained downtrend (hindsight always finds a
>=10% bounce), so the model learns to buy knives. v3 (adding entry FEATURES) was flat
because the TARGET still rewarded entering knives — the fix must be at the LABEL.

v5 gates the ENTER label on the trailing-120d return (causal): only ENTER a bottom whose
ret_120d > threshold (drops the downtrend knife cohort, keeps the profitable momentum/
healthy-dip entries). Same v1 exit policy; only the entry label changes.

Usage: python stock_ml/scripts/build_smac_v5_regime.py
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
SWEEP = {  # name -> entry_min_ret_120
    "n2_smac_v5_reg10": -0.10,  # chosen
    "n2_smac_v5_reg20": -0.20,  # looser sensitivity check
}


async def make_template(new_name: str, thr: float) -> int:
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
                tc["entry_min_ret_120"] = thr
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": sl.feature_set_name,
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
            description=f"SMAC v5: regime-aware ENTER oracle (entry_min_ret_120={thr}) — only "
            "ENTER bottoms whose trailing-120d return exceeds the threshold, "
            "dropping the downtrend knife-catch cohort. Same v1 exit. Pure-ML, no rule.",
            hypothesis="Recurring bad trades = bottoms bought in a 120d downtrend (7% catastrophic). "
            "Gating the ENTER LABEL on trailing trend removes them from training so the "
            "model stops taking knives, keeping the net-positive momentum/healthy-dip entries.",
            universe_slug=base.universe_slug,
            model_mode="ml_only",
        )
        await s.commit()
        print(f"created template id={t.id} name={new_name} entry_min_ret_120={thr}")
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
    for new_name, thr in SWEEP.items():
        tid = asyncio.run(make_template(new_name, thr))
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
            print(f"== {new_name} (ret120>{thr}): MEAN={mean:.1f} std={std:.1f} seeds={seeds}\n")
    print(
        "== vs SMAC v1 (no regime gate) MEAN=-201.0 mdd0.518 WR0.62 hold28 tr1259 ; baselines 197.4/397.8 ; champ 704"
    )
    print("BUILD_SMAC_V5_DONE")


if __name__ == "__main__":
    main()
