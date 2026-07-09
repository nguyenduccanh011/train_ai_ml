"""PnL-FRAME model (user wants RAW PnL, accepts MDD): structure-ride + max hold-continuation
(skip the signal-exit whenever the entry head is still positive -> ride continuations through the
between-trade gaps = capture the +1929% between-chunk). seed42 = 130.5 PnL (+5.3 / +4.2% vs the
composite champion 125.2), PF 5.86 (+23%), MDD 0.227 (+0.034). Higher raw PnL + PF but higher MDD
and more seed variance -> deploy as a SEPARATE PnL-frame model alongside the composite champion 2482.
Usage: python stock_ml/scripts/build_2429_pnlmax.py
"""
from __future__ import annotations
import asyncio, copy, json, statistics, sys
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
BASE, NEW_NAME, SEEDS = 2429, "n2_2429_riskstruct80_pnlmax", [555, 42, 7, 99]
OV = {"trailing_struct_donch_win": 80, "trailing_struct_apply_overext": True,
      "signal_exit_skip_if_entry_z": 0.0}


async def make():
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"exists {ex.id}"); return ex.id
        b = await repo.get_by_id(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))} for sl in b.component_slots]
        eng = b.engine_config; eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update(OV)
        t = await repo.create(
            name=NEW_NAME, market=b.market, strategy=b.strategy, feature_set_id=b.feature_set_id,
            target_id=b.target_id, component_slots=copy.deepcopy(slots), direction=b.direction,
            signal_mode=b.signal_mode, signal_threshold=b.signal_threshold,
            entry_threshold=b.entry_threshold, exit_threshold=b.exit_threshold,
            split_config=b.split_config, engine_config=eng, validation_config=b.validation_config,
            seed=b.seed, description="PnL-FRAME: structure-ride + max hold-continuation "
            "(signal_exit_skip_if_entry_z=0) rides continuations through the between-trade gaps. "
            "~+4% RAW PnL & +23% PF vs composite champ, at +MDD (0.23) — for raw-PnL priority.",
            hypothesis="holding continuations (skip signal-exit while the entry head is positive) "
            "captures the between-trade up-chunk for higher raw PnL, accepting more drawdown.",
            universe_slug=b.universe_slug, model_mode=b.model_mode)
        await s.commit(); print(f"created {t.id}"); return t.id


def rd(rid):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades FROM leaderboard_runs WHERE run_id=%s", (rid,))
    r = cur.fetchone(); con.close(); return r


def main():
    nid = asyncio.run(make()); asyncio.run(async_engine.dispose())
    pnls = []
    for sd in SEEDS:
        r = run_template_experiment(template_id=nid, seed=sd); row = rd(r.get("run_id"))
        pnls.append(float(row[1]))
        print(f"  seed {sd}: comp={row[0]} PnL={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
    print(f"\n== {NEW_NAME} (id {nid}): PnL mean={statistics.mean(pnls):.1f} std={statistics.pstdev(pnls):.1f} "
          f"min={min(pnls):.1f} max={max(pnls):.1f}  (composite champ 2482 PnL~125.4)")
    print("BUILD_PNLMAX_DONE")


if __name__ == "__main__":
    main()
