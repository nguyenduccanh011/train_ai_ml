"""NEW-EXIT-LABEL probe: clone champion 2646, swap ONLY the exit slot's target_config
(feature set exit_vol_downpress and everything else held fixed), run seed 42.
Usage: python stock_ml/analysis/serving_blindspot/signalq/xl_probe.py <probe_id>

Rationale (registry + DB archaeology, 2026-07-09): velocity_exit_regression is the basin
optimum of 558 templates -> only a DIFFERENT label type has room. Candidates chosen for
mechanism vs velocity's failure modes (churn/sells-healthy-pauses + runner giveback):
  - swing_value(sell, alpha!=1): velocity trims the upside WINDOW (U=8); alpha scales the
    upside WEIGHT over the full 20 bars -> alpha>1 refuses to sell while ANY upside remains.
  - early_wave_exit (0 slots ever): pure drawdown-EVENT probability, magnitude/vol-free;
    needs sell_high=True (legacy -1/0 is sign-inverted for the z(exit)>thr sell band).
  - downleg_depth peak_decay>0 on the modern base: event-anchored top-rollover, silent
    mid-trend -> cannot sell a healthy pause (never run with decay on any >=700 base).
  - trend_scanning windows [10,20,40]: longer-window t-stat (prior [5,10,20] point 711.5).
In decoupled, exit_threshold is INERT; the sell bar is z(exit)>signal_threshold (2.0).
Follow-up threshold points move signal_threshold, not exit_threshold.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2646  # champion (composite seed-42 729.6) — read-only

# probe_id -> (top-level overrides, exit-slot target_config)
PROBES = {
    "xl_swal2": ({}, {"type": "swing_value_regression", "horizon": 20, "direction": "sell",
                      "alpha": 2.0, "vol_normalize": True, "vol_window": 40}),
    "xl_swal06": ({}, {"type": "swing_value_regression", "horizon": 20, "direction": "sell",
                       "alpha": 0.6, "vol_normalize": True, "vol_window": 40}),
    "xl_ewx05": ({}, {"type": "early_wave_exit", "forward_window": 21,
                      "loss_threshold": 0.05, "sell_high": True}),
    "xl_dld3": ({}, {"type": "downleg_depth_regression", "pct": 0.06, "max_span": 40,
                     "min_leg_bars": 0, "peak_decay": 3.0}),
    "xl_ts40": ({}, {"type": "trend_scanning_exit", "windows": [10, 20, 40]}),
    # follow-up slots (threshold points / param second shots)
    "xl_swal2_st24": ({"signal_threshold": 2.4},
                      {"type": "swing_value_regression", "horizon": 20, "direction": "sell",
                       "alpha": 2.0, "vol_normalize": True, "vol_window": 40}),
    "xl_swal2_st16": ({"signal_threshold": 1.6},
                      {"type": "swing_value_regression", "horizon": 20, "direction": "sell",
                       "alpha": 2.0, "vol_normalize": True, "vol_window": 40}),
    "xl_swal15": ({}, {"type": "swing_value_regression", "horizon": 20, "direction": "sell",
                       "alpha": 1.5, "vol_normalize": True, "vol_window": 40}),
    "xl_swal3": ({}, {"type": "swing_value_regression", "horizon": 20, "direction": "sell",
                      "alpha": 3.0, "vol_normalize": True, "vol_window": 40}),
    "xl_ewx08": ({}, {"type": "early_wave_exit", "forward_window": 21,
                      "loss_threshold": 0.08, "sell_high": True}),
    "xl_ewx05_st24": ({"signal_threshold": 2.4},
                      {"type": "early_wave_exit", "forward_window": 21,
                       "loss_threshold": 0.05, "sell_high": True}),
    "xl_dld3_st16": ({"signal_threshold": 1.6},
                     {"type": "downleg_depth_regression", "pct": 0.06, "max_span": 40,
                      "min_leg_bars": 0, "peak_decay": 3.0}),
    "xl_ts40_st24": ({"signal_threshold": 2.4},
                     {"type": "trend_scanning_exit", "windows": [10, 20, 40]}),
}


async def make_clone(name: str, top: dict, exit_tc: dict) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = json.loads(sl.target_config) if isinstance(sl.target_config, str) else copy.deepcopy(sl.target_config)
            if sl.slot_type == "exit":
                tc = copy.deepcopy(exit_tc)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=top.get("signal_threshold", base.signal_threshold),
            entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"new-exit-label probe over champion 2646: exit slot target -> {exit_tc}; top={top}",
            hypothesis="Exit LABEL family is the biggest historical lever; velocity basin is "
                       "exhausted -> try mechanistically different labels (upside-weight asymmetry, "
                       "drawdown-event probability, pivot-anchored rollover, long-window trend t-stat).",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        # verify the exit slot target actually changed
        chk = await repo.get_by_id(t.id)
        for sl in chk.component_slots:
            if sl.slot_type == "exit":
                tcv = json.loads(sl.target_config) if isinstance(sl.target_config, str) else sl.target_config
                assert tcv.get("type") == exit_tc["type"], f"exit slot swap failed: {tcv}"
                print(f"created clone id={t.id} name={name} exit_target={tcv} top={top}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    top, exit_tc = PROBES[pid]
    tid = asyncio.run(make_clone(pid, top, exit_tc))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=42)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        d = float(row[0]) - 729.6
        print(f"XL_RESULT {pid} tmpl={tid} comp={row[0]} d={d:+.1f} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]} run_id={r.get('run_id')}", flush=True)
    else:
        print(f"XL_RESULT {pid} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
