"""VIRGIN-AXIS probe: clone champion 2646, override ONLY the probed keys, run seed 42.
Usage: python stock_ml/analysis/serving_blindspot/signalq/vx_probe.py <probe_id>
Axes from ARCHAEOLOGY.md: downleg_skip_bull / exit_force_gate_vn30 / entry_head_csrank_gate /
entry_z_low_threshold / trend_scanning_exit main-head swap (retrains exit head).
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
BASE_TMPL = 2646  # champion n2_2643_wavestruct_la05_lamp02 (composite seed-42 729.6) — read-only

# probe_id -> (engine-dict deep updates, top-level overrides, slot_overrides)
# slot_overrides: {slot_type: {"target_config": {...}}}
PROBES = {
    # A1 downleg_skip_bull (experiment.py:2349-2358): release the downleg12 tail-cut in a
    # confirmed SUSTAINED bull. Defaults ma_win 50 / persist 3; second run matches the
    # champion's own nonbull regime constants (35/2).
    "vx_dsb50": ({"downleg_skip_bull": {"ma_win": 50, "persist": 3}}, {}, {}),
    "vx_dsb35": ({"downleg_skip_bull": {"ma_win": 35, "persist": 2}}, {}, {}),
    # A2 exit_force_gate_vn30 (experiment.py:1681-1702, 2391-2398): real-index risk-off exit
    # tightening. Documented defaults: downtrend (close<ma20<ma50) and drawdown 10%.
    "vx_vn30dt": ({"exit_force_gate_vn30": {"index_symbol": "VN30F1M", "gate": "downleg6",
                                            "mode": "downtrend", "ma_short": 20, "ma_long": 50}}, {}, {}),
    "vx_vn30dd": ({"exit_force_gate_vn30": {"index_symbol": "VN30F1M", "gate": "downleg6",
                                            "mode": "drawdown", "dd_thresh": 0.10}}, {}, {}),
    # A3 entry_head_csrank_gate (experiment.py:2492-2498): AND-gate the MAIN momentum buy on
    # the mfe head's (ensemble3 -> column score4) same-day cross-sectional rank >= pct.
    "vx_csr75": ({"entry_head_csrank_gate": {"col": "score4", "pct": 0.75}}, {}, {}),
    "vx_csr85": ({"entry_head_csrank_gate": {"col": "score4", "pct": 0.85}}, {}, {}),
    # A4 entry_z_low_threshold (experiment.py:2239-2240): U-shaped entry — ALSO buy the
    # extreme-LOW zE tail (buy | zE < low; additive, main band zE>-1.9 untouched).
    "vx_zlo25": ({"entry_z_low_threshold": -2.5}, {}, {}),
    "vx_zlo30": ({"entry_z_low_threshold": -3.0}, {}, {}),
    # A5 trend_scanning_exit as MAIN exit head (retrains exit head, ~15 min): shape copied
    # from historical t2434 (windows [5,10,20]); keep exit_vol_downpress features + thresholds.
    "vx_tscan": ({}, {}, {"exit": {"target_config": {"type": "trend_scanning_exit",
                                                     "windows": [5, 10, 20]}}}),
    # follow-up slots (filled if any axis >= +1.5)
    "vx_dsb50p5": ({"downleg_skip_bull": {"ma_win": 50, "persist": 5}}, {}, {}),
    "vx_vn30dd15": ({"exit_force_gate_vn30": {"index_symbol": "VN30F1M", "gate": "downleg6",
                                              "mode": "drawdown", "dd_thresh": 0.15}}, {}, {}),
    "vx_vn30dtg4": ({"exit_force_gate_vn30": {"index_symbol": "VN30F1M", "gate": "downleg4",
                                              "mode": "downtrend", "ma_short": 20, "ma_long": 50}}, {}, {}),
    "vx_csr60": ({"entry_head_csrank_gate": {"col": "score4", "pct": 0.60}}, {}, {}),
    "vx_zlo28": ({"entry_z_low_threshold": -2.8}, {}, {}),
    "vx_tscan40": ({}, {}, {"exit": {"target_config": {"type": "trend_scanning_exit",
                                                       "windows": [10, 20, 40]}}}),
    "vx_dsbvn30": ({"downleg_skip_bull": {"ma_win": 50, "persist": 3},
                    "exit_force_gate_vn30": {"index_symbol": "VN30F1M", "gate": "downleg6",
                                             "mode": "downtrend", "ma_short": 20, "ma_long": 50}}, {}, {}),
    "vx_dsb60": ({"downleg_skip_bull": {"ma_win": 60, "persist": 3}}, {}, {}),
    "vx_dsb70": ({"downleg_skip_bull": {"ma_win": 70, "persist": 3}}, {}, {}),
    "vx_dsb50lb30": ({"downleg_skip_bull": {"ma_win": 50, "persist": 3},
                      "exit_force_gate_lowbreadth": {"threshold": 0.30}}, {}, {}),
}


async def make_clone(name: str, eng_updates: dict, top: dict, slot_over: dict) -> int:
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
            if sl.slot_type in slot_over and "target_config" in slot_over[sl.slot_type]:
                tc = copy.deepcopy(slot_over[sl.slot_type]["target_config"])
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        for k, sub in eng_updates.items():
            if isinstance(sub, dict) and isinstance(eng.get(k), dict):
                eng[k] = {**eng[k], **sub}
            else:
                eng[k] = sub
        t = await repo.create(
            name=name, market=base.market,
            strategy=top.get("strategy", base.strategy),
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=top.get("signal_threshold", base.signal_threshold),
            entry_threshold=top.get("entry_threshold", base.entry_threshold),
            exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"virgin-axis probe over champion 2646: eng={eng_updates} top={top} slots={slot_over}",
            hypothesis="ARCHAEOLOGY virgin axes: regime-conditioned exit release/tighten, csrank "
                       "entry selection, U-shaped entry, trend-scanning exit label.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} eng={eng_updates} top={top} slots={slot_over}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    eng_updates, top, slot_over = PROBES[pid]
    tid = asyncio.run(make_clone(pid, eng_updates, top, slot_over))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=42)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        d = float(row[0]) - 729.6
        print(f"VX_RESULT {pid} tmpl={tid} comp={row[0]} d={d:+.1f} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]} run_id={r.get('run_id')}", flush=True)
    else:
        print(f"VX_RESULT {pid} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
