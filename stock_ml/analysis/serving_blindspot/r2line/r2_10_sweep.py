# -*- coding: utf-8 -*-
"""TUYEN R2 vong 1: sweep NAV-first cho fc_rule2 (t2835 clone cua t2005).

Base NAV K25 x14.78 / CAGR 51.21% / MaxDD -12.25% — muc tieu NAV > x15.5.
Truc MO (chua co an o NAV-frame):
  (a) snr_extend/giveback 4 key gb (rule market-level, hop le 0-ML, defer kenh signal-exit
      1.072 lenh -18.4u -> trao cho trail/overext khi regime clean-trend & trade dang thang)
  (b) luoi pullback depth x window (base 4.5%/50)
  (c) dtstop level (base -6%)
  (d) entry_gate token (base upleg_abovema20)
Deterministic (khong ML fit) -> seed 42 du cho screening.

Usage: python r2_10_sweep.py <group>   # group in {base, snr, pb, dt, gate}
Resume: variant da co trades CSV -> chi in lai ket qua leaderboard neu tim thay run.
"""
from __future__ import annotations
import asyncio, copy, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = "fc_rule2"
HERE = Path(__file__).parent

GROUPS = {
    "base": [
        ("r2_base", {}),
    ],
    "snr": [
        # gb parity: thr 0.8 / w20 / min_gain 0.27 / giveback 0.08
        ("r2_snr_gb", {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                       "exit_snr_min_gain": 0.27, "exit_snr_defer_min_giveback": 0.08}),
        # min_gain ha xuong 0.12 — winner cua rule2 nho hon gb (p95 +23.5%), 0.27 gan nhu khong bao gio bat
        ("r2_snr_g12", {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                        "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.08}),
        ("r2_snr_g12_nogb", {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                             "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.0}),
        # regime rong hon (thr 0.6)
        ("r2_snr06_g12", {"exit_snr_extend_threshold": 0.6, "exit_snr_extend_window": 20,
                          "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.08}),
    ],
    "pb": [
        ("r2_pb35_30", {"entry_pullback_pct": 0.035, "entry_pullback_window": 30}),
        ("r2_pb35_50", {"entry_pullback_pct": 0.035, "entry_pullback_window": 50}),
        ("r2_pb35_70", {"entry_pullback_pct": 0.035, "entry_pullback_window": 70}),
        ("r2_pb45_30", {"entry_pullback_pct": 0.045, "entry_pullback_window": 30}),
        ("r2_pb45_70", {"entry_pullback_pct": 0.045, "entry_pullback_window": 70}),
        ("r2_pb55_30", {"entry_pullback_pct": 0.055, "entry_pullback_window": 30}),
        ("r2_pb55_50", {"entry_pullback_pct": 0.055, "entry_pullback_window": 50}),
        ("r2_pb55_70", {"entry_pullback_pct": 0.055, "entry_pullback_window": 70}),
    ],
    "dt": [
        ("r2_dt05", {"downtrend_hard_stop_pct": -0.05}),
        ("r2_dt08", {"downtrend_hard_stop_pct": -0.08}),
    ],
    "gate": [
        ("r2_gate_up", {"entry_gate": "upleg"}),               # bo abovema20 -> nhieu entry hon (throughput)
        ("r2_gate_ama10", {"entry_gate": "upleg_abovema10"}),  # long hon
        ("r2_gate_ama50", {"entry_gate": "upleg_abovema50"}),  # chat hon (trend dai)
    ],
}


async def make_clones(variants):
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    tids = {}
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_name(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        for name, patch in variants:
            ex = await repo.get_by_name(name)
            if ex:
                tids[name] = ex.id
                print(f"exists: {name} id={ex.id}", flush=True)
                continue
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            eng = copy.deepcopy(eng); eng.update(patch)
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"R2 line round1 NAV-first sweep: {BASE} + {patch}",
                hypothesis="R2: grow NAV past x15.5 (base x14.78 beats gb_x08 x14.44); rank by NAV not composite",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            tids[name] = t.id
            print(f"created: {name} id={t.id}", flush=True)
        await s.commit()
    await async_engine.dispose()
    return tids


GROUPS["pb1"] = GROUPS["pb"][:4]
GROUPS["pb2"] = GROUPS["pb"][4:]
GROUPS["r2b"] = [
    # combo 2 truc NAV-duong doc lap
    ("r2_c1_pbsnr", {"entry_pullback_pct": 0.035, "entry_pullback_window": 50,
                     "exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                     "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.08}),
    # do ridge depth quanh 3.5 tai window 50
    ("r2_pb30_50", {"entry_pullback_pct": 0.030, "entry_pullback_window": 50}),
    ("r2_pb40_50", {"entry_pullback_pct": 0.040, "entry_pullback_window": 50}),
]
GROUPS["r2c"] = [
    ("r2_c2_pb40snr", {"entry_pullback_pct": 0.040, "entry_pullback_window": 50,
                       "exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                       "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.08}),
]


def main():
    group = sys.argv[1]
    variants = GROUPS[group]
    tids = asyncio.run(make_clones(variants))
    for name, patch in variants:
        csv_path = HERE / f"{name}_s42_trades.csv"
        if csv_path.exists():
            print(f"SKIP (csv exists): {name}", flush=True)
        else:
            r = run_template_experiment(template_id=tids[name], seed=42)
            con = psycopg2.connect(**PG); cur = con.cursor()
            cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                        "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
            row = cur.fetchone()
            print(f"R2_RESULT {name} s42 comp={row[0]:.1f} d_base={row[0]-568.3:+.1f} pnl={row[1]:.1f} "
                  f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} "
                  f"run_id={r.get('run_id')}", flush=True)
            tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                              "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                              con, params=(r.get("run_id"),))
            con.close()
            tdf.to_csv(csv_path, index=False)
            print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
        # NAV K25
        out = subprocess.run([sys.executable, str(HERE / "r2_nav.py"), str(csv_path), name, "25"],
                             capture_output=True, text=True)
        print(out.stdout.strip(), flush=True)
        if out.returncode != 0:
            print(out.stderr[-2000:], flush=True)
    print(f"R2_SWEEP_DONE group={group}")


if __name__ == "__main__":
    main()
