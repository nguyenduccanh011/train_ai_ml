"""SIGNALQ exit-family sweep (regime/context-conditional EXIT levers) over champion 2646.

Phase 1 (this script), seed 42, sequential, config-only clones (cache-hit backtest re-runs):
  SNR neighborhood (window 20 fixed; known: {1.0,0.27}=+2.2, {1.2,0.27}=+0.5):
    xq_snr_t08_g15 {0.8, 0.15} / xq_snr_t08_g27 {0.8, 0.27} /
    xq_snr_t10_g15 {1.0, 0.15} / xq_snr_t12_g15 {1.2, 0.15}
  Consolidation-conditional trail: xq_cons_m05 / xq_cons_m07 (mult 0.5/0.7, defaults w20/r0.02/thr2)
    NOTE: champion runs the Donchian-80 STRUCT trail (apply_overext=true, trend_only=false), which
    bypasses trail_pct when don_low is valid — cons mult may be near-inert; empirical check.
  Two-tier EXP2: xq_tier2_a12_s11 {activate 0.12, stop 0.11} (engine-comment prior, seed_t1930_tier2)
  Pop-lock: xq_pop_a05_t05 {arm 0.05, trail 0.05, w10, ext_thr 0.0} (seed_poplock_1378 prior)

Pattern = week0/run_line_ops.py. Champion 2646 is NEVER modified, only cloned.
Usage: python stock_ml/analysis/serving_blindspot/signalq/run_exit_family.py
"""
from __future__ import annotations
import asyncio, copy, json, sys, time
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
BASE_TMPL = 2646  # champion — NEVER modified, only cloned
SEED = 42
GUARD_S = 10 * 60
CONTROL = {"composite": 729.6, "total_pnl": 127.253, "pf": 5.963, "mdd": 0.17455, "trades": 1384}

LINES = [
    ("xq_snr_t08_g15", {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                        "exit_snr_min_gain": 0.15},
     "SNR runner-extension neighborhood: thr 0.8, min_gain 0.15"),
    ("xq_snr_t08_g27", {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
                        "exit_snr_min_gain": 0.27},
     "SNR runner-extension neighborhood: thr 0.8, min_gain 0.27"),
    ("xq_snr_t10_g15", {"exit_snr_extend_threshold": 1.0, "exit_snr_extend_window": 20,
                        "exit_snr_min_gain": 0.15},
     "SNR runner-extension neighborhood: thr 1.0, min_gain 0.15"),
    ("xq_snr_t12_g15", {"exit_snr_extend_threshold": 1.2, "exit_snr_extend_window": 20,
                        "exit_snr_min_gain": 0.15},
     "SNR runner-extension neighborhood: thr 1.2, min_gain 0.15"),
    ("xq_cons_m05", {"trailing_cons_tight_mult": 0.5},
     "consolidation-conditional tighter trail, mult 0.5 (defaults w20/range0.02/thr2)"),
    ("xq_cons_m07", {"trailing_cons_tight_mult": 0.7},
     "consolidation-conditional tighter trail, mult 0.7"),
    ("xq_tier2_a12_s11", {"trailing_tier2_activate_pct": 0.12, "trailing_tier2_stop_pct": 0.11},
     "EXP2 two-tier trail: tier2 arms at +12% MFE, band 11% (protect 10-27% giveback band)"),
    ("xq_pop_a05_t05", {"pop_lock_arm_pct": 0.05, "pop_lock_trail_pct": 0.05,
                        "pop_lock_ext_window": 10, "pop_lock_ext_threshold": 0.0},
     "strength-gated pop lock: weak +5% pops trail at 5%"),
]


async def make_clone(s, name: str, add: dict, desc: str) -> int:
    repo = StrategyTemplateRepository(s)
    ex = await repo.get_by_name(name)
    if ex:
        print(f"clone exists: id={ex.id} name={name}", flush=True)
        return ex.id
    base = await repo.get_by_id(BASE_TMPL)
    slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
              "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
              "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                else copy.deepcopy(sl.target_config))}
             for sl in base.component_slots]
    eng = base.engine_config
    eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
    eng.update(add)  # ADD/override ONLY the lever keys; everything else from 2646 untouched
    t = await repo.create(
        name=name, market=base.market, strategy=base.strategy,
        feature_set_id=base.feature_set_id, target_id=base.target_id,
        component_slots=copy.deepcopy(slots), direction=base.direction,
        signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
        entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
        split_config=base.split_config, engine_config=eng,
        validation_config=base.validation_config, seed=base.seed,
        description=f"SIGNALQ exit-family sweep over champion 2646: {desc}. ADD={add}",
        hypothesis="Regime/context-conditional EXIT lever family (w0_snr_10 mechanism: convert "
                   "signal-exits on runners into trail exits; 863 runners mfe>=27% gave back 100.8u).",
        universe_slug=base.universe_slug, model_mode=base.model_mode)
    await s.commit()
    print(f"created clone: id={t.id} name={name} ADD={add}", flush=True)
    return t.id


async def make_all_clones() -> dict:
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        for name, add, desc in LINES:
            ids[name] = await make_clone(s, name, add, desc)
    await async_engine.dispose()
    return ids


def read_row(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone()
    con.close()
    return r


def main():
    results = []
    ids = asyncio.run(make_all_clones())
    for name, add, desc in LINES:
        tid = ids[name]
        t0 = time.time()
        r = run_template_experiment(template_id=tid, seed=SEED)
        dt = time.time() - t0
        row = read_row(r.get("run_id"))
        comp, pnl, pf, mdd, tr = (float(row[0]), float(row[1]), float(row[2]),
                                  float(row[3]), int(row[4]))
        d = {"name": name, "template_id": tid, "seed": SEED, "composite": comp,
             "total_pnl": pnl, "pf": pf, "mdd": mdd, "trades": tr, "runtime_s": round(dt, 1)}
        results.append(d)
        print(f"RESULT {name} tmpl={tid} seed={SEED} runtime={dt:.1f}s: "
              f"comp={comp:.1f} (dC {comp-CONTROL['composite']:+.1f}) "
              f"pnl={pnl:.3f} (dC {pnl-CONTROL['total_pnl']:+.3f}) "
              f"pf={pf:.3f} (dC {pf-CONTROL['pf']:+.3f}) "
              f"mdd={mdd:.5f} (dC {mdd-CONTROL['mdd']:+.5f}) "
              f"tr={tr} (dC {tr-CONTROL['trades']:+d})", flush=True)
        if dt > GUARD_S:
            print(f"WARNING runtime {dt:.0f}s exceeded kill-guard {GUARD_S}s", flush=True)
    out = Path(__file__).with_name("run_exit_family_results.json")
    out.write_text(json.dumps({"control_seed42": CONTROL, "results": results}, indent=2))
    print(f"wrote {out}")
    print("RUN_EXIT_FAMILY_DONE")


if __name__ == "__main__":
    main()
