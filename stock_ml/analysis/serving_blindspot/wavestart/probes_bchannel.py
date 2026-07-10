"""Wave-start B-channel probes over champion template 2646 (composite seed-42 baseline 729.6).

Driver mode (default): runs each probe as a subprocess (25-min kill guard), tees output to
logs/<probe>.log, prints each RESULT_JSON as it lands.
Worker mode (--worker NAME): clones 2646 with the probe's engine-config overrides, runs
run_template_experiment(seed=42), reads the leaderboard row, prints RESULT_JSON.

Order: P5 first (engine-knob-only, cache hit) then the head-training probes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # f:/PROJECTS/train_ai_ml
LOGS = HERE / "logs"
BASE_TMPL = 2646
BASELINE = {"comp": 729.6, "pnl": 127.25, "pf": 5.963, "mdd": 0.17455, "trades": 1384}
KILL_GUARD_S = 25 * 60
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

PROBES = [
    # P5 — engine-knob-only (cache hit, run first)
    ("p_anchor_swing", {
        "entry_pullback_structural_lookback": 10,
        "entry_pullback_structural_buffer": 0.01,
        "entry_pullback_fill_if_missed": True,
        "fill_if_missed_max_premium": 0.02,
    }),
    # P1 — 5th union head (bottom-structure), fills still via champion 4.5% pullback
    ("p_ens5_bottom", {
        "entry_ensemble5": {
            "target": {"type": "bottom_structure_entry_regression", "horizon": 8,
                       "penalty": 1.5, "dip_window": 50, "require_turn": True,
                       "park_window": 20, "park_penalty": 0.5},
            "z_threshold": 0.9, "norm": "zscore",
        },
    }),
    # P2 — score-only zigzag bottom head (no z_threshold => no union) driving fill deepen + trail
    ("p_zigzag_deepen", {
        "entry_ensemble5": {
            "target": {"type": "zigzag_pivot", "direction": "bottom", "one_sided": "pre",
                       "pct": 0.10, "tau": 5, "min_fwd_leg": 0.10},
        },
        "bot_deepen_k": 1.0, "bot_deepen_cap": 2.0, "bot_ride_k": 0.5,
    }),
    # P3 — re-param champion reversal head: concentrated wave-base dips
    ("p_rev_dw20_r15", {
        "entry_ensemble": {
            "target": {"type": "reversal_entry_regression", "horizon": 10, "penalty": 1.0,
                       "dip_window": 20, "min_fwd_rally": 0.15},
            "z_threshold": 0.9,
        },
    }),
    # P3b — same, dip_window 30 / rally 0.12
    ("p_rev_dw30_r12", {
        "entry_ensemble": {
            "target": {"type": "reversal_entry_regression", "horizon": 10, "penalty": 1.0,
                       "dip_window": 30, "min_fwd_rally": 0.12},
            "z_threshold": 0.9,
        },
    }),
    # P4 — loosen reversal z to 0.6 + post-union RS knife-gate
    ("p_rev_z06_rsgate", {
        "entry_ensemble": {
            "target": {"type": "reversal_entry_regression", "horizon": 10, "penalty": 1.0,
                       "dip_window": 50, "min_fwd_rally": 0.1},
            "z_threshold": 0.6,
        },
        "entry_rs_gate": {"feature": "rs_vsma", "threshold": 0.0},
    }),
    # F1 — isolate P5's miss-capture leg (no structural re-anchor): wave starts never reach
    # the 4.5% limit, so take the runaway at window-end close (<=2% premium) only.
    ("p_fill_if_missed", {
        "entry_pullback_fill_if_missed": True,
        "fill_if_missed_max_premium": 0.02,
    }),
    # F2 — isolate P2's exit-side trail-widen from the entry-side deepen (deepen contradicts
    # the ~2% wave-start dip anatomy; ride monetizes true-bottom entries occupancy-free).
    ("p_zz_ride_only", {
        "entry_ensemble5": {
            "target": {"type": "zigzag_pivot", "direction": "bottom", "one_sided": "pre",
                       "pct": 0.10, "tau": 5, "min_fwd_leg": 0.10},
        },
        "bot_ride_k": 0.5,
    }),
]
PROBE_MAP = dict(PROBES)


# ------------------------------------------------------------------ worker
def worker(name: str) -> None:
    sys.path.insert(0, str(REPO))
    import asyncio
    import copy

    import psycopg2
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from stock_ml.db.engine import async_engine
    from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

    overrides = PROBE_MAP[name]
    result = {"probe": name, "overrides": overrides}

    async def make_clone() -> int:
        Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with Session() as s:
            repo = StrategyTemplateRepository(s)
            ex = await repo.get_by_name(name)
            if ex:
                print(f"[{name}] clone exists: id={ex.id}", flush=True)
                return ex.id
            base = await repo.get_by_id(BASE_TMPL)
            slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                      "rule_component_id": sl.rule_component_id,
                      "feature_set_name": sl.feature_set_name,
                      "target_config": (json.loads(sl.target_config)
                                        if isinstance(sl.target_config, str)
                                        else copy.deepcopy(sl.target_config))}
                     for sl in base.component_slots]
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            eng.update(copy.deepcopy(overrides))
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"wave-start B-channel probe {name} over champion 2646",
                hypothesis="wave starts offer ~2% dips vs champion 4.5% pullback; probe which "
                           "mechanism (union head / fill deepen / re-param / anchor) adds "
                           "wave-start fills without knife quality loss",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"[{name}] created clone: id={t.id}", flush=True)
            return t.id

    try:
        tmpl_id = asyncio.run(make_clone())
        asyncio.run(async_engine.dispose())
        result["template_id"] = tmpl_id

        # patch run_backtest (imported by name into experiment.py) to verify score6 presence.
        # NOTE: run_template.py imports the pipeline as `src.pipeline.experiment` (stock_ml/ on
        # sys.path), a DIFFERENT module identity than stock_ml.src.pipeline.experiment — import
        # run_template first (it sets sys.path) and patch that identity.
        from stock_ml.scripts import run_template as _rt  # noqa: F401  (sets sys.path)
        import src.pipeline.experiment as expmod
        _orig_rb = expmod.run_backtest
        probe_info: dict = {}

        def _rb(signals, ohlcv, cfg=None):
            probe_info["score6_in_signals"] = bool("score6" in signals.columns)
            if "score6" in signals.columns:
                s6 = signals["score6"]
                probe_info["score6_stats"] = {
                    "n": int(s6.notna().sum()),
                    "mean": round(float(s6.mean()), 4),
                    "std": round(float(s6.std()), 4),
                }
            probe_info["bot_deepen_k"] = getattr(cfg, "bot_deepen_k", None)
            return _orig_rb(signals, ohlcv, cfg)

        expmod.run_backtest = _rb
        run_template_experiment = _rt.run_template_experiment

        t0 = time.time()
        r = run_template_experiment(template_id=tmpl_id, seed=42)
        dt = time.time() - t0
        result["runtime_s"] = round(dt, 1)
        result.update({k: v for k, v in probe_info.items()})
        if not r.get("success", True) or not r.get("run_id"):
            result["error"] = f"run failed: {r.get('error')}"
        else:
            result["run_id"] = r["run_id"]
            con = psycopg2.connect(**PG)
            cur = con.cursor()
            cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                        "FROM leaderboard_runs WHERE run_id=%s", (r["run_id"],))
            row = cur.fetchone()
            con.close()
            if row:
                result.update({"comp": float(row[0]), "pnl": float(row[1]),
                               "pf": float(row[2]), "mdd": float(row[3]),
                               "trades": int(row[4])})
                result["delta"] = round(result["comp"] - BASELINE["comp"], 2)
            else:
                result["error"] = "no leaderboard row"
    except Exception:
        result["error"] = traceback.format_exc()[-3000:]

    print("RESULT_JSON " + json.dumps(result, default=str), flush=True)


# ------------------------------------------------------------------ driver
def driver(only: list[str] | None = None) -> None:
    LOGS.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    todo = [(n, o) for n, o in PROBES if (not only or n in only)]
    results = []
    for name, _ in todo:
        log = LOGS / f"{name}.log"
        print(f"=== {name} starting (log: {log}) ===", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            p = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                  "--worker", name],
                                 cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, env=env)
            try:
                p.wait(timeout=KILL_GUARD_S)
            except subprocess.TimeoutExpired:
                p.kill()
                print(f"!!! {name} KILLED at {KILL_GUARD_S}s guard", flush=True)
                results.append({"probe": name, "error": "kill-guard 25min timeout",
                                "runtime_s": round(time.time() - t0, 1)})
                continue
        res = None
        for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
            if line.startswith("RESULT_JSON "):
                res = json.loads(line[len("RESULT_JSON "):])
                break
        if res is None:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
            res = {"probe": name, "error": "no RESULT_JSON; log tail:\n" + tail,
                   "runtime_s": round(time.time() - t0, 1)}
        results.append(res)
        if "comp" in res:
            print(f">>> {name}: comp={res['comp']:.1f} (D {res['delta']:+.1f}) "
                  f"pnl={res['pnl']:.2f} pf={res['pf']:.3f} mdd={res['mdd']:.5f} "
                  f"tr={res['trades']} runtime={res['runtime_s']:.0f}s "
                  f"score6={res.get('score6_in_signals')}", flush=True)
        else:
            print(f">>> {name}: FAILED runtime={res.get('runtime_s')}s\n"
                  f"{str(res.get('error'))[:2000]}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(results, indent=1, default=str), flush=True)
    print("PROBES_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        driver(sys.argv[1:] or None)
