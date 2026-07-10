"""Quality-gated shallow-fill (bot_shallow_*) sweep over champion template 2646.

Wave-start design (WAVESTART_DESIGN.md): score6 (bottom-structure head) z HIGH for a buy
-> shrink the pullback limit depth (4.5% -> ~2%) so genuine wave-start shallow dips fill;
low z keeps the full knife-filter depth. Two head modes: score-only (bs_so_*, no union
buys) and union z0.9 (bs_un_*, adds wave-start buys too).

Driver mode (default): runs each variant as a subprocess (10-min kill guard), tees output
to logs/<name>.log. Worker mode (--worker NAME): clones 2646 + overrides, runs
run_template_experiment(seed=42, export_csv) and prints RESULT_JSON.

Usage: python stock_ml/analysis/serving_blindspot/wavestart/bs_sweep.py [names...]
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
RUNS = HERE / "runs"
BASE_TMPL = 2646
BASELINE = {"comp": 729.6, "pnl": 127.25, "pf": 5.963, "mdd": 0.17455, "trades": 1384}
KILL_GUARD_S = 10 * 60
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

ENS5_TARGET = {"type": "bottom_structure_entry_regression", "horizon": 8, "penalty": 1.5,
               "dip_window": 50, "require_turn": True, "park_window": 20, "park_penalty": 0.5}
ENS5 = {"entry_ensemble5": {"target": ENS5_TARGET, "z_threshold": 0.9, "norm": "zscore"}}
ENS5_SCOREONLY = {"entry_ensemble5": {"target": ENS5_TARGET, "norm": "zscore"}}

SWEEP = [
    ("bs_so_k05_f44", {**ENS5_SCOREONLY, "bot_shallow_k": 0.5, "bot_shallow_floor": 0.44}),
    ("bs_so_k10_f44", {**ENS5_SCOREONLY, "bot_shallow_k": 1.0, "bot_shallow_floor": 0.44}),
    ("bs_so_k10_f55", {**ENS5_SCOREONLY, "bot_shallow_k": 1.0, "bot_shallow_floor": 0.55}),
    ("bs_un_k05_f44", {**ENS5, "bot_shallow_k": 0.5, "bot_shallow_floor": 0.44}),
    ("bs_un_k10_f44", {**ENS5, "bot_shallow_k": 1.0, "bot_shallow_floor": 0.44}),
    ("bs_un_k10_f55", {**ENS5, "bot_shallow_k": 1.0, "bot_shallow_floor": 0.55}),
    # follow-ups (run explicitly by name if the first 6 justify them):
    ("bs_so_k15_f44", {**ENS5_SCOREONLY, "bot_shallow_k": 1.5, "bot_shallow_floor": 0.44}),
    ("bs_so_k05_f66", {**ENS5_SCOREONLY, "bot_shallow_k": 0.5, "bot_shallow_floor": 0.66}),
    ("bs_un_k15_f44", {**ENS5, "bot_shallow_k": 1.5, "bot_shallow_floor": 0.44}),
    ("bs_un_k05_f66", {**ENS5, "bot_shallow_k": 0.5, "bot_shallow_floor": 0.66}),
]
SWEEP_MAP = dict(SWEEP)
DEFAULT_RUNS = [n for n, _ in SWEEP[:6]]


def worker(name: str) -> None:
    sys.path.insert(0, str(REPO))
    import asyncio
    import copy

    import psycopg2
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from stock_ml.db.engine import async_engine
    from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

    overrides = SWEEP_MAP[name]
    result = {"run": name, "overrides": overrides}

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
                description=f"quality-gated shallow fill sweep {name} over champion 2646",
                hypothesis="wave starts offer ~2% dips (4.5% buffer in only 30%); score6 "
                           "bottom-structure z HIGH -> shrink that signal's pullback depth so "
                           "the genuine shallow dip fills; low z keeps the 4.5% knife filter",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"[{name}] created clone: id={t.id}", flush=True)
            return t.id

    try:
        tmpl_id = asyncio.run(make_clone())
        asyncio.run(async_engine.dispose())
        result["template_id"] = tmpl_id

        from stock_ml.scripts.run_template import run_template_experiment

        out_dir = RUNS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = run_template_experiment(template_id=tmpl_id, seed=42,
                                    out_dir=out_dir, export_csv=True)
        result["runtime_s"] = round(time.time() - t0, 1)
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


def driver(only: list[str] | None = None) -> None:
    LOGS.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    todo = only or DEFAULT_RUNS
    results = []
    for name in todo:
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
                results.append({"run": name, "error": "kill-guard 10min timeout",
                                "runtime_s": round(time.time() - t0, 1)})
                continue
        res = None
        for line in reversed(log.read_text(encoding="utf-8", errors="replace").splitlines()):
            if line.startswith("RESULT_JSON "):
                res = json.loads(line[len("RESULT_JSON "):])
                break
        if res is None:
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:])
            res = {"run": name, "error": "no RESULT_JSON; log tail:\n" + tail,
                   "runtime_s": round(time.time() - t0, 1)}
        results.append(res)
        if "comp" in res:
            print(f">>> {name}: comp={res['comp']:.1f} (D {res['delta']:+.1f}) "
                  f"pnl={res['pnl']:.2f} pf={res['pf']:.3f} mdd={res['mdd']:.5f} "
                  f"tr={res['trades']} runtime={res['runtime_s']:.0f}s", flush=True)
        else:
            print(f">>> {name}: FAILED runtime={res.get('runtime_s')}s\n"
                  f"{str(res.get('error'))[:2000]}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(results, indent=1, default=str), flush=True)
    print("BS_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        driver(sys.argv[1:] or None)
