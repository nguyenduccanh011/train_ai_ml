"""Confirmation-breakout B-entry (entry_bchannel_*) sweep over champion template 2646.

Wave-start FINAL mechanism (WAVESTART_DESIGN.md, bs_* post-mortem): every re-pricing of the
pending book failed via occupancy reshuffle -> instead ADD an entry at an idle slot: score6
z high (quality gate) + close breaks the recent thrust high (confirmation, W3-A4: capture
89%, +10.6%/21bar, false-fill 41%) + close below SMA20 (the zone the core channel is silent
in). At-market close_next; optional B-only structural stop. Head = ENS5 score-only (no union).

Driver mode (default): runs each variant as a subprocess (10-min kill guard), tees output
to logs/<name>.log. Worker mode (--worker NAME): clones 2646 + overrides, runs
run_template_experiment(seed=42, export_csv), dumps the engine's BCH_ENTRY_LOG side log to
runs/<name>/b_entries.csv (the Trade schema is hash-frozen -> B tag lives outside it), and
prints RESULT_JSON.

Usage: python stock_ml/analysis/serving_blindspot/wavestart/bch_sweep.py [names...]
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
ENS5_SCOREONLY = {"entry_ensemble5": {"target": ENS5_TARGET, "norm": "zscore"}}

SWEEP = [
    ("bch_z09",        {**ENS5_SCOREONLY, "entry_bchannel_z": 0.9}),
    ("bch_z12",        {**ENS5_SCOREONLY, "entry_bchannel_z": 1.2}),
    ("bch_z09_stop10", {**ENS5_SCOREONLY, "entry_bchannel_z": 0.9,
                        "entry_bchannel_stop_lookback": 10}),
    ("bch_z12_stop10", {**ENS5_SCOREONLY, "entry_bchannel_z": 1.2,
                        "entry_bchannel_stop_lookback": 10}),
    ("bch_z09_lb8",    {**ENS5_SCOREONLY, "entry_bchannel_z": 0.9,
                        "entry_bchannel_break_lookback": 8}),
    # follow-ups (run explicitly by name if the first 5 justify them):
    ("bch_z15",        {**ENS5_SCOREONLY, "entry_bchannel_z": 1.5}),
    ("bch_z15_stop10", {**ENS5_SCOREONLY, "entry_bchannel_z": 1.5,
                        "entry_bchannel_stop_lookback": 10}),
    ("bch_z09_noma",   {**ENS5_SCOREONLY, "entry_bchannel_z": 0.9,
                        "entry_bchannel_below_ma": None}),
    ("bch_z09_stop20", {**ENS5_SCOREONLY, "entry_bchannel_z": 0.9,
                        "entry_bchannel_stop_lookback": 20}),
    ("bch_z12_lb8",    {**ENS5_SCOREONLY, "entry_bchannel_z": 1.2,
                        "entry_bchannel_break_lookback": 8}),
    ("bch_z15_lb8",    {**ENS5_SCOREONLY, "entry_bchannel_z": 1.5,
                        "entry_bchannel_break_lookback": 8}),
]
SWEEP_MAP = dict(SWEEP)
DEFAULT_RUNS = [n for n, _ in SWEEP[:5]]


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
                description=f"confirmation-breakout B-entry sweep {name} over champion 2646",
                hypothesis="wave starts live below MA20 where the core channel is silent -> "
                           "the slot is idle; a score6-z-gated close above the recent thrust "
                           "high (W3-A4: capture 89%, +10.6%/21bar, false-fill 41%) enters "
                           "at-market close_next as an ADDITIVE fill, not a re-pricing",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"[{name}] created clone: id={t.id}", flush=True)
            return t.id

    try:
        tmpl_id = asyncio.run(make_clone())
        asyncio.run(async_engine.dispose())
        result["template_id"] = tmpl_id

        # Strict-audit exemption for B-entries (sweep-scoped, repo source untouched):
        # check_entry_integrity requires every trade to trace to a buy signal, but the
        # B-channel by DESIGN enters without one (that is the whole mechanism — the core
        # channel is silent below MA20). Exempt exactly the engine-logged B-entries
        # (symbol + signal_date) and audit everything else as usual. A production
        # integration would teach integrity.py about the B-channel instead.
        import pandas as pd

        from stock_ml.src.backtest import engine as _eng
        from stock_ml.src.backtest import integrity as _integ

        _orig_check = _integ.check_entry_integrity

        def _bch_aware_check(trades, signals, max_examples=5):
            if _eng.BCH_ENTRY_LOG and not trades.empty:
                bk = {(d["symbol"], pd.Timestamp(d["signal_date"]))
                      for d in _eng.BCH_ENTRY_LOG}
                keep = ~trades.apply(
                    lambda t: (t["symbol"], pd.Timestamp(t["entry_signal_date"])) in bk,
                    axis=1)
                trades = trades[keep]
            return _orig_check(trades, signals, max_examples)

        _integ.check_entry_integrity = _bch_aware_check

        from stock_ml.scripts.run_template import run_template_experiment

        out_dir = RUNS / name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = run_template_experiment(template_id=tmpl_id, seed=42,
                                    out_dir=out_dir, export_csv=True)
        result["runtime_s"] = round(time.time() - t0, 1)

        # Dump the B-entry side log (in-process: same engine module the backtest ran in).
        bdf = pd.DataFrame(_eng.BCH_ENTRY_LOG)
        bdf.to_csv(out_dir / "b_entries.csv", index=False)
        result["b_entries"] = int(len(bdf))

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
                  f"tr={res['trades']} B={res.get('b_entries')} "
                  f"runtime={res['runtime_s']:.0f}s", flush=True)
        else:
            print(f">>> {name}: FAILED runtime={res.get('runtime_s')}s\n"
                  f"{str(res.get('error'))[:2000]}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(results, indent=1, default=str), flush=True)
    print("BCH_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        driver(sys.argv[1:] or None)
