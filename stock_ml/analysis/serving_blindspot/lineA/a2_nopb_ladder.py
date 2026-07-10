"""Line A no-pullback ladder over champion 2646 (n2_2643_wavestruct_la05_lamp02), seed-42 screens.

Measures the raw gap of canonical at-market (close_next) entries vs the champion's
pullback-limit fill (4.5%/40), then climbs it with trade-management knobs (config-only).

Main mode runs each rung as a subprocess with a 480s kill-guard, appends results to
results.jsonl, and prints each rung as it lands.

Usage:
  python a2_nopb_ladder.py                 # base rungs 1-6
  python a2_nopb_ladder.py --rungs n1,n2   # selected rung names only
  python a2_nopb_ladder.py --worker <name> # internal
"""
from __future__ import annotations
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
BASE_TMPL = 2646
CHAMP = 729.6
SEED = 42
TIMEOUT_S = 480

# name -> engine_config overrides (None = JSON null = set key to None explicitly)
SPECS: dict[str, dict] = {
    "a2_nopb":         {"entry_pullback_pct": None},
    "a2_nopb_ss10":    {"entry_pullback_pct": None, "structural_stop_lookback": 10, "structural_stop_buffer": 0.0},
    "a2_nopb_ss20":    {"entry_pullback_pct": None, "structural_stop_lookback": 20, "structural_stop_buffer": 0.0},
    "a2_nopb_ss20b1":  {"entry_pullback_pct": None, "structural_stop_lookback": 20, "structural_stop_buffer": 0.01},
    "a2_nopb_hs08":    {"entry_pullback_pct": None, "hard_stop_pct": 0.08},
    "a2_nopb_ta15":    {"entry_pullback_pct": None, "trailing_activate_pct": 0.15},
    # combo rungs (added after inspecting rungs 1-6) go here:
    # hs08 was a literal no-op (identical row) -> the binding threshold is below 8%; 0.05 = the size
    # of the lost pullback buffer, tests whether capping the loss at -5% recovers PF.
    "a2_nopb_hs05":    {"entry_pullback_pct": None, "hard_stop_pct": 0.05},
    # champion min_hold_bars=2 holds through the first bars where the at-market entry mispricing
    # bites; 0 lets the exit machinery act immediately after an at-market fill.
    "a2_nopb_mh0":     {"entry_pullback_pct": None, "min_hold_bars": 0},
    # hs08/hs05 were INERT: engine.py:1827 only checks hard_stop_pct when "hard_stop" is in
    # exit_priority (champion's = trailing_stop/overext/signal), and the convention is NEGATIVE
    # (mtm_low <= hard_stop_pct). This is the corrected -8% hard stop, checked first in priority.
    "a2_nopb_hs08fix": {"entry_pullback_pct": None, "hard_stop_pct": -0.08,
                        "exit_priority": ["hard_stop", "trailing_stop", "overext", "signal"]},
}
BASE_RUNGS = ["a2_nopb", "a2_nopb_ss10", "a2_nopb_ss20", "a2_nopb_ss20b1", "a2_nopb_hs08", "a2_nopb_ta15"]


# ---------------------------------------------------------------- worker ----
def worker(name: str) -> None:
    sys.path.insert(0, str(REPO))
    import asyncio
    import psycopg2
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from stock_ml.db.engine import async_engine
    from stock_ml.db.repositories.template_repo import StrategyTemplateRepository
    from stock_ml.scripts.run_template import run_template_experiment

    overrides = SPECS[name]

    async def make_clone() -> int:
        Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        async with Session() as s:
            repo = StrategyTemplateRepository(s)
            ex = await repo.get_by_name(name)
            if ex:
                return ex.id
            base = await repo.get_by_id(BASE_TMPL)
            slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                      "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                      "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                        else copy.deepcopy(sl.target_config))}
                     for sl in base.component_slots]
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            eng.update(overrides)  # entry_pullback_pct set to None explicitly, never deleted
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"lineA no-pullback ladder over champion 2646: overrides={overrides}",
                hypothesis="at-market entries lose the 4.5% pullback price buffer; measure the raw gap "
                           "and recover it with stop/trailing trade-management knobs (config-only).",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            return t.id

    tid = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    print(f"[worker] template {name} id={tid} overrides={overrides}", flush=True)

    r = run_template_experiment(template_id=tid, seed=SEED)
    run_id = r.get("run_id")
    con = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                           user="stockml", password="stockml_dev")
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    row = cur.fetchone()
    con.close()
    out = {"name": name, "template_id": tid, "run_id": run_id,
           "composite": float(row[0]) if row and row[0] is not None else None,
           "pnl": float(row[1]) if row else None, "pf": float(row[2]) if row else None,
           "mdd": float(row[3]) if row else None, "trades": int(row[4]) if row else None}
    print("RESULT " + json.dumps(out), flush=True)


# ------------------------------------------------------------------ main ----
def load_results() -> dict[str, dict]:
    res = {}
    f = HERE / "results.jsonl"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                res[d["name"]] = d
    return res


def main(rungs: list[str]) -> None:
    results = load_results()
    for name in rungs:
        if name in results and results[name].get("composite") is not None:
            print(f"[skip] {name} already done: comp={results[name]['composite']}", flush=True)
            continue
        log = HERE / f"{name}.log"
        t0 = time.time()
        print(f"[run ] {name} overrides={SPECS[name]} ...", flush=True)
        try:
            with open(log, "w") as lf:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", name],
                               stdout=lf, stderr=subprocess.STDOUT, timeout=TIMEOUT_S,
                               cwd=str(REPO), check=True)
        except subprocess.TimeoutExpired:
            print(f"[KILL] {name} exceeded {TIMEOUT_S}s -> cache-miss, skipping", flush=True)
            results[name] = {"name": name, "composite": None, "status": "cache_miss_killed"}
            with open(HERE / "results.jsonl", "a") as f:
                f.write(json.dumps(results[name]) + "\n")
            continue
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {name} rc={e.returncode}; tail of {log.name}:", flush=True)
            print("\n".join(log.read_text().splitlines()[-15:]), flush=True)
            continue
        row = None
        for line in log.read_text().splitlines():
            if line.startswith("RESULT "):
                row = json.loads(line[7:])
        if row is None:
            print(f"[FAIL] {name}: no RESULT line; see {log}", flush=True)
            continue
        results[name] = row
        with open(HERE / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        nopb = results.get("a2_nopb", {}).get("composite")
        c = row["composite"]
        dch = f"{c - CHAMP:+.1f}" if c is not None else "n/a"
        dnp = f"{c - nopb:+.1f}" if (c is not None and nopb is not None) else "n/a"
        print(f"[done] {name}: comp={c} (Δchamp {dch}, Δnopb {dnp}) pnl={row['pnl']:.2f} "
              f"pf={row['pf']:.3f} mdd={row['mdd']:.5f} trades={row['trades']} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    print("LADDER_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "--rungs":
        main([x for x in sys.argv[2].split(",") if x])
    else:
        main(BASE_RUNGS)
