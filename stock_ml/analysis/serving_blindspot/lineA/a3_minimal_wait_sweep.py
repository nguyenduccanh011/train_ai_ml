"""a3 minimal-wait & hybrid pullback-execution sweep over champion 2646 (seed-42 screens).

Champion entry = pullback-limit 0.045/40: only 33.5% of limits fill, 7,582 signals dropped,
~213 live pending limits/day. Sweep: very shallow+short windows (0.015/5, 0.02/10, +-confirm
reversal) and hybrid fill-if-missed (premium-capped chase) to trade minimal alpha for operability.

Pattern = stock_ml/scripts/ops/deploy_wavestruct.py: clone 2646 via StrategyTemplateRepository
(copy component_slots + engine_config, update ONLY the pullback-execution keys; conviction
scaling entry_pullback_conv_* untouched), run seed 42, read leaderboard row by run_id.

Driver mode (default): runs each variant as a subprocess of itself with a 480s guard
(kill >8 min = cache-miss, continue). Child mode: --one <variant>.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2646  # n2_2643_wavestruct_la05_lamp02 (champion — never modified, only cloned)
SEED = 42
# Champion 2646 seed-42 baseline (established; not on leaderboard for seed 42):
CHAMP = dict(comp=729.6, pnl=127.25, pf=5.963, mdd=0.17455, trades=1384)

VARIANTS: dict[str, dict] = {
    "a3_pb015_w5": {"entry_pullback_pct": 0.015, "entry_pullback_window": 5},
    "a3_pb015_w5_cr": {"entry_pullback_pct": 0.015, "entry_pullback_window": 5,
                       "entry_pullback_confirm_reversal": True},
    "a3_pb02_w10": {"entry_pullback_pct": 0.02, "entry_pullback_window": 10},
    "a3_pb02_w10_cr": {"entry_pullback_pct": 0.02, "entry_pullback_window": 10,
                       "entry_pullback_confirm_reversal": True},
    "a3_hyb_fm02": {"entry_pullback_pct": 0.045, "entry_pullback_window": 40,
                    "entry_pullback_fill_if_missed": True, "fill_if_missed_max_premium": 0.02},
    "a3_hyb_fm04": {"entry_pullback_pct": 0.045, "entry_pullback_window": 40,
                    "entry_pullback_fill_if_missed": True, "fill_if_missed_max_premium": 0.04},
    "a3_hyb_w10_fm02": {"entry_pullback_pct": 0.045, "entry_pullback_window": 10,
                        "entry_pullback_fill_if_missed": True, "fill_if_missed_max_premium": 0.02},
    # follow-up combos (picked after the 7-variant screen):
    # fm premium 0.02 -> 0.04 monotonically worse => try tighter cap 0.01
    "a3_hyb_fm01": {"entry_pullback_pct": 0.045, "entry_pullback_window": 40,
                    "entry_pullback_fill_if_missed": True, "fill_if_missed_max_premium": 0.01},
    # window-only w20 lost just -41.4; capped chase should recover truncated deep fills at half the wait
    "a3_hyb_w20_fm02": {"entry_pullback_pct": 0.045, "entry_pullback_window": 20,
                        "entry_pullback_fill_if_missed": True, "fill_if_missed_max_premium": 0.02},
}


# ---------------------------------------------------------------- child mode
async def make_clone(name: str, add: dict) -> int:
    import copy
    import json

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from stock_ml.db.engine import async_engine
    from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}", flush=True)
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update(add)  # ONLY the pullback-execution keys; everything else from 2646 untouched
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"a3 minimal-wait/hybrid execution sweep over champion 2646: {add}",
            hypothesis="champion pullback 0.045/40 fills only 33.5% of limits (7,582 drops, ~213 "
                       "pending limits/day); a shallow short-window limit (+confirm-reversal) or a "
                       "premium-capped fill-if-missed hybrid keeps most alpha with far less waiting.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} ADD={add}", flush=True)
        return t.id


def run_one(name: str) -> None:
    import asyncio
    import json as _json

    import psycopg2

    from stock_ml.db.engine import async_engine
    from stock_ml.scripts.run_template import run_template_experiment

    add = VARIANTS[name]
    tid = asyncio.run(make_clone(name, add))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=SEED)
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
    row = cur.fetchone()
    con.close()
    if row is None:
        print(f"RESULT {_json.dumps({'name': name, 'error': 'no leaderboard row', 'run_id': r.get('run_id')})}",
              flush=True)
        return
    out = dict(name=name, tid=tid, run_id=r.get("run_id"), comp=float(row[0]),
               pnl=float(row[1]), pf=float(row[2]), mdd=float(row[3]), trades=int(row[4]))
    print(f"RESULT {_json.dumps(out)}", flush=True)


# --------------------------------------------------------------- driver mode
def drive(names: list[str]) -> None:
    import json as _json
    import subprocess
    import time

    log_dir = Path(__file__).resolve().parent
    results = {}
    for name in names:
        log = log_dir / f"{name}.log"
        t0 = time.time()
        print(f"--- {name}: {VARIANTS[name]}", flush=True)
        try:
            with open(log, "w") as fh:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name],
                               cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=480, check=False)
        except subprocess.TimeoutExpired:
            print(f"  GUARD: {name} exceeded 8 min -> killed, marked CACHE-MISS, continuing", flush=True)
            results[name] = {"name": name, "error": "cache-miss (>8min, killed)"}
            continue
        txt = log.read_text(errors="replace")
        res = None
        for line in txt.splitlines():
            if line.startswith("RESULT "):
                res = _json.loads(line[7:])
        if res is None:
            tail = "\n".join(txt.splitlines()[-15:])
            print(f"  FAILED (no RESULT line, {time.time()-t0:.0f}s). Log tail:\n{tail}", flush=True)
            results[name] = {"name": name, "error": "failed"}
            continue
        results[name] = res
        if "comp" in res:
            d = res["comp"] - CHAMP["comp"]
            print(f"  {name}: comp={res['comp']:.1f} (Δ{d:+.1f}) pnl={res['pnl']:.2f} "
                  f"pf={res['pf']:.3f} mdd={res['mdd']:.5f} tr={res['trades']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        else:
            print(f"  {name}: {res}", flush=True)
    print("SWEEP_JSON " + _json.dumps(results), flush=True)
    print("A3_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--one":
        run_one(sys.argv[2])
    else:
        names = sys.argv[1:] or list(VARIANTS)
        drive(names)
