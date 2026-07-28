"""Clone champion 2783 (gb_x08) + cross-sectional DISPERSION entry gate (campaign D9).

The entry head is a regime-fragile ranker: its IC vs forward return flips negative in LOW
cross-sectional dispersion regimes (2023/2024 macro tapes) where selection turns to noise, and
stays positive in HIGH-dispersion idiosyncratic regimes (D9 diagnostic, bootstrap-verified). Gate
= skip BUYs on dates where the universe cross-sectional std of dist-to-MA20 sits below its trailing
252-bar `pct` quantile. Engine-only knob (no retrain -> predictions cache-hit). Sweep pct to find
the quality/volume trade-off; seed 42 first, multi-seed the survivor.

Usage: venv/Scripts/python.exe stock_ml/scripts/experiments/build_dispersion_gate.py
"""
from __future__ import annotations
import asyncio, copy, json, sys
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
BASE_TMPL = 2783
SEEDS = [42]
# (name, pct) — higher pct gates more days (keeps fewer, higher-dispersion entries)
VARIANTS = [
    ("ds_p33", 0.33),
    ("ds_p50", 0.50),
    ("ds_p66", 0.66),
]


async def make_all() -> dict:
    """Create every variant clone inside ONE event loop (async_engine's pool binds to the loop;
    calling asyncio.run repeatedly with a shared engine pings a dead loop -> asyncpg crash)."""
    ids = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id,
                          "feature_set_name": sl.feature_set_name, "target_config": tc})
        base_eng = base.engine_config
        base_eng = json.loads(base_eng) if isinstance(base_eng, str) else dict(base_eng)
        for new_name, pct in VARIANTS:
            ex = await repo.get_by_name(new_name)
            if ex:
                print(f"clone exists: id={ex.id} name={new_name}"); ids[new_name] = (ex.id, pct); continue
            eng = copy.deepcopy(base_eng)
            eng.update({
                "entry_dispersion_gate_enabled": True,
                "entry_dispersion_ma": 20,
                "entry_dispersion_pct": pct,
                "entry_dispersion_lookback": 252,
            })
            t = await repo.create(
                name=new_name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"gb_x08 (2783) + cross-sectional dispersion entry gate pct={pct} (D9). "
                            "Skip buys on low-dispersion macro days where the entry ranker turns to noise.",
                hypothesis="Entry head IC flips negative in low cross-sectional dispersion regimes "
                           "(2023/24); gating entries to high-dispersion days restores regime-robust "
                           "selection and lifts the dead years (2024/2026).",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            await s.commit()
            print(f"created id={t.id} name={new_name} pct={pct}")
            ids[new_name] = (t.id, pct)
    return ids


def read(run_id):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,avg_pnl,pf,mdd_per_symbol,trades,avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    ids = asyncio.run(make_all())
    asyncio.run(async_engine.dispose())
    for name, (tid, pct) in ids.items():
        for sd in SEEDS:
            r = run_template_experiment(template_id=tid, seed=sd)
            rid = r.get("run_id")
            row = read(rid)
            if row:
                print(f"  {name}(t{tid},pct{pct}) seed={sd}: comp={row[0]} pnl={row[1]:.1f} "
                      f"avg={row[2]:.4f} pf={row[3]:.2f} mdd={row[4]:.3f} tr={row[5]} hold={row[6]:.1f} "
                      f"run_id={rid}", flush=True)
            else:
                print(f"  {name} seed={sd}: NO ROW run_id={rid}", flush=True)
    print("BUILD_DISPERSION_GATE_DONE")


if __name__ == "__main__":
    main()
