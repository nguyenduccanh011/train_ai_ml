"""GIVEBACK-AT-DEFER GUARD sweep: clone candidate 2730 (xq_snr_t08_g27), add ONLY the new
engine knob exit_snr_defer_min_giveback (entry-price units, matching ax_join.py giveback_at_defer).
Never touches templates 2646/2730. Usage: python gb_sweep.py <probe_id|tmpl_id> [seed]
Pattern copied from ax_rescue.py.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2730  # candidate xq_snr_t08_g27 — read-only

PROBES = {
    "gb_p00": {"exit_snr_defer_min_giveback": 0.0},   # parity: must be byte-identical to 2730
    "gb_x05": {"exit_snr_defer_min_giveback": 0.05},
    "gb_x08": {"exit_snr_defer_min_giveback": 0.08},
    "gb_x10": {"exit_snr_defer_min_giveback": 0.10},
    "gb_x12": {"exit_snr_defer_min_giveback": 0.12},
}

CHAMP = {42: 729.6, 7: 731.5, 99: 722.5, 555: 730.4, 123: 728.3}
CAND = {42: 733.3, 7: 735.0, 99: 726.8, 555: 733.5, 123: 731.5}  # 2730 per-seed


async def make_clone(name: str, eng_updates: dict) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        for k, sub in eng_updates.items():
            if isinstance(sub, dict) and isinstance(eng.get(k), dict):
                eng[k] = {**eng[k], **sub}
            else:
                eng[k] = sub
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"GIVEBACK GUARD over candidate 2730: eng={eng_updates}",
            hypothesis="Autopsy seed-42 §1c: defers with giveback_at_defer<=0.10 (entry units) are "
                       "83% worse (-0.46u, n=12); defers after deeper giveback are +2.97u (n=39) "
                       "and hold both mega-runners. New engine knob exit_snr_defer_min_giveback "
                       "only allows the snr-extend defer when (peak_high-close)/entry >= X.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} eng={eng_updates}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    if pid in PROBES:
        tid = asyncio.run(make_clone(pid, PROBES[pid]))
        asyncio.run(async_engine.dispose())
    else:
        tid = int(pid)
        pid = f"tmpl{tid}"
    r = run_template_experiment(template_id=tid, seed=seed)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        dch = f"{float(row[0]) - CHAMP[seed]:+.1f}" if seed in CHAMP else "n/a"
        dcd = f"{float(row[0]) - CAND[seed]:+.1f}" if seed in CAND else "n/a"
        print(f"GB_RESULT {pid} seed={seed} tmpl={tid} comp={row[0]} dChamp={dch} dCand={dcd} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]} "
              f"run_id={r.get('run_id')}", flush=True)
    else:
        print(f"GB_RESULT {pid} seed={seed} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
