"""OPERATING-POINT retest sweep: clone the rejected exit-label candidates (xl_swal06=2766,
xl_swal2=2762, pv_full=2779) and move ONLY the exit operating point — the sell z-band
signal_threshold (decoupled: sell = z(exit)>signal_threshold; exit_threshold is INERT) and,
for pv_full, the z_norm_window engine key. Label/target/features untouched. Never touches
2646/2730/2783. Pattern copied from gb_sweep.py. Usage: python op_grid.py <probe_id> [seed]
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

# probe_id -> (base_template, signal_threshold override or None, engine updates)
PROBES = {
    # xl_swal06 (2766, old verdict 726.2 @ st2.0; q-match z*=2.07)
    "op_s06_st16": (2766, 1.6, {}),
    "op_s06_st18": (2766, 1.8, {}),
    "op_s06_st21": (2766, 2.1, {}),   # quantile-matched point (2.067)
    "op_s06_st22": (2766, 2.2, {}),
    "op_s06_st24": (2766, 2.4, {}),
    # xl_swal2 (2762, old verdict 725.2 @ st2.0; DIST MISMATCH: fire@2.0=3.9% vs champ 5.4%,
    # q-match z*=1.81 -> old point too tight; retest at the fair point regardless of swal06)
    "op_sw2_st18": (2762, 1.8, {}),
    "op_sw2_st16": (2762, 1.6, {}),
    "op_sw2_st20q": (2762, 1.81, {}),  # exact q-match (only if 1.8/1.6 disagree)
    "op_sw2_st22": (2762, 2.2, {}),
    # pv_full (2779, old verdict 728.8 @ st2.0; dist ~= velocity, small room)
    "op_pv_st18": (2779, 1.8, {}),
    "op_pv_st22": (2779, 2.2, {}),
    "op_pv_zw189": (2779, None, {"z_norm_window": 189}),
    "op_pv_zw126": (2779, None, {"z_norm_window": 126}),
}

CHAMP42 = 729.6   # champion 2646 seed-42
GBX42 = 735.0     # gb_x08 2783 seed-42
OLD = {2766: 726.2, 2762: 725.2, 2779: 728.8}


async def make_clone(name: str, base_tmpl: int, st: float | None, eng_updates: dict) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(base_tmpl)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update(eng_updates)
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=st if st is not None else base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"operating-point retest over {base_tmpl} ({base.name}): "
                        f"signal_threshold={st} eng={eng_updates}",
            hypothesis="Rejected exit-label candidates were only ever tested at the champion's "
                       "operating point (sell z-band 2.0, z-norm 252/60) which was tuned FOR the "
                       "velocity head. Score-distribution audit: swal2's z is near-symmetric "
                       "(skew 0.05 vs 1.03) so 2.0 fires 28% less -> its fair point is z*=1.81; "
                       "swal06/pv are near-parity. Retest each label at ITS OWN operating point.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} base={base_tmpl} st={st} eng={eng_updates}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    base_tmpl, st, eng_up = PROBES[pid]
    tid = asyncio.run(make_clone(pid, base_tmpl, st, eng_up))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=seed)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        c = float(row[0])
        print(f"OP_RESULT {pid} seed={seed} tmpl={tid} comp={c} dChamp={c - CHAMP42:+.1f} "
              f"dGbx08={c - GBX42:+.1f} dOld={c - OLD[base_tmpl]:+.1f} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]} "
              f"run_id={r.get('run_id')}", flush=True)
    else:
        print(f"OP_RESULT {pid} seed={seed} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
