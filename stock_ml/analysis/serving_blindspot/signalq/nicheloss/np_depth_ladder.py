"""Depth-bleed ladder over gb_x08 (tmpl 2783, top-1 song: exit stack moi snr_extend+giveback).
Cau hoi: exit stack moi co lam duong chay mau depth NONG hon khong?
- np_atmkt: bo pullback hoan toan (entry_pullback_pct=None, giong a2_nopb tren 2646)
- np_pb02 / np_pb03: depth 0.02 / 0.03, window giu 40
- op_pb02 / op_pb03: CONTROL exit cu — clone 2646 cung depth/window (bang cu chi co pb02_w10)
Khong dung 2646/2730/2783 (chi read-only clone). Seed 42. Pattern copy tu gb_sweep.py.
Usage: python np_depth_ladder.py <probe> [seed]
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

# probe -> (base_tmpl, engine overrides)
PROBES = {
    "np_atmkt": (2783, {"entry_pullback_pct": None}),
    "np_pb02":  (2783, {"entry_pullback_pct": 0.02}),
    "np_pb03":  (2783, {"entry_pullback_pct": 0.03}),
    "op_pb02":  (2646, {"entry_pullback_pct": 0.02}),
    "op_pb03":  (2646, {"entry_pullback_pct": 0.03}),
}
REF = {2783: 735.0, 2646: 729.6}  # seed-42 comp cua base


async def make_clone(name: str, base_tmpl: int, eng_updates: dict) -> int:
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
        eng.update(eng_updates)  # set explicit (None = at-market), never delete
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"depth-bleed ladder base={base_tmpl}: eng={eng_updates}",
            hypothesis="exit stack moi (snr_extend 0.8/20/0.27 + defer_min_giveback 0.08) co lam "
                       "duong chay mau depth (4.5->3->2->0%) nong hon so voi exit cu khong; neu "
                       "doc y nguyen -> dem gia 4.5% la alpha truc giao voi moi cai tien exit.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} base={base_tmpl} eng={eng_updates}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    base_tmpl, upd = PROBES[pid]
    tid = asyncio.run(make_clone(pid, base_tmpl, upd))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=seed)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        dref = f"{float(row[0]) - REF[base_tmpl]:+.1f}"
        print(f"NP_RESULT {pid} base={base_tmpl} seed={seed} tmpl={tid} comp={row[0]} dBase={dref} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]} "
              f"run_id={r.get('run_id')}", flush=True)
    else:
        print(f"NP_RESULT {pid} seed={seed} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
