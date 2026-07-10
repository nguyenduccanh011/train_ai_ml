# -*- coding: utf-8 -*-
"""PV-CHANNEL probe: clone champion 2646, point the EXIT head at a new feature set
(exit_vol_downpress + pv channel), run one seed. New exit features => exit head retrains
(~15-20 min/run). Champion template 2646 is READ-ONLY (leaderboard upserts by run_name,
so clones use new names pv_*).

Usage: python stock_ml/analysis/serving_blindspot/ohlcv/pv_probe.py <probe_id> [seed]
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
BASE_TMPL = 2646  # champion n2_2643_wavestruct_la05_lamp02 (composite seed-42 729.6) — read-only
CHAMP = {42: 729.6, 7: 731.5, 99: 722.5, 555: 730.4, 123: 728.3}

# probe_id -> (exit feature set, engine-dict updates, top-level overrides)
SNR = {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
       "exit_snr_min_gain": 0.27}  # the 3 keys of template 2730 (snr08 promote candidate)
PROBES = {
    "pv_full": ("exit_vol_downpress_pv", {}, {}),
    "pv_only": ("exit_vol_downpress_pvonly", {}, {}),
    "pv_snr": ("exit_vol_downpress_pv", SNR, {}),
    "pv_only_snr": ("exit_vol_downpress_pvonly", SNR, {}),
}


async def make_clone(name: str, exit_fs: str, eng_updates: dict, top: dict) -> int:
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
        for sl in slots:
            if sl["slot_type"] == "exit":
                sl["feature_set_name"] = exit_fs
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        for k, sub in eng_updates.items():
            if isinstance(sub, dict) and isinstance(eng.get(k), dict):
                eng[k] = {**eng[k], **sub}
            else:
                eng[k] = sub
        t = await repo.create(
            name=name, market=base.market,
            strategy=top.get("strategy", base.strategy),
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=top.get("signal_threshold", base.signal_threshold),
            entry_threshold=top.get("entry_threshold", base.entry_threshold),
            exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"pv-channel probe over champion 2646: exit_fs={exit_fs} "
                        f"eng={eng_updates} top={top}",
            hypothesis="OHLCV screening survivor pv_corr_10 (resid-IC +0.041 vexit, 5/5 folds) "
                       "feeds the exit head new price-volume co-movement information.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} exit_fs={exit_fs} eng={eng_updates}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    pid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    exit_fs, eng_updates, top = PROBES[pid]
    tid = asyncio.run(make_clone(pid, exit_fs, eng_updates, top))
    asyncio.run(async_engine.dispose())
    r = run_template_experiment(template_id=tid, seed=seed)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        d = float(row[0]) - CHAMP.get(seed, 729.6)
        print(f"PV_RESULT {pid} seed={seed} tmpl={tid} comp={row[0]} d={d:+.1f} "
              f"pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} tr={row[4]}", flush=True)
    else:
        print(f"PV_RESULT {pid} seed={seed} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
