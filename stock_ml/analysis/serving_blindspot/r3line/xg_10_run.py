# -*- coding: utf-8 -*-
"""xg_10: cross-apply max_hold len 2 gem khac (NAV_GEM_SCAN): dyncsr88 (t2531),
v19_fullwave (t1799). Clone config-only, run seed 42, dump trades, cham nh_nav2
CA HAI che do (adv 0.08% + no-adv) x (full/f22/f23). Khuon = r3_10_run.py.

Usage: python xg_10_run.py <variant> [<variant> ...]   hoac  --list
Clone name = "xg_" + variant. KHONG dung canonical.
"""
from __future__ import annotations
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "r2line" / "na_audit"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")


def mh(n):
    return {"max_hold_bars": n, "_prepend_max_hold": True}


# variant -> (base_template_id, add_dict, remove_keys)
VARIANTS = {
    # dyncsr88 = t2531 (2429 + PURE-dynamics csrank head)
    "dyn_mh12": (2531, mh(12), []),
    "dyn_mh16": (2531, mh(16), []),
    "dyn_mh25": (2531, mh(25), []),
    # v19_fullwave = t1799 (champion 1798 + trend-intact trail-hold ma10)
    "v19_mh12": (1799, mh(12), []),
    "v19_mh16": (1799, mh(16), []),
    "v19_mh25": (1799, mh(25), []),
}


async def make_clone(variant: str) -> int:
    base_id, add, remove = VARIANTS[variant]
    name = f"xg_{variant}"
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: {name} id={ex.id}", flush=True)
            await async_engine.dispose()
            return ex.id
        base = await repo.get_by_id(base_id)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config)
                                    if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        add = dict(add)
        if add.pop("_prepend_max_hold", False):
            pri = list(eng.get("exit_priority") or ["trailing_stop", "overext", "signal"])
            if "max_hold" not in pri:
                pri = ["max_hold"] + pri
            eng["exit_priority"] = pri
        eng.update(add)
        for k in remove:
            eng.pop(k, None)
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"XG cross-apply maxhold: clone t{base_id} + {add} - {remove}",
            hypothesis="XGEM_CROSSAPPLY: max_hold la co che pho quat hay dac san t2429? "
                       "Sweep mh 12/16/25 tren dyncsr88 + v19_fullwave, cham nh_nav2 adv+noadv",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created: {name} id={t.id} base={base_id} add={add} rm={remove}", flush=True)
        tid = t.id
    await async_engine.dispose()
    return tid


def run_variant(variant: str):
    tid = asyncio.run(make_clone(variant))
    csv_path = HERE / f"xg_{variant}_s42_trades.csv"
    if not csv_path.exists():
        r = run_template_experiment(template_id=tid, seed=42)
        con = psycopg2.connect(**PG)
        cur = con.cursor()
        cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                    "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
        row = cur.fetchone()
        print(f"XGRUN xg_{variant} s42 comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
              f"mdd={row[3]:.3f} tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f}", flush=True)
        tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                          "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                          con, params=(r.get("run_id"),))
        con.close()
        tdf.to_csv(csv_path, index=False)
        print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
    for lo, tag in (("2020-01-01", "full"), ("2022-01-01", "f22"), ("2023-01-01", "f23")):
        sim = NavSim2(str(csv_path), date_lo=lo)
        a = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=0.0008, n=20)
        n = shuffle_stats(sim, K=25, roundtrip=0.006, settle_lag=2, advance_fee=None, n=20)
        print(f"XGNAV xg_{variant} {tag}: adv x{a['mean']:.2f}±{a['sd']:.2f} "
              f"DDw {a['dd_worst']*100:.1f}% | noadv x{n['mean']:.2f}±{n['sd']:.2f} "
              f"DDw {n['dd_worst']*100:.1f}%", flush=True)


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        for k, v in VARIANTS.items():
            print(k, "->", v)
        return
    if args[0] == "--all":
        args = list(VARIANTS)
    for v in args:
        if v not in VARIANTS:
            print(f"UNKNOWN variant {v}")
            continue
        print(f"\n===== VARIANT {v} =====", flush=True)
        try:
            run_variant(v)
        except Exception as e:  # noqa: BLE001
            print(f"XGFAIL {v}: {e!r}", flush=True)
    print("XG_10_DONE", flush=True)


if __name__ == "__main__":
    main()
