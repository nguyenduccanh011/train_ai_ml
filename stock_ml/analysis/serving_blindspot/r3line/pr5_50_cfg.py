# -*- coding: utf-8 -*-
"""pr5_50: (T4) verify t2936 = t2783 - 2 key T + mh16 + prepend max_hold, khong gi khac;
(T6 prep) diff t2531 (dyncsr88) vs t2429 — head csrank nam o key nao."""
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402


def eng_of(t):
    e = t.engine_config
    return json.loads(e) if isinstance(e, str) else dict(e)


def diff(name, a, b, ea, eb):
    print(f"\n===== DIFF {name} =====")
    keys = sorted(set(ea) | set(eb))
    n_same = 0
    for k in keys:
        va, vb = ea.get(k, "<ABSENT>"), eb.get(k, "<ABSENT>")
        if va != vb:
            print(f"  ENG {k}: {va!r} | {vb!r}")
        else:
            n_same += 1
    print(f"  (giong nhau: {n_same})")
    for f in ("market", "strategy", "signal_mode", "signal_threshold", "entry_threshold",
              "exit_threshold", "universe_slug", "model_mode", "feature_set_id",
              "target_id", "split_config"):
        va, vb = getattr(a, f), getattr(b, f)
        if va != vb:
            print(f"  META {f}: {va!r} | {vb!r}")
    sa = [(s.slot_type, s.ml_component_id, s.rule_component_id, s.feature_set_name,
           s.target_config) for s in a.component_slots]
    sb = [(s.slot_type, s.ml_component_id, s.rule_component_id, s.feature_set_name,
           s.target_config) for s in b.component_slots]
    if sa != sb:
        print(f"  SLOTS A: {sa}")
        print(f"  SLOTS B: {sb}")
    else:
        print(f"  SLOTS: giong het ({len(sa)})")


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        t2783 = await repo.get_by_id(2783)
        t2936 = await repo.get_by_name("ab_noT")
        t2429 = await repo.get_by_id(2429)
        t2531 = await repo.get_by_id(2531)
        diff("t2936 (ab_noT) vs t2783 (gb_x08)", t2936, t2783, eng_of(t2936), eng_of(t2783))
        diff("t2531 (dyncsr88) vs t2429", t2531, t2429, eng_of(t2531), eng_of(t2429))
        e = eng_of(t2531)
        for k in sorted(e):
            if "ensemble" in k:
                print(f"  t2531 {k} = {json.dumps(e[k])}")
    await async_engine.dispose()

asyncio.run(main())
print("PR5_50_DONE")
