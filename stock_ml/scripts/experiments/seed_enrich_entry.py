"""Batch 4 — ENTRY-quality frontier. Trade outcome is separable at entry (OOS AUC 0.634 >
head's own score 0.583). The lean head already HAS volr/ret20/sma_50_ratio, so enrichment
must add GENUINELY-NEW info dimensions the per-symbol price head lacks:
  cheap  = mid-horizon cheapness (dist_63d_low, dist_126d_low, sma_200_ratio) [probe valued dist_lo60]
  accdist= accumulation/distribution-day counts (volume-FLOW, beyond volume_ratio)
  xsec   = cross-sectional ranks (this stock vs all others — new info, +24% fwd-IC per note)
  recov  = short-horizon recovery (dist_10d_low, range_pos_20, recov_setup)

Each = champion 1327 with ONLY the entry slot's feature_set_name swapped (exit + engine config
identical). Retrains the entry head on the richer features; backtest vs 404.3.
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402

BASE = 1327
GRID = [
    ("n2_af04_ent_cheap", "entry_lvup126_cheap"),
    ("n2_af04_ent_accdist", "entry_lvup126_accdist"),
    ("n2_af04_ent_xsec", "entry_lvup126_xsec"),
    ("n2_af04_ent_recov", "entry_lvup126_recov"),
]


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        repo = StrategyTemplateRepository(session)
        base = await repo.get_by_id(BASE)
        es = next(s for s in base.component_slots if s.slot_type == "entry")
        xs = next(s for s in base.component_slots if s.slot_type == "exit")

        def tc(sl):
            t = sl.target_config
            return json.loads(t) if isinstance(t, str) else copy.deepcopy(t)

        be = base.engine_config
        be = json.loads(be) if isinstance(be, str) else be
        created = []
        for name, fs in GRID:
            ex = await repo.get_by_name(name)
            if ex:
                print(f"= {name} ({ex.id})"); created.append((ex.id, name)); continue
            slots = [
                {"slot_type": "entry", "ml_component_id": es.ml_component_id, "rule_component_id": None,
                 "feature_set_name": fs, "target_config": tc(es)},
                {"slot_type": "exit", "ml_component_id": xs.ml_component_id, "rule_component_id": None,
                 "feature_set_name": xs.feature_set_name, "target_config": tc(xs)},
            ]
            tmpl = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(be),
                validation_config=base.validation_config, seed=base.seed,
                description=f"enrich entry feature set -> {fs}; base {BASE}.",
                hypothesis="Entry-quality separable (AUC 0.634); add NEW info dims to the entry head -> beat 404.3.",
                universe_slug=base.universe_slug,
            )
            print(f"* {name} ({tmpl.id}) fs={fs}"); created.append((tmpl.id, name))
        await session.commit()
        print("IDS=" + ",".join(str(t) for t, _ in created))
    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
