# -*- coding: utf-8 -*-
"""P1-E2 buoc 1 — tao templates cho tuyen xsec_rank_topk (append-only, key moi).

Tao:
  - model component 'xr_lgbm_lambdarank_e1' (lightgbm, params DUNG p1_01_train.py E1)
  - template xr_k20  (chinh: K=20, exit rot top-30, reb 20 bar, close_next, downleg12)
  - template xr_k10  (sensitivity: K=10, exit rot top-20)
  - template xr_smoke_champ2646 (clone NGUYEN VAN champion 2646 — smoke append-only,
    composite seed42 phai khop 729.6)

Split = split chuan champion (wf-year, train 2y, gap 85, test 2020..2025+tail 2026H1).
Universe/cost giu nguyen leaderboard (61 ma, 0.7% roundtrip).
"""
from __future__ import annotations
import asyncio, copy, json, logging, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)

from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import (  # noqa: E402
    ModelComponentRepository, StrategyTemplateRepository,
)

CHAMP_ID = 2646  # champion canonical — CHI DOC de clone, khong run lai canonical

# params lambdarank = DUNG PARAMS cua p1_01_train.py (E1)
RANKER_PARAMS = dict(num_leaves=15, learning_rate=0.03, n_estimators=600,
                     min_child_samples=200, feature_fraction=0.8,
                     bagging_fraction=0.8, bagging_freq=5,
                     lambda_l1=0.1, lambda_l2=1.0, n_jobs=-1, verbosity=-1)

TARGET_H20 = {"type": "forward_return_regression", "horizon": 20}
FS_ENTRY = "entry_lvup126_recov"


def engine_cfg(k: int) -> dict:
    return {
        "costs": {"tax": 0.001, "slippage": 0.0015, "commission": 0.0015},
        "max_hold_bars": 10000,
        "min_hold_bars": 1,
        "hard_stop_pct": None,
        "exit_priority": ["signal"],
        "entry_bar_fill_type": "close_next",
        "signal_exit_enabled": True,
        "xsec_rank": {
            "k": k,
            "k_exit": k + 10,
            "rebalance_bars": 20,
            "force_exit_gate": "downleg12",
            "breadth_features": ["pct_above_ma50", "adv_pct"],
        },
    }


async def main():
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    out = {}
    async with Session() as s:
        comp_repo = ModelComponentRepository(s)
        tmpl_repo = StrategyTemplateRepository(s)

        comp = await comp_repo.get_by_name("xr_lgbm_lambdarank_e1")
        if comp is None:
            comp = await comp_repo.create(
                name="xr_lgbm_lambdarank_e1", role="entry", algorithm="lightgbm",
                params=RANKER_PARAMS,
                description="P1-E2 lambdarank ranker head — params = p1_01_train.py E1 "
                            "(objective/label_gain gan trong nhanh strategy xsec_rank_topk)")
            await s.flush()
        print(f"component xr_lgbm_lambdarank_e1 id={comp.id}")

        base = await tmpl_repo.get_by_id(CHAMP_ID)
        if base is None:
            raise SystemExit(f"champion template {CHAMP_ID} not found")
        base_split = (json.loads(base.split_config) if isinstance(base.split_config, str)
                      else copy.deepcopy(base.split_config))

        # ---- xr_k20 / xr_k10 ----
        for name, k in [("xr_k20", 20), ("xr_k10", 10)]:
            ex = await tmpl_repo.get_by_name(name)
            if ex:
                print(f"template exists: id={ex.id} name={name}")
                out[name] = ex.id
                continue
            slots = [
                {"slot_type": "entry", "ml_component_id": comp.id, "rule_component_id": None,
                 "feature_set_name": FS_ENTRY, "target_config": dict(TARGET_H20)},
                # exit slot rong (type none) CHI de per-slot feature/target khop entry —
                # tranh resolve them bo feature global; exit that su = membership signal.
                {"slot_type": "exit", "ml_component_id": None, "rule_component_id": None,
                 "feature_set_name": FS_ENTRY, "target_config": dict(TARGET_H20)},
            ]
            t = await tmpl_repo.create(
                name=name, market=base.market, strategy="xsec_rank_topk",
                feature_set_id=base.feature_set_id, target_id=4,  # fwd_return_reg_h20
                component_slots=slots, direction="long",
                signal_mode="entry_first", signal_threshold=0.0,
                entry_threshold=None, exit_threshold=None,
                split_config=copy.deepcopy(base_split), engine_config=engine_cfg(k),
                validation_config={"n_seeds": 1}, seed=42,
                description=(f"P1-E2 xsec ranking top-{k}: lambdarank (group=ngay, relevance "
                             f"quintile fwd20), rank moi 20 bar, hysteresis vao top-{k}/ra "
                             f"top-{k + 10}, close_next khong pullback, phanh downleg12"),
                hypothesis="E1 GO: Rank-IC .036-.043 (3 seed), spread thang net +3.4%/nam "
                           "median 95-100% phase; alpha selection tuong doi khac champion "
                           "(overlap ~1/3). Bar: composite >= 625.8; kill < 400.",
                universe_slug=base.universe_slug, model_mode="ml_only")
            await s.commit()
            print(f"created template: id={t.id} name={name}")
            out[name] = t.id

        # ---- smoke clone champion (append-only proof) ----
        name = "xr_smoke_champ2646"
        ex = await tmpl_repo.get_by_name(name)
        if ex:
            print(f"template exists: id={ex.id} name={name}")
            out[name] = ex.id
        else:
            slots = []
            for sl in base.component_slots:
                slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                              "rule_component_id": sl.rule_component_id,
                              "feature_set_name": sl.feature_set_name,
                              "target_config": (json.loads(sl.target_config)
                                                if isinstance(sl.target_config, str)
                                                else copy.deepcopy(sl.target_config))})
            eng = (json.loads(base.engine_config) if isinstance(base.engine_config, str)
                   else copy.deepcopy(base.engine_config))
            t = await tmpl_repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=slots, direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=copy.deepcopy(base_split), engine_config=eng,
                validation_config=(copy.deepcopy(base.validation_config)
                                   if base.validation_config else None),
                seed=base.seed,
                description="P1-E2 smoke: clone nguyen van champion 2646 — composite s42 "
                            "phai khop 729.6 (chung minh append-only khong dung duong cu)",
                hypothesis="parity clone", universe_slug=base.universe_slug,
                model_mode=base.model_mode)
            await s.commit()
            print(f"created template: id={t.id} name={name}")
            out[name] = t.id
    await async_engine.dispose()
    print("XR_TEMPLATES", json.dumps(out))


if __name__ == "__main__":
    asyncio.run(main())
