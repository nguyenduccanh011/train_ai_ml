# -*- coding: utf-8 -*-
"""hb_62: model MỚI — exit head regime-aware. Clone t2936, doi exit slot feature_set
-> exit_vol_regime (exit_vol_downpress + mkt>MA50/mom60 + sma_50 + rs60). Entry GIU
NGUYEN (dong). Doi exit_features -> _fp doi -> KHONG cache -> TRAIN LAI day du.
Seed 42. In composite + run_id. NAV/forensic cham rieng sau.
"""
from __future__ import annotations
import asyncio, copy, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from db.engine import async_engine  # noqa: E402
from db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = 2936
NEWNAME = "n2_2783_noT_exitregime"
NEW_EXIT_FS = "exit_vol_regime"


async def make() -> int:
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEWNAME)
        if ex:
            return ex.id
        base = await repo.get_by_id(BASE)
        slots = []
        for sl in base.component_slots:
            fs = NEW_EXIT_FS if sl.slot_type == "exit" else sl.feature_set_name
            slots.append({
                "slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                "rule_component_id": sl.rule_component_id, "feature_set_name": fs,
                "target_config": copy.deepcopy(sl.target_config),
            })
        t = await repo.create(
            name=NEWNAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42,
            description="ab_noT + exit head REGIME-AWARE (exit_vol_regime: +mkt>MA50/mom60 "
                        "+sma_50 +rs60). Muc tieu: giu bull-shakeout, ban bear-top "
                        "(shakeout_vs_top forensic AUC 0.78). Entry giu nguyen.",
            hypothesis="49% signal-exits la shakeout; regime (mkt>MA50) tach shakeout/top "
                       "AUC 0.78 trong khi price cuc bo ~0.5. Them regime -> exit dung hon.",
            universe_slug=base.universe_slug, model_mode="ml_only",
        )
        await s.commit()
        return t.id


def main():
    tid = asyncio.run(make())
    asyncio.run(async_engine.dispose())
    print(f"[hb_62] template t{tid} = {NEWNAME} (exit_fs={NEW_EXIT_FS})", flush=True)
    r = run_template_experiment(template_id=tid, seed=42)
    rid = r.get("run_id")
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("select composite_score,total_pnl,pf,mdd_per_symbol,trades from leaderboard_runs where run_id=%s", (rid,))
    row = cur.fetchone(); con.close()
    if row:
        print(f"[hb_62] {rid}: comp={row[0]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
    print(f"[hb_62] RUN_ID={rid}", flush=True)
    print("HB_62_DONE", flush=True)


if __name__ == "__main__":
    main()
