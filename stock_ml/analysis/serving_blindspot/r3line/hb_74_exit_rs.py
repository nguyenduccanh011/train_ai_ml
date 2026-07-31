# -*- coding: utf-8 -*-
"""hb_74: EXIT RS head — clone ft_rs (t3102, frontier: entry RS + regime-skip300), doi exit
feature set -> exit_vol_rs (exit_vol_downpress + cross-sectional RS ranks). Test hypothesis:
cross-sectional RS SONG SOT z-scoring (khac market-regime common-factor da that bai) -> exit
ML CO THE dung RS de ban mã yeu-RS / giu mã manh-RS. Retrain (doi exit_features). So NAV vs
ft_rs x26.53.
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
BASE = 3102          # ft_rs (frontier)
NEWNAME = "ft_rs_exitrs"
NEW_EXIT_FS = "exit_vol_rs"


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
            slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                          "rule_component_id": sl.rule_component_id, "feature_set_name": fs,
                          "target_config": copy.deepcopy(sl.target_config)})
        t = await repo.create(
            name=NEWNAME, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config),
            engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42,
            description="ft_rs (entry RS + regime-skip300) + EXIT head cross-sectional RS "
                        "(exit_vol_rs). Test: RS ranks survive z-scoring -> exit ML dung duoc "
                        "(khac market-regime that bai). Ban mã yeu-RS / giu manh-RS.",
            hypothesis="cross-sectional RS per-symbol survive z-scoring; exit head them RS ranks "
                       "-> exit dung hon theo relative-strength.",
            universe_slug=base.universe_slug, model_mode="ml_only",
        )
        await s.commit()
        return t.id


def main():
    tid = asyncio.run(make())
    asyncio.run(async_engine.dispose())
    print(f"[hb_74] t{tid} = {NEWNAME} (exit_fs={NEW_EXIT_FS})", flush=True)
    r = run_template_experiment(template_id=tid, seed=42)
    rid = r.get("run_id")
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("select composite_score,total_pnl,pf,mdd_per_symbol,trades from leaderboard_runs where run_id=%s", (rid,))
    row = cur.fetchone(); con.close()
    if row:
        print(f"[hb_74] {rid}: comp={row[0]} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
    print(f"[hb_74] RUN_ID={rid}", flush=True); print("HB_74_DONE", flush=True)


if __name__ == "__main__":
    main()
