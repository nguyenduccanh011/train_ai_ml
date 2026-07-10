# -*- coding: utf-8 -*-
"""TUYEN R2 vong 2: plateau quanh dinh pb4.0/50 + winner-riding combo (prefix r2b_).

Khung y het r2_10_sweep.py (clone fc_rule2 t2835, seed 42, leaderboard + NAV K25).
Groups:
  pa  — A: plateau depth {3.8,3.9,4.0,4.1,4.2} x window {45,50,55} tren nen snr g12
        (tam 4.0/50 = r2_c2_pb40snr da co, khong re-run)
  wc  — C: winner-riding combo tren tam plateau: overext_pct {0.12,0.16,0.18} x snr window {20,40}
        + 1 o overext_trail_pct 0.04 (overext -> arm trail chat thay vi ban thang, key gb-parity)
Usage: python r2b_10_sweep.py <group>
"""
from __future__ import annotations
import os
os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

import asyncio, copy, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = "fc_rule2"
HERE = Path(__file__).parent

SNR = {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
       "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.08}


def pb(d, w, extra=None):
    p = {"entry_pullback_pct": d, "entry_pullback_window": w}
    p.update(SNR)
    if extra:
        p.update(extra)
    return p


GROUPS = {
    # A — plateau quanh dinh 4.0/50 (tam = r2_c2_pb40snr, da co)
    "pa": [
        ("r2b_p38_45", pb(0.038, 45)),
        ("r2b_p40_45", pb(0.040, 45)),
        ("r2b_p42_45", pb(0.042, 45)),
        ("r2b_p38_50", pb(0.038, 50)),
        ("r2b_p39_50", pb(0.039, 50)),
        ("r2b_p41_50", pb(0.041, 50)),
        ("r2b_p42_50", pb(0.042, 50)),
        ("r2b_p38_55", pb(0.038, 55)),
        ("r2b_p40_55", pb(0.040, 55)),
        ("r2b_p42_55", pb(0.042, 55)),
    ],
    # chk — tai lap tam c2 (config y het r2_c2_pb40snr, ten moi) de xac nhan
    # khong co engine drift giua 2 vong (ky vong comp 569.9 / NAV x16.34 dung tung chu so)
    "chk": [
        ("r2b_chk_c2", pb(0.040, 50)),
    ],
    # C — winner-riding combo tren tam plateau 4.0/50 + snr g12
    # (o (ox12, w20) = chinh r2_c2_pb40snr — dung lam base cell, khong re-run)
    "wc": [
        ("r2b_ox16_w20", pb(0.040, 50, {"overext_pct": 0.16})),
        ("r2b_ox18_w20", pb(0.040, 50, {"overext_pct": 0.18})),
        ("r2b_ox12_w40", pb(0.040, 50, {"exit_snr_extend_window": 40})),
        ("r2b_ox16_w40", pb(0.040, 50, {"overext_pct": 0.16, "exit_snr_extend_window": 40})),
        ("r2b_ox18_w40", pb(0.040, 50, {"overext_pct": 0.18, "exit_snr_extend_window": 40})),
        ("r2b_oxtrail04", pb(0.040, 50, {"overext_trail_pct": 0.04})),
    ],
}


async def make_clones(variants):
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    tids = {}
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        base = await repo.get_by_name(BASE)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        for name, patch in variants:
            ex = await repo.get_by_name(name)
            if ex:
                tids[name] = ex.id
                print(f"exists: {name} id={ex.id}", flush=True)
                continue
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
            eng = copy.deepcopy(eng); eng.update(patch)
            t = await repo.create(
                name=name, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=eng,
                validation_config=base.validation_config, seed=base.seed,
                description=f"R2 line round2 (r2b_): {BASE} + {patch}",
                hypothesis="R2 round2: plateau-vs-spike quanh pb4.0/50 + winner-riding ox-x-snrw combo",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            tids[name] = t.id
            print(f"created: {name} id={t.id}", flush=True)
        await s.commit()
    await async_engine.dispose()
    return tids


def main():
    group = sys.argv[1]
    variants = GROUPS[group]
    tids = asyncio.run(make_clones(variants))
    for name, patch in variants:
        csv_path = HERE / f"{name}_s42_trades.csv"
        if csv_path.exists():
            print(f"SKIP (csv exists): {name}", flush=True)
        else:
            r = run_template_experiment(template_id=tids[name], seed=42)
            con = psycopg2.connect(**PG); cur = con.cursor()
            cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades, wr, avg_hold "
                        "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
            row = cur.fetchone()
            print(f"R2B_RESULT {name} s42 comp={row[0]:.1f} d_base={row[0]-568.3:+.1f} "
                  f"d_c2={row[0]-569.9:+.1f} pnl={row[1]:.1f} pf={row[2]:.2f} mdd={row[3]:.3f} "
                  f"tr={row[4]} wr={row[5]:.3f} hold={row[6]:.1f} run_id={r.get('run_id')}", flush=True)
            tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                              "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                              con, params=(r.get("run_id"),))
            con.close()
            tdf.to_csv(csv_path, index=False)
            print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
        # NAV K25 full-frame
        out = subprocess.run([sys.executable, str(HERE / "r2_nav.py"), str(csv_path), name, "25"],
                             capture_output=True, text=True)
        print(out.stdout.strip(), flush=True)
        if out.returncode != 0:
            print(out.stderr[-2000:], flush=True)
        # NAV-tu-2022 (chong ghost) — wc group can gate song/chet ngay tai cho
        if group == "wc":
            out2 = subprocess.run([sys.executable, str(HERE / "r2_nav.py"), str(csv_path),
                                   f"{name}_from2022", "25", "2022-01-01"],
                                  capture_output=True, text=True)
            print(out2.stdout.strip(), flush=True)
            if out2.returncode != 0:
                print(out2.stderr[-2000:], flush=True)
    print(f"R2B_SWEEP_DONE group={group}")


if __name__ == "__main__":
    main()
