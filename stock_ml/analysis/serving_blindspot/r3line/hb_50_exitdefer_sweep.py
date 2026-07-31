# -*- coding: utf-8 -*-
"""hb_50: exit-defer sweep quanh ab_noT (t2936, comp 652) — model champion cua tuyen
exit-defer (single-slot, no-multipos). Clone slots + engine_config, chi doi knob EXIT
(khong dung entry/predictions), copy fold-cache de KHONG train lai -> moi variant chi
chay backtest+score. Muc tieu: vuot 652 bang cach monetize residual giveback tren
lenh 'signal' (avg pnl -1.44%, tra lai 5.66% tu dinh — forensic hb_giveback).

Chay:  python stock_ml/analysis/serving_blindspot/r3line/hb_50_exitdefer_sweep.py [batch]
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stock_ml"))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from db.engine import async_engine  # noqa: E402
from db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from src.pipeline.experiment import ExperimentConfig  # noqa: E402
from scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2936
SEED = 42
RESULTS = REPO / "results"
BASELINE_COMP = 652.0

# --- SWEEP BATCHES: variant_name -> {knob: value} overrides tren engine_config cua 2936 ---
BATCHES: dict[str, dict[str, dict]] = {
    # pilot: xac thuc cache-reuse + tai tao baseline
    "pilot": {
        "xd_base": {},                                    # sanity: phai ~652
        "xd_mh20": {"max_hold_bars": 20},
        "xd_snrgb06": {"exit_snr_defer_min_giveback": 0.06},
    },
    # batch1: one-knob quanh toi uu hien tai (max_hold la lever total_pnl manh nhat)
    "batch1": {
        "xd_mh14": {"max_hold_bars": 14},
        "xd_mh18": {"max_hold_bars": 18},
        "xd_mh22": {"max_hold_bars": 22},
        "xd_mh25": {"max_hold_bars": 25},
        "xd_mh30": {"max_hold_bars": 30},
        "xd_snrgb05": {"exit_snr_defer_min_giveback": 0.05},
        "xd_snrgb10": {"exit_snr_defer_min_giveback": 0.10},
        "xd_snrgb12": {"exit_snr_defer_min_giveback": 0.12},
        "xd_snrmg20": {"exit_snr_min_gain": 0.20},
        "xd_snrmg24": {"exit_snr_min_gain": 0.24},
        "xd_snrmg35": {"exit_snr_min_gain": 0.35},
        "xd_extatr10": {"signal_exit_hold_ext_atr": 1.0},
        "xd_extatr16": {"signal_exit_hold_ext_atr": 1.6},
        "xd_extatr20": {"signal_exit_hold_ext_atr": 2.0},
        "xd_sk3z12": {"signal_exit_skip_if_score3_z": 1.2},
        "xd_sk3z14": {"signal_exit_skip_if_score3_z": 1.4},
        "xd_sk3z20": {"signal_exit_skip_if_score3_z": 2.0},
        "xd_pfloor00": {"signal_exit_hold_profit_floor": 0.0},
        "xd_pfloor05": {"signal_exit_hold_profit_floor": -0.05},
        "xd_dropk20": {"signal_exit_protect_release_drop_k": 2.0},
        "xd_dropk30": {"signal_exit_protect_release_drop_k": 3.0},
        "xd_tstop07": {"trailing_stop_pct": 0.07},
        "xd_tstop10": {"trailing_stop_pct": 0.10},
        "xd_oxt03": {"overext_trail_pct": 0.03},
        "xd_oxt05": {"overext_trail_pct": 0.05},
    },
    # batch2: day max_hold cao hon (lever total_pnl) de tim dinh + combo micro-positives
    "batch2": {
        "xd_mh35": {"max_hold_bars": 35},
        "xd_mh40": {"max_hold_bars": 40},
        "xd_mh50": {"max_hold_bars": 50},
        "xd_mh75": {"max_hold_bars": 75},
        "xd_mh100": {"max_hold_bars": 100},
        "xd_mh150": {"max_hold_bars": 150},
        "xd_mhuncap": {"max_hold_bars": 10000},
        # remove max_hold from exit_priority entirely (pure trail/overext/signal, like gb)
        "xd_mhuncap_nopri": {"max_hold_bars": 10000,
                             "exit_priority": ["trailing_stop", "overext", "signal"]},
    },
    # batch3: uncap max_hold + stack cac positive nho + tinh chinh overext (gio la exit chinh)
    "batch3": {
        "xd_uncap_oxt03": {"max_hold_bars": 10000, "overext_trail_pct": 0.03},
        "xd_uncap_oxp10": {"max_hold_bars": 10000, "overext_pct": 0.10},
        "xd_uncap_oxp15": {"max_hold_bars": 10000, "overext_pct": 0.15},
        "xd_uncap_oxp10_oxt03": {"max_hold_bars": 10000, "overext_pct": 0.10, "overext_trail_pct": 0.03},
        "xd_uncap_stack": {"max_hold_bars": 10000, "overext_trail_pct": 0.03,
                           "signal_exit_skip_if_score3_z": 2.0, "exit_snr_min_gain": 0.20,
                           "signal_exit_hold_profit_floor": 0.0},
        "xd_uncap_tact20": {"max_hold_bars": 10000, "trailing_activate_pct": 0.20},
        "xd_uncap_tact35": {"max_hold_bars": 10000, "trailing_activate_pct": 0.35},
        "xd_uncap_tstop10": {"max_hold_bars": 10000, "trailing_stop_pct": 0.10},
    },
    # batch4: HUONG NAV-DUONG (nguoc composite) — giu NGAN hon + overext trail chat hon.
    # muc tieu toi uu NAV/CAGR (tieu chi that), f22 khong duoc giam.
    "batch4": {
        "xd_mh12": {"max_hold_bars": 12},
        "xd_mh13": {"max_hold_bars": 13},
        "xd_mh15": {"max_hold_bars": 15},
        "xd_oxt02": {"overext_trail_pct": 0.02},
        "xd_oxt03_smg20": {"overext_trail_pct": 0.03, "exit_snr_min_gain": 0.20},
        "xd_mh14_oxt03": {"max_hold_bars": 14, "overext_trail_pct": 0.03},
        "xd_mh13_oxt03": {"max_hold_bars": 13, "overext_trail_pct": 0.03},
        "xd_mh14_oxt02": {"max_hold_bars": 14, "overext_trail_pct": 0.02},
        "xd_mh14_oxt03_smg20": {"max_hold_bars": 14, "overext_trail_pct": 0.03,
                                "exit_snr_min_gain": 0.20},
        "xd_mh12_oxt03": {"max_hold_bars": 12, "overext_trail_pct": 0.03},
    },
    # batch6: APPROACH A — regime hold-gate (signal_exit_hold_mkt_scale) tren t2936. Giu signal-exit
    # SAU hon khi VNINDEX>MA (bull=shakeout hoi), thoat som khi bear (top). NGOAI z-scoring -> khai
    # thac tin hieu regime AUC-0.78 ma ML head khong the (z-scoring wash-out). Cache-reuse.
    "batch6": {
        "xr_ma50_s1": {"signal_exit_hold_mkt_scale": 1.0, "signal_exit_hold_mkt_feature": "ma50"},
        "xr_ma50_s2": {"signal_exit_hold_mkt_scale": 2.0, "signal_exit_hold_mkt_feature": "ma50"},
        "xr_ma50_s4": {"signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "ma50"},
        "xr_ma50_s8": {"signal_exit_hold_mkt_scale": 8.0, "signal_exit_hold_mkt_feature": "ma50"},
        "xr_ma100_s2": {"signal_exit_hold_mkt_scale": 2.0, "signal_exit_hold_mkt_feature": "ma100"},
        "xr_ma100_s4": {"signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "ma100"},
        "xr_ma200_s4": {"signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "ma200"},
        "xr_pos120_s4": {"signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "pos120"},
        "xr_dd120_s4": {"signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "dd120"},
    },
    # batch7: APPROACH A direct — regime SKIP hard rule (skip signal-exit khi VNINDEX>MA(N)).
    # Rong hon hold-gate: khong doi trade in-profit+trend path hep. Robust rule (chi giu bull tape,
    # khong lat vao bear). winner_only=chi giu winner (loser van cat). Cache-reuse.
    "batch7": {
        "xg_skip50": {"signal_exit_skip_if_mkt_above_ma": 50},
        "xg_skip50w": {"signal_exit_skip_if_mkt_above_ma": 50, "signal_exit_skip_if_mkt_winner_only": True},
        "xg_skip100w": {"signal_exit_skip_if_mkt_above_ma": 100, "signal_exit_skip_if_mkt_winner_only": True},
        "xg_skip200w": {"signal_exit_skip_if_mkt_above_ma": 200, "signal_exit_skip_if_mkt_winner_only": True},
        "xg_skip50m2w": {"signal_exit_skip_if_mkt_above_ma": 50, "signal_exit_skip_if_mkt_margin": 0.02, "signal_exit_skip_if_mkt_winner_only": True},
        "xg_skip50m5w": {"signal_exit_skip_if_mkt_above_ma": 50, "signal_exit_skip_if_mkt_margin": 0.05, "signal_exit_skip_if_mkt_winner_only": True},
        "xg_skip100m3w": {"signal_exit_skip_if_mkt_above_ma": 100, "signal_exit_skip_if_mkt_margin": 0.03, "signal_exit_skip_if_mkt_winner_only": True},
        "xg_skip50m2": {"signal_exit_skip_if_mkt_above_ma": 50, "signal_exit_skip_if_mkt_margin": 0.02},
    },
    # batch8: COMBINE 2 cai thien NAV doc lap — mh14_oxt03 (NAV-frontier +8%) + regime-skip100w (+4%).
    "batch8": {
        "xc_mh14o3_sk100w": {"max_hold_bars": 14, "overext_trail_pct": 0.03,
                             "signal_exit_skip_if_mkt_above_ma": 100, "signal_exit_skip_if_mkt_winner_only": True},
        "xc_mh14o3_sk150w": {"max_hold_bars": 14, "overext_trail_pct": 0.03,
                             "signal_exit_skip_if_mkt_above_ma": 150, "signal_exit_skip_if_mkt_winner_only": True},
        "xc_mh14o3_sk200w": {"max_hold_bars": 14, "overext_trail_pct": 0.03,
                             "signal_exit_skip_if_mkt_above_ma": 200, "signal_exit_skip_if_mkt_winner_only": True},
        "xc_mh14o3_sk100m3w": {"max_hold_bars": 14, "overext_trail_pct": 0.03,
                               "signal_exit_skip_if_mkt_above_ma": 100, "signal_exit_skip_if_mkt_margin": 0.03,
                               "signal_exit_skip_if_mkt_winner_only": True},
        "xc_mh16_sk150w": {"signal_exit_skip_if_mkt_above_ma": 150, "signal_exit_skip_if_mkt_winner_only": True},
    },
    # batch5: NAV-axis tren nen mh14_oxt03 (winner). Lever NAV-specific: reentry_cooldown
    # (toc do quay vong slot/von — truc giao composite), min_hold, overext/trailing tinh.
    # Moi variant da bao gom max_hold 14 + overext_trail 0.03.
    "batch5": {
        "xd_v5_rc0": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "reentry_cooldown_bars": 0},
        "xd_v5_rc1": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "reentry_cooldown_bars": 1},
        "xd_v5_rc2": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "reentry_cooldown_bars": 2},
        "xd_v5_rc6": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "reentry_cooldown_bars": 6},
        "xd_v5_mnh1": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "min_hold_bars": 1},
        "xd_v5_mnh3": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "min_hold_bars": 3},
        "xd_v5_oxt025": {"max_hold_bars": 14, "overext_trail_pct": 0.025},
        "xd_v5_oxt035": {"max_hold_bars": 14, "overext_trail_pct": 0.035},
        "xd_v5_ts06": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "trailing_stop_pct": 0.06},
        "xd_v5_ts07": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "trailing_stop_pct": 0.07},
        "xd_v5_ta20": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "trailing_activate_pct": 0.20},
        "xd_v5_oxp10": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "overext_pct": 0.10},
        "xd_v5_oxp15": {"max_hold_bars": 14, "overext_trail_pct": 0.03, "overext_pct": 0.15},
        "xd_v5_mh15": {"max_hold_bars": 15, "overext_trail_pct": 0.03},
    },
}

# batch9: dao quanh WINNER (mh14+oxt03+regime-skip MA200 winner-only). Tinh chinh skip window/margin,
# KET HOP 2 co che regime (skip + hold_mkt_scale), tune SNR-extend, noi overext/trail cho lenh deferred.
_WIN = {"max_hold_bars": 14, "overext_trail_pct": 0.03,
        "signal_exit_skip_if_mkt_above_ma": 200, "signal_exit_skip_if_mkt_winner_only": True}
BATCHES["batch9"] = {
    "xh_skip150w": {**_WIN, "signal_exit_skip_if_mkt_above_ma": 150},
    "xh_skip250w": {**_WIN, "signal_exit_skip_if_mkt_above_ma": 250},
    "xh_skip300w": {**_WIN, "signal_exit_skip_if_mkt_above_ma": 300},
    "xh_skip200m2w": {**_WIN, "signal_exit_skip_if_mkt_margin": 0.02},
    "xh_holdmkt100_s2": {**_WIN, "signal_exit_hold_mkt_scale": 2.0, "signal_exit_hold_mkt_feature": "ma100"},
    "xh_holdmkt50_s4": {**_WIN, "signal_exit_hold_mkt_scale": 4.0, "signal_exit_hold_mkt_feature": "ma50"},
    "xh_snrext06": {**_WIN, "exit_snr_extend_threshold": 0.6},
    "xh_snrext10": {**_WIN, "exit_snr_extend_threshold": 1.0},
    "xh_oxt04": {**_WIN, "overext_trail_pct": 0.04},
    "xh_tstop10": {**_WIN, "trailing_stop_pct": 0.10},
}
# batch10: dinh cua skip-window (MA cang cham cang loc bull nghiem; tim peak NAV)
BATCHES["batch10"] = {
    f"xi_skip{w}w": {**_WIN, "signal_exit_skip_if_mkt_above_ma": w}
    for w in (275, 325, 350, 400, 450, 500, 600)
}


def compute_fp(cfg) -> str:
    fp_src = {
        "strategy": cfg.strategy, "split": cfg.split, "seed": cfg.seed,
        "feature_set": cfg.feature_set, "entry_features": cfg.entry_features,
        "exit_features": cfg.exit_features, "entry_target": cfg.entry_target,
        "exit_target": cfg.exit_target, "target": cfg.target,
        "entry_model": cfg.entry_model, "exit_model": cfg.exit_model,
    }
    return hashlib.sha256(json.dumps(fp_src, sort_keys=True, default=str).encode()).hexdigest()[:10]


async def _load_cfg(tid: int) -> ExperimentConfig:
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        return await ExperimentConfig.from_template_id_async(tid, s)


async def make_template(name: str, overrides: dict) -> int:
    S = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{
            "slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
            "rule_component_id": sl.rule_component_id,
            "feature_set_name": sl.feature_set_name,
            "target_config": copy.deepcopy(sl.target_config),
        } for sl in base.component_slots]
        ec = copy.deepcopy(base.engine_config)
        ec.update(overrides)
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=ec,
            validation_config=base.validation_config, seed=SEED,
            description=f"exit-defer sweep off t2936: {json.dumps(overrides)}",
            hypothesis="monetize residual signal-exit giveback (hb_giveback forensic).",
            universe_slug=base.universe_slug, model_mode="ml_only",
        )
        await s.commit()
        return t.id


def seed_cache(new_id: int, fp: str) -> bool:
    """Copy fold-cache cua base 2936 sang dir cua variant de skip retrain."""
    src = RESULTS / f"tmpl_{BASE_TMPL}_{fp}" / "folds"
    dst = RESULTS / f"tmpl_{new_id}_{fp}" / "folds"
    if not src.is_dir():
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*.parquet"):
        if not (dst / p.name).exists():
            shutil.copy2(p, dst / p.name)
    return True


def read(run_id):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades,wr,avg_hold "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    sweep = BATCHES[batch]
    # base fp (seed 42) de copy cache
    base_cfg = asyncio.run(_load_cfg(BASE_TMPL)); base_cfg.seed = SEED
    fp = compute_fp(base_cfg)
    asyncio.run(async_engine.dispose())
    print(f"[hb_50] batch={batch}  base=t{BASE_TMPL}  fp={fp}  baseline_comp={BASELINE_COMP}", flush=True)
    print(f"        cache src exists: {(RESULTS/f'tmpl_{BASE_TMPL}_{fp}'/'folds').is_dir()}", flush=True)

    results = {}
    for name, ov in sweep.items():
        t0 = time.time()
        tid = asyncio.run(make_template(name, ov))
        asyncio.run(async_engine.dispose())
        seeded = seed_cache(tid, fp)
        r = run_template_experiment(template_id=tid, seed=SEED)
        rid = r.get("run_id")
        row = read(rid) if rid else None
        dt = time.time() - t0
        if row and row[0] is not None:
            comp = float(row[0]); delta = comp - BASELINE_COMP
            results[name] = comp
            print(f"  {name:14s} t{tid} comp={comp:7.1f} ({delta:+6.1f}) pnl={row[1]:6.1f} "
                  f"pf={row[2]:5.2f} mdd={row[3]:.3f} tr={row[4]} hold={row[6]:.1f} "
                  f"[{'cache' if seeded else 'RETRAIN'} {dt:.0f}s] {ov}", flush=True)
        else:
            print(f"  {name:14s} t{tid} NO RESULT [{dt:.0f}s] {ov}", flush=True)

    print("\n== SUMMARY (vs baseline 652.0) ==", flush=True)
    for name, comp in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:14s} {comp:7.1f} ({comp-BASELINE_COMP:+.1f})", flush=True)
    print("HB_50_DONE", flush=True)


if __name__ == "__main__":
    main()
