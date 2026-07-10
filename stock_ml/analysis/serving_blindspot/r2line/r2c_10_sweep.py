# -*- coding: utf-8 -*-
"""TUYEN R2 vong 3 (prefix r2c_): luoi quanh diem van hanh tot nhat r2b_oxtrail04.

Tam = fc_rule2 + pb4.0/50 + snr g12 + overext_trail_pct 0.04 (t2884, doc tu DB).
Truc: trail width {0.03,0.05,0.06} x arm threshold overext_pct {0.10,0.14}
      + snr window 40 + depth {3.9,4.1} interaction.
Cham diem MOI diem = nh_nav2 sim v2 (shuffle-mean 20 perm, R0.6, lag2 + adv 0.08%),
K25 full + f22, delta vs gb K25 adv — KHONG dung r2_nav.py.
Usage: python r2c_10_sweep.py <g1|g2|g3>
"""
from __future__ import annotations
import os
os.environ["STOCK_DATA_DIR"] = "F:/PROJECTS/train_ai_ml/market_data/market.duckdb"

import asyncio, copy, csv as csvmod, json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "na_audit"))

import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402
from nh_nav2 import NavSim2, shuffle_stats  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE = "fc_rule2"
ADV = 0.0008

SNR = {"exit_snr_extend_threshold": 0.8, "exit_snr_extend_window": 20,
       "exit_snr_min_gain": 0.12, "exit_snr_defer_min_giveback": 0.08}


def ox(trail, extra=None):
    p = {"entry_pullback_pct": 0.040, "entry_pullback_window": 50,
         "overext_trail_pct": trail}
    p.update(SNR)
    if extra:
        p.update(extra)
    return p


GROUPS = {
    # truc trail width (tam 0.04 = r2b_oxtrail04, khong re-run)
    "g1": [
        ("r2c_oxt03", ox(0.03)),
        ("r2c_oxt05", ox(0.05)),
        ("r2c_oxt06", ox(0.06)),
    ],
    # truc arm threshold (overext_pct, mac dinh 0.12) x trail
    "g2": [
        ("r2c_oxt04_ox10", ox(0.04, {"overext_pct": 0.10})),
        ("r2c_oxt04_ox14", ox(0.04, {"overext_pct": 0.14})),
        ("r2c_oxt03_ox10", ox(0.03, {"overext_pct": 0.10})),
        ("r2c_oxt05_ox14", ox(0.05, {"overext_pct": 0.14})),
    ],
    # bien the phu: snr window 40 + depth plateau interaction
    "g3": [
        ("r2c_oxt04_w40", ox(0.04, {"exit_snr_extend_window": 40})),
        ("r2c_oxt04_p39", ox(0.04, {"entry_pullback_pct": 0.039})),
        ("r2c_oxt04_p41", ox(0.04, {"entry_pullback_pct": 0.041})),
    ],
    # do day truc trail (g1 monotonic ve phia chat: 0.03 >> 0.04 > 0.05/0.06)
    "g4": [
        ("r2c_oxt025", ox(0.025)),
        ("r2c_oxt02", ox(0.02)),
    ],
    # combo cac truc f22-duong doc lap (oxt03 2.2sd, w40 1.7sd, p41 3.7sd-full)
    # + p42 kiem plateau-vs-gai cua truc depth duoi trail
    "g5": [
        ("r2c_oxt03_w40", ox(0.03, {"exit_snr_extend_window": 40})),
        ("r2c_oxt03_p41", ox(0.03, {"entry_pullback_pct": 0.041})),
        ("r2c_oxt04_p42", ox(0.04, {"entry_pullback_pct": 0.042})),
    ],
    # truc depth duoi trail monotonic tang den 4.2 -> do tiep 4.3 va 4.5 (= depth goc fc_rule2)
    "g6": [
        ("r2c_oxt04_p43", ox(0.04, {"entry_pullback_pct": 0.043})),
        ("r2c_oxt04_p45", ox(0.04, {"entry_pullback_pct": 0.045})),
    ],
}

# gb K25 adv reference (nh_frontier_results.csv)
GB = None
with open(HERE / "na_audit" / "nh_frontier_results.csv") as f:
    for r in csvmod.DictReader(f):
        if r["sys"] == "gb" and r["mode"] == "lag2-adv":
            GB = {k: float(r[k]) for k in ("full_mean", "full_sd", "f22_mean", "f22_sd")}
assert GB


def delta(mc, sc, mg, sg):
    d = (mc / mg - 1) * 100
    sd = (mc / mg) * math.sqrt((sc / mc) ** 2 + (sg / mg) ** 2) * 100
    return d, sd


def score(name, csv_path):
    sim_f = NavSim2(str(csv_path), date_lo="2020-01-01")
    sim_2 = NavSim2(str(csv_path), date_lo="2022-01-01")
    stf = shuffle_stats(sim_f, K=25, roundtrip=0.006, settle_lag=2, advance_fee=ADV)
    st2 = shuffle_stats(sim_2, K=25, roundtrip=0.006, settle_lag=2, advance_fee=ADV)
    df, sf = delta(stf["mean"], stf["sd"], GB["full_mean"], GB["full_sd"])
    d2, s2 = delta(st2["mean"], st2["sd"], GB["f22_mean"], GB["f22_sd"])
    print(f"R2C_SCORE {name} K25adv: full x{stf['mean']:.2f}±{stf['sd']:.2f} "
          f"DD {stf['dd_mean']*100:.1f}%/{stf['dd_worst']*100:.1f}% (vs gb {df:+.1f}%±{sf:.1f} "
          f"= {df/sf:.1f}sd) | f22 x{st2['mean']:.2f}±{st2['sd']:.2f} "
          f"(vs gb {d2:+.1f}%±{s2:.1f} = {d2/s2:.1f}sd)", flush=True)


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
                description=f"R2 line round3 (r2c_): {BASE} + {patch}",
                hypothesis="R2 round3: luoi quanh oxtrail04 (trail width x arm thr x snrw/depth)",
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
            print(f"SKIP run (csv exists): {name}", flush=True)
        else:
            r = run_template_experiment(template_id=tids[name], seed=42)
            con = psycopg2.connect(**PG); cur = con.cursor()
            cur.execute("SELECT composite_score, total_pnl, pf, trades, wr, avg_hold "
                        "FROM leaderboard_runs WHERE run_id=%s", (r.get("run_id"),))
            row = cur.fetchone()
            print(f"R2C_RESULT {name} s42 comp={row[0]:.1f} pnl={row[1]:.1f} pf={row[2]:.2f} "
                  f"tr={row[3]} wr={row[4]:.3f} hold={row[5]:.1f} run_id={r.get('run_id')}",
                  flush=True)
            tdf = pd.read_sql("select symbol, entry_date, exit_date, entry_price, exit_price, "
                              "holding_days, pnl_pct, exit_reason from run_trades where run_id=%s",
                              con, params=(r.get("run_id"),))
            con.close()
            tdf.to_csv(csv_path, index=False)
            print(f"dumped {len(tdf)} -> {csv_path.name}", flush=True)
        score(name, csv_path)
    print(f"R2C_SWEEP_DONE group={group}")


if __name__ == "__main__":
    main()
