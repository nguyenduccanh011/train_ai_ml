# -*- coding: utf-8 -*-
"""convscale: clone 2783 (gb_x08) + knob conv_snr_w / conv_dma20_w, chay seed-42, autopsy reshuffle.

Usage: python cv_01_deploy.py [variant ...]   (default: tat ca theo thu tu, parity dau tien)
"""
from __future__ import annotations
import asyncio, copy, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2783
BASE_RUN_ID = "template/gb_x08-32a8dfee"   # gb_x08 seed-42 canonical
BASE_COMP_S42 = 735.0

VARIANTS = {
    # name: (snr_w, dma20_w)
    "cv_parity":  (0.0, 0.0),
    "cv_s03":     (0.3, 0.0),
    "cv_s06":     (0.6, 0.0),
    "cv_d03":     (0.0, 0.3),
    "cv_d06":     (0.0, 0.6),
    "cv_s03d03":  (0.3, 0.3),
    "cv_s06d03":  (0.6, 0.3),
}


async def make_clone(name: str, snr_w: float, dma_w: float) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}"); return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        eng.update({"entry_pullback_conv_snr_w": snr_w,
                    "entry_pullback_conv_dma20_w": dma_w})
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=f"gb_x08 (2783) + ex-ante structure conviction: snr_w={snr_w} dma20_w={dma_w} "
                        "(ENTRY_STRUCTURE_MAP lever #1: z(snr_sym20)+z(dist_MA20) 252/60 -> sigmoid -> "
                        "conv strength blend; continuous, no gate).",
            hypothesis="snr_sig Q4 = +15.8%/WR 68.8% (duy nhat duong ca 2024+2026), dma20_sig Q4 "
                       "+17.5%/WR 77.2% ex-ante tai bar signal — nghieng fill-depth ve cohort tot "
                       "se dich chuyen 3-8u noi bo ma khong cat lenh nao.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} snr_w={snr_w} dma20_w={dma_w}")
        return t.id


def read_lb(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score,total_pnl,pf,mdd_per_symbol,trades FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def autopsy(run_id: str):
    """Chu ky reshuffle: so cau truc trades voi gb_x08 s42 (join symbol+entry_date)."""
    con = psycopg2.connect(**PG)
    new = pd.read_sql("select symbol,entry_date,entry_price,exit_date,pnl_pct from run_trades where run_id=%(r)s",
                      con, params={"r": run_id})
    base = pd.read_sql("select symbol,entry_date,entry_price,exit_date,pnl_pct from run_trades where run_id=%(r)s",
                       con, params={"r": BASE_RUN_ID})
    con.close()
    for df in (new, base):
        df["k"] = df.symbol.astype(str) + "|" + df.entry_date.astype(str)
    m = new.merge(base, on="k", suffixes=("_n", "_b"))
    lost = base[~base.k.isin(new.k)]
    fresh = new[~new.k.isin(base.k)]
    dprice = m.entry_price_n / m.entry_price_b - 1
    wr = lambda s: float((s > 0).mean()) if len(s) else float("nan")
    mn = lambda s: float(s.mean()) if len(s) else float("nan")
    print(f"  [reshuffle] new={len(new)} base={len(base)} shared={len(m)} ({len(m)/max(len(base),1):.1%} base)"
          f"  fresh={len(fresh)} lost={len(lost)}")
    print(f"  [reshuffle] shared fill: same-price={float((dprice.abs()<1e-9).mean()):.1%}"
          f"  dat_hon(+)={float((dprice>1e-9).mean()):.1%} re_hon(-)={float((dprice<-1e-9).mean()):.1%}"
          f"  mean_dprice={float(dprice.mean()):+.4%}")
    print(f"  [reshuffle] pnl shared_new mean={mn(m.pnl_pct_n):+.4f} WR={wr(m.pnl_pct_n):.1%}"
          f" | shared_base mean={mn(m.pnl_pct_b):+.4f} WR={wr(m.pnl_pct_b):.1%}")
    print(f"  [reshuffle] fresh (lenh moi)  n={len(fresh)} mean={mn(fresh.pnl_pct):+.4f} WR={wr(fresh.pnl_pct):.1%} sum={fresh.pnl_pct.sum():+.2f}")
    print(f"  [reshuffle] lost  (lenh mat)  n={len(lost)} mean={mn(lost.pnl_pct):+.4f} WR={wr(lost.pnl_pct):.1%} sum={lost.pnl_pct.sum():+.2f}")
    # delta pnl tren lenh trung (cung entry date, gia khac -> exit path co the khac)
    dd = m.pnl_pct_n - m.pnl_pct_b
    print(f"  [reshuffle] shared dPnL: mean={float(dd.mean()):+.4f} sum={float(dd.sum()):+.2f}"
          f"  n_changed={int((dd.abs()>1e-9).sum())}")


def main():
    names = sys.argv[1:] or list(VARIANTS.keys())
    for name in names:
        snr_w, dma_w = VARIANTS[name]
        async def _clone_and_dispose():
            t = await make_clone(name, snr_w, dma_w)
            await async_engine.dispose()
            return t
        tid = asyncio.run(_clone_and_dispose())
        r = run_template_experiment(template_id=tid, seed=42)
        rid = r.get("run_id")
        row = read_lb(rid)
        comp = float(row[0]) if row and row[0] is not None else None
        d = f"{comp - BASE_COMP_S42:+.1f}" if comp is not None else "n/a"
        print(f"== {name} (tmpl {tid}) seed=42 run={rid}: comp={comp} (gb_x08 {BASE_COMP_S42}, D{d})"
              f" pnl={row[1]:.2f} pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}", flush=True)
        autopsy(rid)
        if name == "cv_parity":
            ok = (comp == BASE_COMP_S42 and int(row[4]) == 1378)
            print(f"== PARITY {'PASS' if ok else 'FAIL'}: comp={comp} trades={row[4]} (can 735.0/1378)")
            if not ok:
                print("== DUNG SWEEP: parity fail."); break
    print("CV_DEPLOY_DONE")


if __name__ == "__main__":
    main()
