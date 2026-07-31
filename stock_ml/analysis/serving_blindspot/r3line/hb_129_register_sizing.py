# -*- coding: utf-8 -*-
"""hb_129: REGISTER conviction-sizing champions (user cho phep sizing, rang buoc TONG VON<=1 khong don
bay). struct_to + preempt + meta-prio conviction-size (alpha0.6, pscale=0.03 FIXED no-lookahead).
K25 fair ~105% CAGR/-14.2 (verified 3/3 seed, per-year all up incl dead-2024, pscale-fixed refutes
lookahead). Register x2_struct_to_k25size (fair) + x2_struct_to_k16size (aggressive); supersede preempt rows."""
from __future__ import annotations
import os, sys, warnings, asyncio, copy, hashlib
from pathlib import Path
warnings.filterwarnings("ignore")
import logging; logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "stock_ml"))
sys.path.insert(0, os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work"))
os.environ.setdefault("STOCK_DATA_DIR", "F:/PROJECTS/train_ai_ml/market_data/market.duckdb")
import psycopg2, pandas as pd
from nh_nav2 import NavSim2
from scripts.run_template import run_template_experiment
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from db.engine import async_engine
from db.repositories.template_repo import StrategyTemplateRepository
import hb_112_meta_target as M
import hb_127_conviction_sizing as S

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
HERE = Path(__file__).parent; ALPHA = 0.6; PSCALE = 0.03

SPECS = [
    ("x2_struct_to_k25size", 25, "template/x2_struct_to_k25preempt-69338138",
     "[K25 board-K + META-FILL + PREEMPT + CONVICTION-SIZING — FAIR-anchor, TONG VON<=1 khong don bay] "
     "Cung signal double-RS+struct-trail. Execution: K25 + preempt + size vi the = (nav/K)*mult, mult theo "
     "META-prio (alpha0.6, pscale0.03 FIXED khong look-ahead), tong invested<=nav (cash floor, position toi "
     "da ~10% NAV, cong bang khong don bay). Meta down-weight trade high-vol -> tilt chat-cao-rui-ro-thap. "
     "Multi-seed: CAGR ~105%/DD-14.2 (3/3 seed, MOI nam tang incl dead-2024 +36->+60%, pscale-fixed refute "
     "look-ahead). Vuot preempt-only 71.7% CUNG DD. Sizing hop le sau khi user noi rang buoc no-sizing."),
    ("x2_struct_to_k16size", 16, "template/x2_struct_to_k16preempt-69338138",
     "[K16 + META-FILL + PREEMPT + CONVICTION-SIZING — aggressive, TONG VON<=1 khong don bay] Nhu k25size "
     "nhung K16 concentration. Multi-seed CAGR ~112%/DD-16.4 (3/3 seed). Profile TAN CONG."),
]


async def make_tmpl(name, desc):
    S2 = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with S2() as s:
        repo = StrategyTemplateRepository(s); ex = await repo.get_by_name(name)
        if ex: return ex.id
        base = await repo.get_by_id(3185)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id, "rule_component_id": sl.rule_component_id,
                  "feature_set_name": sl.feature_set_name, "target_config": copy.deepcopy(sl.target_config)} for sl in base.component_slots]
        t = await repo.create(name=name, market=base.market, strategy=base.strategy, feature_set_id=base.feature_set_id,
            target_id=base.target_id, component_slots=slots, direction=base.direction, signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold, entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=copy.deepcopy(base.split_config), engine_config=copy.deepcopy(base.engine_config),
            validation_config=base.validation_config, seed=42, description=desc, hypothesis="preempt+conviction-sizing total<=1",
            universe_slug=base.universe_slug, model_mode="ml_only")
        await s.commit(); return t.id


def main():
    con = psycopg2.connect(**PG); cur = con.cursor()
    rid = run_template_experiment(template_id=3185, seed=42).get("run_id")
    cvtr = pd.read_sql("select symbol,entry_date,exit_date,entry_price,exit_price from run_trades where run_id=%s and exit_date is not null", con, params=(rid,))
    cv = HERE / "_k129.csv"; cvtr.to_csv(cv, index=False)
    feat = M.features(cvtr.symbol.unique().tolist()); pm = M.meta_preds(M.build_tr(con, rid, feat), tgt='t_pnl')
    for name, K, supersede, desc in SPECS:
        tid = asyncio.run(make_tmpl(name, desc)); asyncio.run(async_engine.dispose())
        cur.execute("update strategy_templates set description=%s where id=%s", (desc, tid)); con.commit()
        r2 = run_template_experiment(template_id=tid, seed=42).get("run_id")
        navf, cagr, dd = S.prun_sized(NavSim2(str(cv), date_lo="2020-01-01"), pm, K=K, alpha=ALPHA, pscale=PSCALE)
        nav22, _, _ = S.prun_sized(NavSim2(str(cv), date_lo="2022-01-01"), pm, K=K, alpha=ALPHA, pscale=PSCALE)
        print(f"{name} seed42: NAV=x{navf:.2f} CAGR={cagr*100:.1f}% DD={dd*100:.1f}% f22=x{nav22:.2f}", flush=True)
        ch = hashlib.md5(f"{r2}_size_a{ALPHA}".encode()).hexdigest()[:16]
        cur.execute("""insert into leaderboard_nav (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav, nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
                       values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                       on conflict (run_id) do update set nav_adv=excluded.nav_adv, cagr_adv=excluded.cagr_adv, maxdd_nav=excluded.maxdd_nav,
                         nav_f22_adv=excluded.nav_f22_adv, years=excluded.years, n_trades_sim=excluded.n_trades_sim, config_hash=excluded.config_hash, computed_at=now()""",
                    (r2, navf, navf, cagr, cagr, dd, nav22, 6.51, len(cvtr), ch))
        cur.execute("update leaderboard_runs set superseded=true where run_id=%s", (supersede,))
        con.commit()
        print(f"  REGISTERED {name} (t{tid}); superseded {supersede.split('/')[-1]}", flush=True)
    con.close(); print("HB_129_DONE", flush=True)


if __name__ == "__main__":
    main()
