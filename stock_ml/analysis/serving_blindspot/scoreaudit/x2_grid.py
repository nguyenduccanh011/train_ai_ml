"""x2 sweep: KENH BAN DOC LAP THU HAI (exit_ensemble OR-union) tren gb_x08 (2783).

Khuyen nghi SCORE_AUDIT.md #5: head moi (reward_risk h10 — tu no co duoi phai 2025-26,
z q99=6.9 khi head cu q99=1.14) phai vao nhu KENH SELL DOC LAP voi nguong z RIENG,
khong blend vao exit_score cu (blend w=0.3 giu lai 2.7% tin hieu).

Ha tang co san: engine_config.exit_ensemble = {target: {...}, z_threshold: Z}
 -> train head exit2 (exit features + exit_model params cua 2783), exit_score2
 -> recombine: sell |= causal_z(exit_score2, 252/60) > Z   (SAU exit_gate cons2_w20,
    khong bi gate — kenh doc lap that su; experiment.py:2489-2491).

Muc z tu x2_00_qmap.csv (causal z cua reward_risk h10, bundle n2_3h):
pooled q97=2.71 q98=3.17 q99=4.03; 2025 pct>3.0=4.6%, 2020-21+2023-24 gan nhu cam.

Clone-only tu 2783 (khong dung 2646/2730/2783). Fold-checkpoint identical giua cac
muc z (fp khong chua engine) -> copy folds tu run z dau tien sang cac muc sau.
Usage: python x2_grid.py <probe_id> [seed]
"""
from __future__ import annotations
import asyncio, copy, glob, json, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = 2783  # gb_x08

RR_TARGET = {"type": "reward_risk_regression", "horizon": 10}

# probe_id -> exit_ensemble z_threshold (None = parity clone, khong them kenh)
PROBES = {
    "x2_parity": None,
    "x2_rr10_z25": 2.5,
    "x2_rr10_z30": 3.0,
    "x2_rr10_z35": 3.5,
    "x2_rr10_z40": 4.0,
}

CHAMP42 = 729.6   # champion 2646 seed-42
GBX42 = 735.0     # gb_x08 2783 seed-42 (1378 trades, pnl 128.4965)


async def make_clone(name: str, z: float | None) -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(name)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = [{"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                  "rule_component_id": sl.rule_component_id, "feature_set_name": sl.feature_set_name,
                  "target_config": (json.loads(sl.target_config) if isinstance(sl.target_config, str)
                                    else copy.deepcopy(sl.target_config))}
                 for sl in base.component_slots]
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else copy.deepcopy(eng)
        if z is not None:
            eng["exit_ensemble"] = {"target": copy.deepcopy(RR_TARGET), "z_threshold": z}
        t = await repo.create(
            name=name, market=base.market, strategy=base.strategy,
            feature_set_id=base.feature_set_id, target_id=base.target_id,
            component_slots=copy.deepcopy(slots), direction=base.direction,
            signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
            split_config=base.split_config, engine_config=eng,
            validation_config=base.validation_config, seed=base.seed,
            description=(f"x2 probe over gb_x08 (2783): independent 2nd SELL channel "
                         f"exit_ensemble=reward_risk h10, z_threshold={z} (parity if None)"),
            hypothesis="SCORE_AUDIT: exit head cu sup phan phoi tu 2025 (z q99=1.14<2.0, "
                       "sell_ml 2025=1 bar); head reward_risk h10 tu no co duoi phai 2025-26 "
                       "(1042 sell 2025 standalone) nhung blend bi nuot (con 2.7% o w=0.3). "
                       "Kenh OR-union doc lap voi nguong rieng mo khoa gia tri bi nuot.",
            universe_slug=base.universe_slug, model_mode=base.model_mode)
        await s.commit()
        print(f"created clone: id={t.id} name={name} z={z}")
        return t.id


def read_row(run_id: str):
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades "
                "FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close(); return r


def copy_folds_from_sibling(tid: int, pid: str) -> None:
    """Reuse fold checkpoints tu mot run x2_rr10_* truoc do (models identical: fp khong
    chua engine.exit_ensemble.z_threshold; chi recombine+backtest khac)."""
    results = REPO / "results"
    dest_dirs = glob.glob(str(results / f"tmpl_{tid}_*"))
    # tim sibling da co folds (uu tien x2_rr10 khac)
    con = psycopg2.connect(**PG); cur = con.cursor()
    cur.execute("SELECT id FROM strategy_templates WHERE name LIKE 'x2_rr10_%%' AND id<>%s", (tid,))
    sib_ids = [r[0] for r in cur.fetchall()]; con.close()
    for sid in sib_ids:
        for d in glob.glob(str(results / f"tmpl_{sid}_*")):
            src = Path(d) / "folds"
            if src.exists() and any(src.glob("*.parquet")):
                fp = Path(d).name.split(f"tmpl_{sid}_")[1]
                dest = results / f"tmpl_{tid}_{fp}" / "folds"
                if dest.exists() and any(dest.glob("*.parquet")):
                    print(f"folds already present: {dest}")
                    return
                dest.mkdir(parents=True, exist_ok=True)
                for f in src.glob("*.parquet"):
                    shutil.copy2(f, dest / f.name)
                print(f"folds copied {src} -> {dest}")
                return
    print("no sibling folds found — full retrain")


def main():
    pid = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    z = PROBES[pid]
    tid = asyncio.run(make_clone(pid, z))
    asyncio.run(async_engine.dispose())
    if z is not None and seed == 42:
        copy_folds_from_sibling(tid, pid)
    r = run_template_experiment(template_id=tid, seed=seed)
    row = read_row(r.get("run_id"))
    if row and row[0] is not None:
        c = float(row[0])
        print(f"X2_RESULT {pid} seed={seed} tmpl={tid} comp={c} dGbx08={c - GBX42:+.1f} "
              f"dChamp={c - CHAMP42:+.1f} pnl={row[1]:.2f} pf={row[2]:.3f} mdd={row[3]:.5f} "
              f"tr={row[4]} run_id={r.get('run_id')}", flush=True)
    else:
        print(f"X2_RESULT {pid} seed={seed} tmpl={tid} NO_ROW run_id={r.get('run_id')}", flush=True)


if __name__ == "__main__":
    main()
