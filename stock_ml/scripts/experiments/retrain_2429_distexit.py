"""EXIT RETRAIN on the current top-1 champion 2429 (n2_consw20_conv04_vg_combo_hb_nbpbw, 704.0).
Sandbox (results/_research_2429/exit_sandbox.py) showed the velocity exit head captures only 0.29-0.31
vs oracle-peak 0.67 -> weak exit = headroom. The champion exit uses exit_vol_market (volume MAGNITUDE,
"lags tops"); the validated top-predictor is the DISTRIBUTION-day volume axis (user domain insight;
exit_vol_dist won +2.78 robust on the nopullback line). This swaps ONLY the exit feature set
exit_vol_market -> exit_vol_dist2 on the FULL 2429 champion (never tried on this line) and retrains.

Per-seed 2429 baseline composites: {42:703.5, 7:698.8, 99:698.5, 555:704.0} (from multiseed_confirm).
Usage: python stock_ml/scripts/retrain_2429_distexit.py [exit_feat] [seed ...]
       default exit_feat=exit_vol_dist2, seeds=[42]
"""

from __future__ import annotations
import asyncio, copy, json, os, statistics, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import psycopg2  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
BASE_TMPL = int(
    os.environ.get("BASE_TMPL", "2429")
)  # set BASE_TMPL=2355 for the nopullback sandbox
BASE42 = float(os.environ.get("BASE42", "703.5"))
EXIT_FEAT = sys.argv[1] if len(sys.argv) > 1 else "exit_vol_dist2"
EXIT_TARGET = sys.argv[2] if len(sys.argv) > 2 else "-"  # "-" keeps champion velocity_exit target
SEEDS = [int(x) for x in sys.argv[3:]] or [42]
_tag = "" if EXIT_TARGET == "-" else f"_{EXIT_TARGET.split('_')[0]}"
NEW_NAME = f"n2_{BASE_TMPL}_{EXIT_FEAT}{_tag}"
# target_config presets for an exit target override (kept simple; horizon ~ champion's 20)
_TARGETS = {
    "fdd": {"type": "forward_drawdown_regression", "horizon": 20},
    "riskexit": {"type": "risk_exit_regression", "horizon": 20},
    "rewardrisk": {"type": "reward_risk_regression", "horizon": 20},
    # targets that read MORE temporal structure / more series (user: "nhiều thông tin/chuỗi hơn")
    "trendscan": {"type": "trend_scanning_exit", "windows": [5, 10, 20]},
    "downleg": {"type": "downleg_depth_regression", "max_span": 20},
    "swingval": {"type": "swing_value_regression", "horizon": 20},
    "mhret": {"type": "multi_horizon_return", "horizons": [5, 10, 20]},
}
BASE_PERSEED = {42: 703.5, 7: 698.8, 99: 698.5, 555: 704.0} if BASE_TMPL == 2429 else {42: BASE42}


async def make_clone() -> int:
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        ex = await repo.get_by_name(NEW_NAME)
        if ex:
            print(f"clone exists: id={ex.id}")
            return ex.id
        base = await repo.get_by_id(BASE_TMPL)
        slots = []
        for sl in base.component_slots:
            tc = sl.target_config
            tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
            if sl.slot_type == "exit" and EXIT_TARGET in _TARGETS:
                tc = copy.deepcopy(_TARGETS[EXIT_TARGET])  # override exit target
            feat = EXIT_FEAT if sl.slot_type == "exit" else sl.feature_set_name
            slots.append(
                {
                    "slot_type": sl.slot_type,
                    "ml_component_id": sl.ml_component_id,
                    "rule_component_id": sl.rule_component_id,
                    "feature_set_name": feat,
                    "target_config": tc,
                }
            )
        eng = base.engine_config
        eng = json.loads(eng) if isinstance(eng, str) else eng
        t = await repo.create(
            name=NEW_NAME,
            market=base.market,
            strategy=base.strategy,
            feature_set_id=base.feature_set_id,
            target_id=base.target_id,
            component_slots=copy.deepcopy(slots),
            direction=base.direction,
            signal_mode=base.signal_mode,
            signal_threshold=base.signal_threshold,
            entry_threshold=base.entry_threshold,
            exit_threshold=base.exit_threshold,
            split_config=base.split_config,
            engine_config=copy.deepcopy(eng),
            validation_config=base.validation_config,
            seed=base.seed,
            description=f"2429 champion + exit feature swap -> {EXIT_FEAT} (distribution-day top axis).",
            hypothesis="velocity exit head weak (sandbox capture 0.30 vs oracle 0.67); distribution-day "
            "volume features predict real tops the magnitude-only exit_vol_market misses.",
            universe_slug=base.universe_slug,
            model_mode=base.model_mode,
        )
        await s.commit()
        print(f"created clone: id={t.id} name={NEW_NAME} exit_feat={EXIT_FEAT}")
        return t.id


def read_composite(run_id: str):
    con = psycopg2.connect(**PG)
    cur = con.cursor()
    cur.execute(
        "SELECT composite_score, total_pnl, pf, mdd_per_symbol, trades FROM leaderboard_runs "
        "WHERE run_id=%s",
        (run_id,),
    )
    r = cur.fetchone()
    con.close()
    return r


def main():
    new_id = asyncio.run(make_clone())
    asyncio.run(async_engine.dispose())
    seeds = {}
    for sd in SEEDS:
        r = run_template_experiment(template_id=new_id, seed=sd)
        row = read_composite(r.get("run_id"))
        comp = float(row[0]) if row and row[0] is not None else None
        base = BASE_PERSEED.get(sd)
        d = f"{comp - base:+.1f}" if (comp is not None and base) else "?"
        seeds[sd] = comp
        print(
            f"  {NEW_NAME} seed={sd}: comp={comp} (vs 2429 {base} Δ{d}) pnl={row[1]:.1f} "
            f"pf={row[2]:.2f} mdd={row[3]:.3f} tr={row[4]}",
            flush=True,
        )
    comps = [v for v in seeds.values() if v is not None]
    if comps:
        mean = statistics.mean(comps)
        _bvals = [BASE_PERSEED[s] for s in seeds if s in BASE_PERSEED]
        if _bvals:
            bmean = statistics.mean(_bvals)
            print(
                f"\n== {NEW_NAME}: MEAN={mean:.1f} seeds={seeds}  vs 2429 baseline MEAN={bmean:.1f} (Δ={mean - bmean:+.1f})"
            )
        else:
            dmean = mean - BASE42
            print(
                f"\n== {NEW_NAME}: MEAN={mean:.1f} seeds={seeds}  vs BASE42={BASE42:.1f} (Δ={dmean:+.1f})"
            )
    print("RETRAIN_DISTEXIT_DONE")


if __name__ == "__main__":
    main()
