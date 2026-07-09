"""Multi-seed MEAN rescoring for the leaderboard (the TRUE ranking).

The leaderboard ranks by a SINGLE seed (config_hash ignores --seed, so each config's row holds
whatever seed ran last — in practice seed 42). Forensic (project_t1837_AB_refuted): the seed
spread on a fixed config is ~±3 composite pts, so the whole recent champion climb (t1830 475.5
-> t1844 481.7 = +6.2) sits INSIDE one seed's noise band. A sub-3pt single-seed gain is NOT a
real improvement. This tool recomputes a multi-seed MEAN for the top configs so the true (noise-
robust) ranking is visible.

Mechanism (NON-colliding with a live session): for each target template it CLONES the config to
a throwaway name, trains the clone on the EXTRA seeds (the original's stored seed is reused as one
sample — its row is never touched), aggregates mean/std, writes to the side table
`leaderboard_seed_stats`, then supersedes the clone's row. The originals are untouched.

Run:
  python stock_ml/scripts/rescore_multiseed.py --top 8                 # plan only (no training)
  python stock_ml/scripts/rescore_multiseed.py --top 8 --run           # train + write stats
  python stock_ml/scripts/rescore_multiseed.py --templates 1844,1842 --extra-seeds 7,99 --run
"""
from __future__ import annotations
import argparse, asyncio, copy, json, statistics, sys
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
DDL = """
CREATE TABLE IF NOT EXISTS leaderboard_seed_stats (
  template_id        INTEGER PRIMARY KEY,
  run_name           TEXT,
  headline_composite DOUBLE PRECISION,
  headline_seed      INTEGER,
  mean_composite     DOUBLE PRECISION,
  std_composite      DOUBLE PRECISION,
  min_composite      DOUBLE PRECISION,
  max_composite      DOUBLE PRECISION,
  n_seeds            INTEGER,
  seeds_json         TEXT,
  mean_minus_headline DOUBLE PRECISION,
  computed_at        TIMESTAMPTZ DEFAULT now()
);
"""


def pg_conn():
    return psycopg2.connect(**PG)


def top_templates(k: int, explicit: list[int] | None):
    con = pg_conn(); cur = con.cursor()
    if explicit:
        cur.execute(
            "SELECT template_id, run_name, composite_score, run_seed FROM leaderboard_runs "
            "WHERE template_id = ANY(%s) AND superseded=false ORDER BY composite_score DESC",
            (explicit,))
        rows = cur.fetchall()
    else:
        cur.execute(
            "SELECT template_id, run_name, composite_score, run_seed FROM leaderboard_runs "
            "WHERE superseded=false ORDER BY composite_score DESC LIMIT %s", (k + 20,))
        # exclude throwaway rescore clones (any name containing the __ms marker)
        rows = [r for r in cur.fetchall() if "__ms" not in (r[1] or "")][:k]
    con.close()
    return rows  # (template_id, run_name, headline_composite, headline_seed)


def read_clone_composite(run_id: str):
    con = pg_conn(); cur = con.cursor()
    cur.execute("SELECT composite_score FROM leaderboard_runs WHERE run_id=%s", (run_id,))
    r = cur.fetchone(); con.close()
    return float(r[0]) if r and r[0] is not None else None


def _clone_name(name: str) -> str:
    # "__msd" = deterministic-fix clone (fresh template_id -> clean fold cache, post-fix).
    return (name[:46] + "__msd")  # marker suffix; excluded from top selection


async def _make_all_clones(targets: list[tuple]) -> dict[int, tuple[int, str]]:
    """Create all throwaway clones in ONE event loop (async_engine is loop-bound; calling
    asyncio.run per template would bind it to a closed loop on the 2nd call). Returns
    {base_template_id: (clone_id, clone_name)}."""
    out: dict[int, tuple[int, str]] = {}
    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        for tid, name, *_ in targets:
            cname = _clone_name(name)
            ex = await repo.get_by_name(cname)
            if ex:
                out[tid] = (ex.id, cname); continue
            base = await repo.get_by_id(tid)
            slots = []
            for sl in base.component_slots:
                tc = sl.target_config
                tc = json.loads(tc) if isinstance(tc, str) else copy.deepcopy(tc)
                slots.append({"slot_type": sl.slot_type, "ml_component_id": sl.ml_component_id,
                              "rule_component_id": sl.rule_component_id,
                              "feature_set_name": sl.feature_set_name, "target_config": tc})
            eng = base.engine_config
            eng = json.loads(eng) if isinstance(eng, str) else eng
            t = await repo.create(
                name=cname, market=base.market, strategy=base.strategy,
                feature_set_id=base.feature_set_id, target_id=base.target_id,
                component_slots=copy.deepcopy(slots), direction=base.direction,
                signal_mode=base.signal_mode, signal_threshold=base.signal_threshold,
                entry_threshold=base.entry_threshold, exit_threshold=base.exit_threshold,
                split_config=base.split_config, engine_config=copy.deepcopy(eng),
                validation_config=base.validation_config, seed=base.seed,
                description=f"Multi-seed rescore clone of template {tid} (throwaway).",
                hypothesis="Seed-robustness rescore; not a real variant.",
                universe_slug=base.universe_slug, model_mode=base.model_mode)
            out[tid] = (t.id, cname)
        await s.commit()
    await async_engine.dispose()
    return out


def supersede_clone(name_like: str):
    con = pg_conn(); cur = con.cursor()
    cur.execute("UPDATE leaderboard_runs SET superseded=true WHERE run_name LIKE %s", (name_like,))
    con.commit(); con.close()


def upsert_stats(row: dict):
    con = pg_conn(); cur = con.cursor()
    cur.execute(DDL)
    cur.execute(
        """INSERT INTO leaderboard_seed_stats
        (template_id,run_name,headline_composite,headline_seed,mean_composite,std_composite,
         min_composite,max_composite,n_seeds,seeds_json,mean_minus_headline,computed_at)
        VALUES (%(template_id)s,%(run_name)s,%(headline_composite)s,%(headline_seed)s,
         %(mean_composite)s,%(std_composite)s,%(min_composite)s,%(max_composite)s,
         %(n_seeds)s,%(seeds_json)s,%(mean_minus_headline)s, now())
        ON CONFLICT (template_id) DO UPDATE SET
         run_name=EXCLUDED.run_name, headline_composite=EXCLUDED.headline_composite,
         headline_seed=EXCLUDED.headline_seed, mean_composite=EXCLUDED.mean_composite,
         std_composite=EXCLUDED.std_composite, min_composite=EXCLUDED.min_composite,
         max_composite=EXCLUDED.max_composite, n_seeds=EXCLUDED.n_seeds,
         seeds_json=EXCLUDED.seeds_json, mean_minus_headline=EXCLUDED.mean_minus_headline,
         computed_at=now()""", row)
    con.commit(); con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--templates", type=str, default=None, help="comma ids; overrides --top")
    ap.add_argument("--seeds", type=str, default="42,7,99", help="seeds to train (all fresh, deterministic)")
    ap.add_argument("--run", action="store_true", help="actually train (default: plan only)")
    args = ap.parse_args()
    explicit = [int(x) for x in args.templates.split(",")] if args.templates else None
    seed_list = [int(x) for x in args.seeds.split(",") if x.strip()]
    targets = top_templates(args.top, explicit)

    print(f"Targets ({len(targets)}): " + ", ".join(f"{t[0]}:{t[1]}({t[2]:.1f}/s{t[3]})" for t in targets))
    print(f"Seeds to train per config (ALL fresh, deterministic): {seed_list}")
    print("NOTE: stored leaderboard composites are PRE-determinism-fix (non-reproducible); retraining all.")
    if not args.run:
        print("\n[PLAN ONLY] re-run with --run to train + write leaderboard_seed_stats.")
        return

    clones = asyncio.run(_make_all_clones(targets))  # one event loop -> no closed-loop bug
    results = []
    for tid, name, headline, hseed in targets:
        clone_id, cname = clones[tid]
        seeds = {}
        for sd in seed_list:
            r = run_template_experiment(template_id=clone_id, seed=sd)
            comp = read_clone_composite(r.get("run_id"))
            if comp is not None:
                seeds[sd] = comp
            print(f"  {name} seed={sd}: {comp}", flush=True)
        supersede_clone(cname[:48] + "%")  # drop the throwaway clone row from the active board
        comps = list(seeds.values())
        mean = statistics.mean(comps)
        std = statistics.pstdev(comps) if len(comps) > 1 else 0.0
        stat = dict(template_id=tid, run_name=name, headline_composite=float(headline),
                    headline_seed=int(hseed), mean_composite=mean, std_composite=std,
                    min_composite=min(comps), max_composite=max(comps), n_seeds=len(comps),
                    seeds_json=json.dumps(seeds), mean_minus_headline=mean - float(headline))
        upsert_stats(stat)
        results.append(stat)
        print(f"== {name}: MEAN={mean:.1f} (headline {headline:.1f}, {mean-float(headline):+.1f}) "
              f"std={std:.1f} seeds={seeds}", flush=True)

    print("\n=== TRUE (multi-seed MEAN) RANKING ===")
    print(f"{'run_name':28} {'mean':>6} {'±std':>5} {'headline':>9} {'Δ':>6} {'spread':>12}")
    for s in sorted(results, key=lambda x: -x["mean_composite"]):
        sp = f"{s['min_composite']:.1f}-{s['max_composite']:.1f}"
        print(f"{s['run_name'][:28]:28} {s['mean_composite']:6.1f} {s['std_composite']:5.1f} "
              f"{s['headline_composite']:9.1f} {s['mean_minus_headline']:+6.1f} {sp:>12}")
    print("RESCORE_DONE")


if __name__ == "__main__":
    main()
