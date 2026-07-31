"""§11.5.4 — Re-baseline as ONE command, with an audit log.

An engine upgrade means "recompute the whole catalogue" (§11.1). This makes that a single,
logged EVENT — never a silent side effect of a container restart (§11.5.2). It cold-trains the
ML layer of the official catalogue (§14.1): the 6 DISTINCT models the 10 strategies map to (the
5 dyn300 portfolio-overlay variants share ONE model — they replay cheaply at the portfolio layer
and are NOT re-trained here). For each model it captures the published curve before→after and
appends ONE line to ``data/rebaseline_log.jsonl``: engine wheel + catalog fingerprint, who/when,
and every curve's delta.

Model→template is resolved by NAME at run time (fail-loud if a name is gone) so this file never
drifts into a stale hardcoded id list — the exact failure §13.9 documents.

Usage:
  python stock_ml/scripts/ops/rebaseline.py --dry-run          # plan + current before-values, no run
  python stock_ml/scripts/ops/rebaseline.py                    # single-seed cold-train, 6 models
  python stock_ml/scripts/ops/rebaseline.py --seeds 42 7 99    # multi-seed (deployed books, §11.7)
  python stock_ml/scripts/ops/rebaseline.py --models dyn900    # subset by model key
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "stock_ml"))

import psycopg2  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository  # noqa: E402
from stock_ml.scripts.run_template import run_template_experiment  # noqa: E402
from stock_ml.src.serving.resolved import _catalog_fingerprint, _wheel_version  # noqa: E402

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")

# The 6 distinct MODELS of the official catalogue (§14.1). Keyed by model, valued by the template
# NAME that trains it (resolved to an id at run time). The 5 dyn300 variants (#1-5) are NOT here —
# they share the dyn300 model and differ only in the portfolio: overlay (replayed, not re-trained).
MODEL_TEMPLATES: dict[str, str] = {
    "dyn300": "_dyn300_onerun",
    "dyn61a2hy": "_dyn61a2hy_onerun",
    "dyn900": "_dyn900_onerun",
    "wavestruct": "n2_2643_wavestruct_la05_lamp02",
    "consw20": "n2_consw20_conv04_vg_combo_hb_nbpbw",
    "velov": "n2_velov_univ150c",
}

_LOG = REPO / "data" / "rebaseline_log.jsonl"

# The published-curve columns the ML-layer leaderboard carries (the NAV/CAGR overlay is a separate,
# cheap replay layer — §14.1). Same set deploy_*.py reads, so before/after are directly comparable.
_CURVE_COLS = ("composite_score", "total_pnl", "pf", "mdd_per_symbol", "trades")


async def _resolve_ids(keys: list[str]) -> dict[str, int]:
    """model key -> template id, by NAME. Fail loud if a catalogue model has no template."""
    out: dict[str, int] = {}
    repo = StrategyTemplateRepository.__new__(StrategyTemplateRepository)
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as s:
        repo = StrategyTemplateRepository(s)
        for k in keys:
            t = await repo.get_by_name(MODEL_TEMPLATES[k])
            if t is None:
                raise SystemExit(
                    f"rebaseline: catalogue model {k!r} template {MODEL_TEMPLATES[k]!r} not in DB — "
                    "the catalogue drifted; fix MODEL_TEMPLATES or the template before re-baselining."
                )
            out[k] = t.id
    return out


def _read_curve(run_name: str, seed: int | None) -> dict | None:
    """Latest non-superseded published curve for a run_name (+ seed if given)."""
    con = psycopg2.connect(**PG)
    try:
        cur = con.cursor()
        q = (f"SELECT {', '.join(_CURVE_COLS)} FROM leaderboard_runs "
             "WHERE run_name=%s AND superseded=false")
        params: list = [run_name]
        if seed is not None:
            q += " AND run_seed=%s"
            params.append(seed)
        q += " ORDER BY created_at DESC LIMIT 1"
        cur.execute(q, params)
        r = cur.fetchone()
        if r is None:
            return None
        return {c: (float(v) if v is not None and c != "trades" else v)
                for c, v in zip(_CURVE_COLS, r, strict=True)}
    finally:
        con.close()


def _delta(before: dict | None, after: dict | None) -> dict:
    if not before or not after:
        return {}
    out = {}
    for c in _CURVE_COLS:
        b, a = before.get(c), after.get(c)
        if b is not None and a is not None:
            out[c] = round(a - b, 4) if c != "trades" else a - b
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry-run", action="store_true", help="show the plan + before-values, run nothing")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="seeds to re-baseline (default: single run on the template's own seed)")
    p.add_argument("--models", nargs="*", default=None,
                   help=f"subset of model keys (default all): {', '.join(MODEL_TEMPLATES)}")
    a = p.parse_args()

    keys = a.models or list(MODEL_TEMPLATES)
    unknown = [k for k in keys if k not in MODEL_TEMPLATES]
    if unknown:
        raise SystemExit(f"rebaseline: unknown model key(s) {unknown}; choose from {list(MODEL_TEMPLATES)}")
    seeds = a.seeds if a.seeds else [None]

    # A re-baseline = recompute the catalogue on the CURRENT engine/data (§11.1). Fold checkpoints are
    # keyed only by config fingerprint, so they'd silently be restored (recomputing nothing) after an
    # engine upgrade or data refresh — the exact stale-restore defect this event exists to prevent. Force
    # fresh retrain of every fold, and persist each fold's models so the deploy bundle can ship the EXACT
    # model that produced these signals (§11.9 attest == backtest by construction).
    if not a.dry_run:
        os.environ["STOCKML_FRESH_FOLDS"] = "1"
        os.environ["STOCKML_PERSIST_FOLD_MODELS"] = "1"

    ids = asyncio.run(_resolve_ids(keys))
    asyncio.run(async_engine.dispose())

    wheel, fp = _wheel_version(), _catalog_fingerprint()
    print(f"[rebaseline] engine wheel={wheel} catalog_fingerprint={fp[:12]}")
    print(f"[rebaseline] {len(keys)} model(s) × {len(seeds)} seed(s): "
          + ", ".join(f"{k}(tmpl {ids[k]})" for k in keys))

    results: list[dict] = []
    for k in keys:
        name = MODEL_TEMPLATES[k]
        for sd in seeds:
            before = _read_curve(name, sd)
            if a.dry_run:
                print(f"  [dry] {k} seed={sd or 'default'}: before={before}")
                results.append({"model": k, "template_id": ids[k], "seed": sd, "before": before})
                continue
            print(f"[rebaseline] cold-train {k} (tmpl {ids[k]}) seed={sd or 'default'} ...", flush=True)
            run_template_experiment(template_id=ids[k], seed=sd)
            after = _read_curve(name, sd)
            d = _delta(before, after)
            print(f"  {k} seed={sd or 'default'}: before={before} after={after} Δ={d}", flush=True)
            results.append({"model": k, "template_id": ids[k], "seed": sd,
                            "before": before, "after": after, "delta": d})

    if a.dry_run:
        print("[rebaseline] --dry-run: nothing trained, no audit line written.")
        return

    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "who": getpass.getuser(),
            "wheel_version": wheel,
            "catalog_fingerprint": fp,
            "seeds": seeds,
            "models": results,
        }, ensure_ascii=False, default=str) + "\n")
    print(f"[rebaseline] DONE — appended 1 audit line to {_LOG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
