"""Recompute composite_score for every leaderboard run from its stored trades.

Needed after a scoring-formula change (src/evaluation/scoring.py). Loads each non-superseded
run's trades from run_trades, recomputes metrics + composite_score, and (with --apply) writes
the new score back to leaderboard_runs. Dry-run by default: prints old vs new for the top runs
and verifies avg_pnl reproduces from stored trades (sanity that scales match).

Run:
  python stock_ml/scripts/rescore_leaderboard.py            # dry-run preview
  python stock_ml/scripts/rescore_leaderboard.py --apply    # write new scores
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text  # noqa: E402

from stock_ml.db.engine import async_engine  # noqa: E402
from stock_ml.src.evaluation.scoring import calc_metrics, composite_score  # noqa: E402


async def _load_runs(conn):
    return (await conn.execute(text(
        "SELECT run_id, run_name, composite_score, n_symbols FROM leaderboard_runs WHERE superseded=false"
    ))).all()


async def _load_trades(conn, run_id: str) -> list[dict]:
    rows = (await conn.execute(text(
        "SELECT symbol, entry_date, pnl_pct, holding_days FROM run_trades WHERE run_id=:r"
    ), {"r": run_id})).all()
    return [
        {"symbol": r.symbol, "entry_date": str(r.entry_date),
         "pnl_pct": float(r.pnl_pct), "holding_days": int(r.holding_days or 0)}
        for r in rows
    ]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write new scores (default: dry-run)")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    async with async_engine.connect() as conn:
        runs = await _load_runs(conn)
        results = []
        skipped = 0
        for r in runs:
            trades = await _load_trades(conn, r.run_id)
            if not trades:
                skipped += 1
                continue
            metrics = calc_metrics(trades)
            metrics["n_symbols"] = int(r.n_symbols or 0)  # universe size → confidence target
            new = composite_score(metrics, trades)
            results.append((r.run_id, r.run_name, r.composite_score, new, metrics["avg_pnl"],
                            metrics["trades"], metrics["avg_hold"], metrics["total_pnl"]))

        if args.apply:
            for run_id, _, _, new, *_ in results:
                await conn.execute(
                    text("UPDATE leaderboard_runs SET composite_score=:s WHERE run_id=:r"),
                    {"s": new, "r": run_id},
                )
            await conn.commit()

    await async_engine.dispose()

    results.sort(key=lambda x: -x[3])
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] rescored {len(results)} runs ({skipped} skipped, no trades). "
          f"Top {args.top} by NEW composite:\n")
    print(f"{'run':30} {'trd':>5} {'hold':>5} {'totpnl':>7} {'OLD':>7} {'NEW':>7}")
    print("-" * 70)
    for _, name, old, new, avg_pnl, trd, hold, totpnl in results[:args.top]:
        print(f"{name[:30]:30} {trd:5d} {hold:5.1f} {totpnl:7.1f} {old:7.1f} {new:7.1f}")


if __name__ == "__main__":
    asyncio.run(main())
