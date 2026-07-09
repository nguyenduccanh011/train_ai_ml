"""A/B backtest of one template on the OLD (unadjusted) vs ADJUSTED duckdb.

Calls run_experiment directly on each data_root with a DISTINCT run_id (so the fold
checkpoints don't collide and each db trains fresh; the feature cache is already
content-addressed by data). Does NOT upsert to the leaderboard or persist run detail —
pure read-only comparison.

Usage:
    python stock_ml/scripts/compare_dbs.py --template-id 1844 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

OLD_DB = str(REPO_ROOT / "market_data" / "market.duckdb")
ADJ_DB = str(REPO_ROOT / "market_data" / "market_adjusted.duckdb")


def _run_async(coro):
    async def _wrapper():
        try:
            return await coro
        finally:
            from db.engine import async_engine
            await async_engine.dispose()
    return asyncio.run(_wrapper())


async def _load(template_id):
    from db.engine import async_engine
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker
    from src.pipeline.experiment import ExperimentConfig
    maker = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as s:
        return await ExperimentConfig.from_template_id_async(template_id, s)


def _score(summary):
    from src.evaluation.scoring import composite_score
    agg = summary.get("aggregate", {})
    frames = summary.get("_run_detail_frames", {})
    tr = frames.get("trades")
    trades_list = None
    if tr is not None and not tr.empty:
        trades_list = [
            {"symbol": r.get("symbol"), "entry_date": str(r.get("entry_date")),
             "pnl_pct": float(r.get("pnl_pct", 0) or 0),
             "holding_days": float(r.get("holding_days", 0) or 0)}
            for r in tr.to_dict("records")
        ]
    return composite_score(
        metrics={
            "trades": agg.get("n_trades", 0), "avg_pnl": agg.get("avg_pnl", 0.0),
            "total_pnl": agg.get("total_pnl", 0.0), "pf": agg.get("profit_factor", 0.0),
            "max_loss": agg.get("max_loss", 0.0), "avg_hold": agg.get("avg_hold_days", 0.0),
            "sharpe": summary.get("sharpe", 0.0), "n_symbols": summary.get("n_symbols", 0),
        },
        trades=trades_list,
    )


def _run_one(cfg, symbols, tag, db):
    from src.pipeline.experiment import run_experiment
    print(f"\n===== RUN [{tag}] data={db} =====", flush=True)
    summary = run_experiment(
        cfg=cfg, data_root=db, symbols=symbols,
        out_dir=str(REPO_ROOT / "results" / f"cmp_{tag}"),
        run_id=f"cmp{cfg.seed}_{tag}", export_csv=False,
    )
    agg = summary.get("aggregate", {})
    return {
        "tag": tag, "trades": agg.get("n_trades", 0), "wr": agg.get("win_rate", 0.0),
        "pf": agg.get("profit_factor", 0.0), "avg_pnl": agg.get("avg_pnl", 0.0),
        "total_pnl": agg.get("total_pnl", 0.0), "sharpe": summary.get("sharpe", 0.0),
        "mdd": summary.get("max_drawdown", 0.0), "mdd_sym": summary.get("mdd_per_symbol", 0.0),
        "composite": _score(summary),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-id", type=int, default=1844)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.utils.config_loader import get_pipeline_symbols
    cfg = _run_async(_load(args.template_id))
    cfg.seed = args.seed
    symbols = get_pipeline_symbols(market=cfg.market, universe_config=cfg.universe)
    print(f"template {args.template_id}: {cfg.name} | seed {args.seed} | {len(symbols)} symbols")

    rows = [_run_one(cfg, symbols, "old", OLD_DB), _run_one(cfg, symbols, "adj", ADJ_DB)]

    print("\n================ A/B RESULT (template %d) ================" % args.template_id)
    hdr = f"{'metric':<12}{'OLD(unadj)':>14}{'ADJUSTED':>14}{'delta':>12}"
    print(hdr); print("-" * len(hdr))
    for k, fmt in [("trades", "{:.0f}"), ("wr", "{:.3f}"), ("pf", "{:.3f}"),
                   ("avg_pnl", "{:.4f}"), ("total_pnl", "{:.3f}"), ("sharpe", "{:.4f}"),
                   ("mdd", "{:.3f}"), ("mdd_sym", "{:.4f}"), ("composite", "{:.2f}")]:
        o, a = rows[0][k], rows[1][k]
        print(f"{k:<12}{fmt.format(o):>14}{fmt.format(a):>14}{fmt.format(a - o):>12}")


if __name__ == "__main__":
    main()
