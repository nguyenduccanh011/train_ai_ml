"""Leaderboard endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.dependencies import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["leaderboard"])


async def _leaderboard_from_db(
    session: AsyncSession,
    market: str | None = None,
    state: str | None = None,
    strategy: str | None = None,
    feature_set: str | None = None,
    entry_model: str | None = None,
    timeframe: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    from stock_ml.db.repositories.run_repo import LeaderboardRunRepository

    repo = LeaderboardRunRepository(session)
    models = await repo.list_ranked(
        market=market,
        state=state,
        strategy=strategy,
        feature_set=feature_set,
        entry_model=entry_model,
        timeframe=timeframe,
        limit=limit,
        offset=offset,
    )
    rows = [
        {
            "run_id": m.run_id,
            "bundle": m.bundle,
            "run_name": m.run_name,
            "state": m.state,
            "composite_score": m.composite_score,
            "name": m.run_name,
            "market": m.market,
            "strategy": m.strategy,
            "feature_set": m.feature_set,
            "entry_model": m.entry_model,
            "target_type": m.target_type,
            "target_forward_window": m.target_forward_window,
            "model_mode": m.model_mode,
            "direction": m.direction,
            "timeframe": m.timeframe,
            "backtest_window_key": m.backtest_window_key,
            "first_test_year": m.first_test_year,
            "last_test_year": m.last_test_year,
            "trades": m.trades,
            "wr": m.wr,
            "pf": m.pf,
            "avg_pnl": m.avg_pnl,
            "pnl_pct": m.pnl_pct,
            "total_pnl": m.total_pnl,
            "max_win": m.max_win,
            "max_loss": m.max_loss,
            "avg_hold": m.avg_hold,
            "max_drawdown": m.max_drawdown,
            "mdd_per_symbol": m.mdd_per_symbol,
            "sharpe": m.sharpe,
            "n_symbols": m.n_symbols,
            "cost_commission": m.cost_commission,
            "cost_tax": m.cost_tax,
            "cost_slippage": m.cost_slippage,
            "audit_status": getattr(m, "audit_status", "—"),
            "date_run": m.generated_at.strftime("%Y-%m-%d") if m.generated_at else "—",
            "experiment_group": m.experiment_group,
            "variant_type": m.variant_type,
            "metadata_notes": m.metadata_notes,
        }
        for m in models
    ]
    return {
        "models": rows,
        "summary": {
            "total_models": len(rows),
            "best_model": rows[0].get("run_id") if rows else None,
        },
    }


@router.get("/leaderboard")
async def get_leaderboard(
    market: str | None = None,
    state: str | None = None,
    strategy: str | None = None,
    feature_set: str | None = None,
    entry_model: str | None = None,
    timeframe: str | None = None,
    offset: int = 0,
    limit: int = 200,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """Get leaderboard ranked by composite_score with optional filters."""
    if limit < 1 or limit > 5000:
        raise HTTPException(status_code=400, detail="limit must be 1-5000")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    # Postgres is the single source of truth. No JSON-file fallback: serving a
    # stale on-disk bundle would resurrect the dual-source problem this refactor
    # removed, so a DB failure surfaces loudly as a 500.
    return await _leaderboard_from_db(
        session,
        market=market,
        state=state,
        strategy=strategy,
        feature_set=feature_set,
        entry_model=entry_model,
        timeframe=timeframe,
        limit=limit,
        offset=offset,
    )


@router.get("/leaderboard/top/{n}")
async def get_top_models(n: int = 10, session: AsyncSession = Depends(get_db)) -> dict:
    """Get top N models."""
    if n < 1 or n > 1000:
        raise HTTPException(status_code=400, detail="n must be 1-1000")
    result = await _leaderboard_from_db(session, limit=n)
    return {"models": result["models"]}


@router.get("/leaderboard/{run_id:path}")
async def get_run(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    """Get a single run by run_id with full metrics for model-details page."""
    from stock_ml.db.repositories.run_repo import LeaderboardRunRepository

    repo = LeaderboardRunRepository(session)
    model = await repo.get_by_run_id(run_id)
    if not model:
        raise HTTPException(status_code=404, detail="Run not found")

    # Pull the z-score trigger thresholds from the source template so the
    # predictions chart can draw the decoupled BUY (z(entry) > entry_threshold) /
    # SELL (z(exit) > signal_threshold) reference lines that actually fire trades.
    entry_threshold = signal_threshold = exit_threshold = None
    engine_config: dict = {}
    if model.template_id is not None:
        from sqlalchemy import select

        from stock_ml.db.models.template import StrategyTemplateModel

        tmpl = (
            await session.execute(
                select(
                    StrategyTemplateModel.entry_threshold,
                    StrategyTemplateModel.signal_threshold,
                    StrategyTemplateModel.exit_threshold,
                    StrategyTemplateModel.engine_config,
                ).where(StrategyTemplateModel.id == model.template_id)
            )
        ).first()
        if tmpl is not None:
            entry_threshold, signal_threshold, exit_threshold, engine_config = tmpl
            engine_config = engine_config or {}

    return {
        "run_id": model.run_id,
        "bundle": model.bundle,
        "run_name": model.run_name,
        "name": model.run_name,
        "state": model.state,
        "composite_score": model.composite_score,
        "market": model.market,
        "strategy": model.strategy,
        "template_id": model.template_id,
        "entry_threshold": entry_threshold,
        "signal_threshold": signal_threshold,
        "exit_threshold": exit_threshold,
        "engine_config": engine_config,
        "feature_set": model.feature_set,
        "entry_model": model.entry_model,
        "target_type": model.target_type,
        "target_forward_window": model.target_forward_window,
        "model_mode": model.model_mode,
        "direction": model.direction,
        "timeframe": model.timeframe,
        "backtest_window_key": model.backtest_window_key,
        "first_test_year": model.first_test_year,
        "last_test_year": model.last_test_year,
        "trades": model.trades,
        "wr": model.wr,
        "pf": model.pf,
        "avg_pnl": model.avg_pnl,
        "pnl_pct": model.pnl_pct,
        "total_pnl": model.total_pnl,
        "max_win": model.max_win,
        "max_loss": model.max_loss,
        "avg_hold": model.avg_hold,
        "max_drawdown": model.max_drawdown,
        "mdd_per_symbol": model.mdd_per_symbol,
        "sharpe": model.sharpe,
        "n_symbols": model.n_symbols,
        "cost_commission": model.cost_commission,
        "cost_tax": model.cost_tax,
        "cost_slippage": model.cost_slippage,
        "experiment_group": model.experiment_group,
        "variant_type": model.variant_type,
        "metadata_notes": model.metadata_notes,
        "generated_at": model.generated_at.isoformat() if model.generated_at else None,
    }
