"""Model lifecycle management routes."""

from __future__ import annotations

import calendar
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.dependencies import get_db
from stock_ml.db.repositories.run_repo import LeaderboardRunRepository
from stock_ml.db.repositories.trade_repo import RunTradeRepository

router = APIRouter(prefix="/api/v1", tags=["runs"])


def _epoch(dt: datetime | None) -> int | None:
    """Naive bar timestamp (exchange wall-clock) -> UTC epoch seconds for lightweight-charts.
    timegm treats the fields as UTC so the chart renders the same wall-clock the OHLCV endpoint
    uses for intraday candles — markers and candles stay aligned. NULL (daily runs) -> None."""
    return int(calendar.timegm(dt.timetuple())) if dt is not None else None


def _results_dir() -> Path:
    from stock_ml.src.utils.env import get_results_dir

    return Path(get_results_dir())


async def _read_rows(session: AsyncSession) -> list[dict[str, Any]]:
    repo = LeaderboardRunRepository(session)
    rows = await repo.list_ranked(limit=10000)
    return [
        {
            "run_id": r.run_id,
            "state": r.state,
            "market": r.market,
            "strategy": r.strategy,
            "feature_set": r.feature_set,
            "entry_model": r.entry_model,
            "composite_score": r.composite_score,
            "bundle": r.bundle,
            "run_name": r.run_name,
        }
        for r in rows
    ]


async def _find_row(session: AsyncSession, run_id: str) -> dict[str, Any] | None:
    repo = LeaderboardRunRepository(session)
    row = await repo.get_by_run_id(run_id)
    return (
        {
            "run_id": row.run_id,
            "state": row.state,
            "market": row.market,
            "strategy": row.strategy,
            "feature_set": row.feature_set,
            "entry_model": row.entry_model,
            "composite_score": row.composite_score,
            "bundle": row.bundle,
            "run_name": row.run_name,
        }
        if row
        else None
    )


async def _resolve(session: AsyncSession, run_id: str):
    row = await _find_row(session, run_id)
    if not row:
        raise HTTPException(404, detail=f"run not found: {run_id}")
    return row, None


@router.get("/runs")
async def list_runs(
    state: str = "", market: str = "", session: AsyncSession = Depends(get_db)
) -> list:
    rows = await _read_rows(session)
    if state:
        rows = [r for r in rows if r.get("state") == state]
    if market:
        rows = [r for r in rows if r.get("market") == market]
    return rows


@router.get("/runs/{run_id:path}/state")
async def get_run_state(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    row, _ = await _resolve(session, run_id)
    return {"run_id": run_id, "state": row.get("state", "trained")}


@router.get("/runs/{run_id:path}/trades")
async def get_run_trades(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    await _resolve(session, run_id)  # 404 for unknown run, consistent with /state
    repo = RunTradeRepository(session)
    trades = await repo.get_by_run_id(run_id)
    return {
        "run_id": run_id,
        "trades": [
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date.isoformat() if t.entry_date else None,
                "exit_date": t.exit_date.isoformat() if t.exit_date else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                # intraday bar epochs (None for daily runs) -> markers on the exact 15m candle
                "entry_ts": _epoch(t.entry_time),
                "exit_ts": _epoch(t.exit_time),
                "holding_days": t.holding_days,
                "pnl_pct": t.pnl_pct,
                "direction": t.direction,
                "exit_reason": t.exit_reason,
            }
            for t in trades
        ],
        "total_trades": len(trades),
    }


@router.get("/runs/{run_id:path}/yearly-stats")
async def get_run_yearly_stats(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import select

    from stock_ml.db.models.yearly_stat import RunYearlyStatModel

    result = await session.execute(
        select(RunYearlyStatModel)
        .where(RunYearlyStatModel.run_id == run_id)
        .order_by(RunYearlyStatModel.year)
    )
    stats = result.scalars().all()
    return {
        "run_id": run_id,
        "yearly_stats": [
            {
                "year": s.year,
                "trades": s.trades,
                "win_rate": s.win_rate,
                "total_pnl": s.total_pnl,
                "max_drawdown": s.max_drawdown,
                "avg_pnl": s.avg_pnl,
                "med_pnl": s.med_pnl,
                "std_pnl": s.std_pnl,
                "max_win": s.max_win,
                "max_loss": s.max_loss,
                "profit_factor": s.profit_factor,
                "avg_hold": s.avg_hold,
            }
            for s in stats
        ],
    }


@router.get("/runs/{run_id:path}/signals")
async def get_run_signals(
    run_id: str,
    symbol: str | None = None,
    limit: int = 2000,
    session: AsyncSession = Depends(get_db),
) -> dict:
    from stock_ml.db.repositories.signal_repo import RunSignalRepository

    repo = RunSignalRepository(session)
    signals = await repo.get_by_run_id(run_id, symbol=symbol, limit=limit)
    return {
        "run_id": run_id,
        "signals": [
            {
                "symbol": s.symbol,
                "date": s.date.isoformat() if hasattr(s.date, "isoformat") else str(s.date),
                "signal": s.signal,
                "score": s.score,
                "exit_score": s.exit_score,
            }
            for s in signals
        ],
        "total_signals": len(signals),
    }


@router.get("/runs/{run_id:path}/symbol-stats")
async def get_run_symbol_stats(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import select

    from stock_ml.db.models.symbol_stat import RunSymbolStatModel

    result = await session.execute(
        select(RunSymbolStatModel)
        .where(RunSymbolStatModel.run_id == run_id)
        .order_by(RunSymbolStatModel.symbol)
    )
    stats = result.scalars().all()
    return {
        "run_id": run_id,
        "symbol_stats": [
            {
                "symbol": s.symbol,
                "trades": s.trades,
                "win_rate": s.win_rate,
                "total_pnl": s.total_pnl,
                "avg_pnl": s.avg_pnl,
                "med_pnl": s.med_pnl,
                "std_pnl": s.std_pnl,
                "max_win": s.max_win,
                "max_loss": s.max_loss,
                "profit_factor": s.profit_factor,
                "avg_hold": s.avg_hold,
            }
            for s in stats
        ],
    }


@router.patch("/runs/{run_id:path}/state")
async def set_run_state(run_id: str, body: dict, session: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import update

    row, _ = await _resolve(session, run_id)

    new_state = body.get("state", "").lower()
    if new_state not in ("trained", "pinned", "retired"):
        raise HTTPException(
            status_code=400, detail="state must be one of: trained, pinned, retired"
        )

    from stock_ml.db.models.run import LeaderboardRunModel

    await session.execute(
        update(LeaderboardRunModel)
        .where(LeaderboardRunModel.run_id == run_id)
        .values(state=new_state)
    )
    await session.commit()
    return {"run_id": run_id, "state": new_state}


@router.delete("/runs/{run_id:path}")
async def delete_run(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import delete

    row, _ = await _resolve(session, run_id)

    from stock_ml.db.models.run import LeaderboardRunModel

    result = await session.execute(
        delete(LeaderboardRunModel).where(LeaderboardRunModel.run_id == run_id)
    )
    await session.commit()
    return {"run_id": run_id, "deleted": result.rowcount > 0}
