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
    """B4 (PORTFOLIO_REGISTRATION_PIPELINE.md): prefer the PORTFOLIO layer
    (run_trades_overlay) when this run has one; fall back to engine BASE trades.
    `layer` tells the UI which one it is looking at."""
    await _resolve(session, run_id)  # 404 for unknown run, consistent with /state
    from sqlalchemy import text

    try:
        ov = (
            await session.execute(
                text(
                    "SELECT symbol, entry_date, exit_date, entry_price, exit_price, "
                    "holding_days, pnl_pct, exit_reason FROM run_trades_overlay "
                    "WHERE run_id=:rid ORDER BY entry_date"
                ),
                {"rid": run_id},
            )
        ).fetchall()
    except Exception:
        await session.rollback()
        ov = []
    if ov:
        return {
            "run_id": run_id,
            "layer": "overlay",
            "trades": [
                {
                    "symbol": t[0],
                    "entry_date": str(t[1]) if t[1] else None,
                    "exit_date": str(t[2]) if t[2] else None,
                    "entry_price": t[3],
                    "exit_price": t[4],
                    "entry_ts": None,
                    "exit_ts": None,
                    "holding_days": t[5],
                    "pnl_pct": t[6],
                    "direction": None,
                    "exit_reason": t[7],
                }
                for t in ov
            ],
            "total_trades": len(ov),
        }
    repo = RunTradeRepository(session)
    trades = await repo.get_by_run_id(run_id)
    return {
        "run_id": run_id,
        "layer": "base",
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


@router.get("/runs/{run_id:path}/portfolio/equity")
async def get_run_portfolio_equity(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    """Daily NAV / exposure / position-count timeline for the portfolio-execution overlay.
    Empty when the run has no persisted portfolio (only combo/portfolio runs populate run_equity)."""
    from sqlalchemy import text

    try:
        rows = (
            await session.execute(
                text(
                    "SELECT date, nav, exposure, n_positions FROM run_equity "
                    "WHERE run_id=:rid ORDER BY date"
                ),
                {"rid": run_id},
            )
        ).fetchall()
    except Exception:
        await session.rollback()
        return {"run_id": run_id, "equity": []}
    return {
        "run_id": run_id,
        "equity": [
            {"date": str(r[0]), "nav": r[1], "exposure": r[2], "n_positions": r[3]} for r in rows
        ],
    }


@router.get("/runs/{run_id:path}/portfolio/day")
async def get_run_portfolio_day(
    run_id: str, date: str, session: AsyncSession = Depends(get_db)
) -> dict:
    """Holdings (with ACTUAL weights) + entries + exits on a given date for the portfolio overlay."""
    from sqlalchemy import text

    try:
        # asyncpg binds date columns to datetime.date objects, not strings -> parse first.
        d_obj = datetime.strptime(date, "%Y-%m-%d").date()
        # B4: per-day trade enrichment reads the PORTFOLIO layer when persisted
        try:
            _has_ov = bool(
                (
                    await session.execute(
                        text("SELECT 1 FROM run_trades_overlay WHERE run_id=:rid LIMIT 1"),
                        {"rid": run_id},
                    )
                ).fetchone()
            )
        except Exception:
            await session.rollback()
            _has_ov = False
        _ttbl = "run_trades_overlay" if _has_ov else "run_trades"
        rows = (
            await session.execute(
                text(
                    "SELECT symbol, weight, entry_date, days_held, is_new, is_exit, exit_reason, conv, "
                    "entry_weight, unreal_pnl "
                    "FROM run_portfolio_daily WHERE run_id=:rid AND date=:d "
                    "ORDER BY is_exit, entry_weight DESC NULLS LAST"
                ),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchall()
        sig = (
            await session.execute(
                text(
                    "SELECT symbol, score FROM run_signals WHERE run_id=:rid AND date=:d "
                    "AND signal=1 ORDER BY score DESC NULLS LAST"
                ),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchall()
        # SELL orders for the NEXT session = positions held on D that ACTUALLY exit the next trading day
        # (exit decided from data up to D, fills at D+1). This reflects the real exit — which fires on a
        # sell signal ONLY when not vetoed (score3/regime/protect), and ALSO on non-signal rules
        # (max_hold / trailing / overext / hard_stop) or portfolio PREEMPTION — so it differs from the raw
        # sell-signal list. Causal: uses only the causally-decided exit, no future data.
        sells = (
            await session.execute(
                text(
                    f"SELECT symbol, exit_reason FROM {_ttbl} WHERE run_id=:rid "
                    "AND entry_date <= :d AND exit_date > :d "
                    "AND exit_date = (SELECT min(date) FROM run_equity WHERE run_id=:rid AND date > :d)"
                ),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchall()
        # PENDING pullback queue (CAUSAL resting book): buy signals still waiting to fill as of D.
        pend = (
            await session.execute(
                text(
                    "SELECT symbol, signal_date, days_waiting, limit_price, ref_price, pct_to_limit, "
                    "outcome, result_date FROM run_pending WHERE run_id=:rid AND date=:d "
                    "ORDER BY days_waiting DESC"
                ),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchall()
        # NAV on D (to express portfolio unrealized P&L as a fraction of NAV).
        nav_row = (
            await session.execute(
                text("SELECT nav FROM run_equity WHERE run_id=:rid AND date=:d"),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchone()
        # CLOSED trades exiting exactly on D -> realized P&L + buy/sell price + dates + sessions held.
        exit_tr = (
            await session.execute(
                text(
                    f"SELECT symbol, entry_date, entry_price, exit_price, pnl_pct, holding_days, exit_reason "
                    f"FROM {_ttbl} WHERE run_id=:rid AND exit_date=:d"
                ),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchall()
        # entry_price of positions OPEN on D (held) -> show buy price on the holdings table.
        open_tr = (
            await session.execute(
                text(
                    f"SELECT symbol, entry_date, entry_price FROM {_ttbl} "
                    f"WHERE run_id=:rid AND entry_date<=:d AND exit_date>:d"
                ),
                {"rid": run_id, "d": d_obj},
            )
        ).fetchall()
    except Exception:
        await session.rollback()
        return {
            "run_id": run_id,
            "date": date,
            "holdings": [],
            "entries": [],
            "exits": [],
            "signals": [],
            "sell_next": [],
            "pending": [],
            "unrealized": None,
        }
    # buy-price lookup for held positions (symbol -> entry_price), keyed by (symbol, entry_date).
    buy_px = {(o[0], str(o[1])): o[2] for o in open_tr}
    # full detail for trades closing today (symbol -> row).
    exit_detail = {e[0]: e for e in exit_tr}
    nav_d = float(nav_row[0]) if nav_row and nav_row[0] else None
    holdings, entries, exits = [], [], []
    unreal_val = 0.0  # sum of open positions' unrealized P&L in NAV units
    for r in rows:
        rec = {
            "symbol": r[0],
            "weight": r[1],
            "entry_date": str(r[2]) if r[2] else None,
            "days_held": r[3],  # now TRADING SESSIONS, not calendar days
            "is_new": bool(r[4]),
            "conv": r[7],
            "entry_weight": r[8],
            "unreal_pnl": r[9],  # % gain of this holding vs its buy price (unrealized)
            "entry_price": buy_px.get((r[0], str(r[2])) if r[2] else None),
        }
        if r[5]:  # is_exit -> enrich with realized P&L + buy/sell price + dates + sessions
            e = exit_detail.get(r[0])
            exits.append(
                {
                    "symbol": r[0],
                    "exit_reason": r[6],
                    "entry_date": str(e[1]) if e and e[1] else None,
                    "entry_price": e[2] if e else None,
                    "exit_price": e[3] if e else None,
                    "pnl_pct": e[4] if e else None,  # REALIZED P&L of the closed trade
                    "holding_days": e[5] if e else None,  # sessions held
                }
            )
        else:
            holdings.append(rec)
            # unrealized $ (in NAV units) = current value - cost = w*nav*(u/(1+u))
            if nav_d and r[1] is not None and r[9] is not None and (1.0 + r[9]) != 0:
                unreal_val += r[1] * nav_d * (r[9] / (1.0 + r[9]))
            if r[4]:
                entries.append(rec)
    held_syms = {h["symbol"] for h in holdings}
    signals = [{"symbol": s[0], "score": s[1], "filled": s[0] in held_syms} for s in sig]
    # actual next-session exits (symbol + reason: signal | max_hold | trailing_stop | overext | preempt | ...)
    sell_next = [{"symbol": s[0], "exit_reason": s[1]} for s in sells]
    # PORTFOLIO-LEVEL unrealized P&L (open positions only, NOT yet realized into cash).
    unrealized = None
    if nav_d:
        pos_value = sum(h["weight"] * nav_d for h in holdings if h["weight"] is not None)
        cost = pos_value - unreal_val
        unrealized = {
            "nav": nav_d,
            "unreal_pnl_nav_pct": (unreal_val / nav_d)
            if nav_d
            else None,  # unrealized as % of total NAV
            "unreal_pnl_cost_pct": (unreal_val / cost)
            if cost
            else None,  # unrealized as % of money invested
            "invested_value": pos_value,  # current MTM value of holdings
            "invested_cost": cost,  # what was paid for them
        }
    pending = [
        {
            "symbol": p[0],
            "signal_date": str(p[1]) if p[1] else None,
            "days_waiting": p[2],
            "limit_price": p[3],
            "ref_price": p[4],
            "pct_to_limit": p[5],
            "outcome": p[6],
            "result_date": str(p[7]) if p[7] else None,
        }
        for p in pend
    ]
    return {
        "run_id": run_id,
        "date": date,
        "holdings": holdings,
        "entries": entries,
        "exits": exits,
        "signals": signals,
        "sell_next": sell_next,
        "pending": pending,
        "unrealized": unrealized,
    }


# Exit reasons that only the PORTFOLIO overlay produces (slot-preemption / green-trail /
# early-cut levers). run_trades rows carrying these are overlay OUTPUT, not engine BASE
# trades — the provenance detector of PORTFOLIO_REGISTRATION_PIPELINE.md (B2/B7).
_OVERLAY_EXIT_REASONS = ("preempt", "green_trail", "early_cut")


@router.get("/runs/{run_id:path}/signal-symbols")
async def get_run_signal_symbols(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    """Per-symbol FIRED-signal counts from run_signals — the run's full signal universe.
    A symbol can fire many signals yet never reach the (portfolio) trade list; sidebars built
    from trades/symbol-stats hide it entirely. Empty when the run persisted no signals."""
    from sqlalchemy import text

    try:
        rows = (
            await session.execute(
                text(
                    "SELECT symbol, "
                    "count(*) FILTER (WHERE signal = 1) AS buys, "
                    "count(*) FILTER (WHERE signal = -1) AS sells "
                    "FROM run_signals WHERE run_id=:rid GROUP BY symbol "
                    "HAVING count(*) FILTER (WHERE signal <> 0) > 0 "
                    "ORDER BY 2 DESC, 1"
                ),
                {"rid": run_id},
            )
        ).fetchall()
    except Exception:
        await session.rollback()
        return {"run_id": run_id, "symbols": []}
    return {
        "run_id": run_id,
        "symbols": [{"symbol": r[0], "buys": r[1], "sells": r[2]} for r in rows],
    }


@router.get("/runs/{run_id:path}/symbol/{symbol}/portfolio-fate")
async def get_symbol_portfolio_fate(
    run_id: str, symbol: str, session: AsyncSession = Depends(get_db)
) -> dict:
    """B7 (PORTFOLIO_REGISTRATION_PIPELINE.md): one row per fired BUY signal of `symbol`,
    joined to its fate — became a base trade? filled into the portfolio? skipped why?

    Layer-aware: if this run's run_trades rows carry overlay-only exit reasons
    (pre-B2 OUTPUT runs, e.g. _static900) they ARE the portfolio trades → became_base
    is unknowable (null). Post-B2, base lives in run_trades and the portfolio layer in
    run_trades_overlay; either table may be absent/empty and the endpoint degrades to
    nulls instead of failing. Guard: n_signals == no_base + base (or filled when layer
    is overlay)."""
    from sqlalchemy import text

    async def _rows(sql: str, params: dict) -> list:
        try:
            return (await session.execute(text(sql), params)).fetchall()
        except Exception:
            await session.rollback()
            return []

    p = {"rid": run_id, "sym": symbol}
    sig = await _rows(
        "SELECT date, score FROM run_signals "
        "WHERE run_id=:rid AND symbol=:sym AND signal=1 ORDER BY date",
        p,
    )
    base = await _rows(
        "SELECT entry_signal_date, entry_date, exit_date, pnl_pct, exit_reason "
        "FROM run_trades WHERE run_id=:rid AND symbol=:sym ORDER BY entry_date",
        p,
    )
    overlay = await _rows(
        "SELECT entry_date, exit_date, pnl_pct, exit_reason "
        "FROM run_trades_overlay WHERE run_id=:rid AND symbol=:sym ORDER BY entry_date",
        p,
    )
    skipped = await _rows(
        "SELECT signal_date, entry_date, skip_reason FROM run_skipped "
        "WHERE run_id=:rid AND symbol=:sym",
        p,
    )

    # Layer is a property of the RUN, not the symbol — a signals-only symbol has no
    # run_trades rows of its own, yet its run may still be an OUTPUT run.
    run_is_output = bool(
        await _rows(
            "SELECT 1 FROM run_trades WHERE run_id=:rid "
            "AND exit_reason IN ('preempt', 'green_trail', 'early_cut') LIMIT 1",
            {"rid": run_id},
        )
    )
    layer = "overlay" if run_is_output else "base"
    base_by_sig = {str(r[0]): r for r in base if r[0] is not None}
    overlay_entry_dates = {str(r[0]) for r in overlay}
    skip_by_sig = {str(r[0]): r[2] for r in skipped if r[0] is not None}
    skip_by_entry = {str(r[1]): r[2] for r in skipped if r[1] is not None}

    out, skip_hist = [], {}
    n_base = n_filled = n_no_base = n_skipped = 0
    for d, score in sig:
        ds = str(d)
        t = base_by_sig.get(ds)
        skip_reason = skip_by_sig.get(ds) or (t and skip_by_entry.get(str(t[1])))
        if layer == "overlay":
            # run_trades rows ARE portfolio trades; base layer was never persisted.
            became_base, filled = None, t is not None
        else:
            became_base = t is not None
            if not t:
                filled = False
            elif overlay:
                filled = str(t[1]) in overlay_entry_dates
            else:
                filled = None  # base trade exists but no overlay layer persisted yet
        if became_base:
            n_base += 1
        if not t:
            n_no_base += 1
        if filled:
            n_filled += 1
        if skip_reason:
            n_skipped += 1
            skip_hist[skip_reason] = skip_hist.get(skip_reason, 0) + 1
        out.append(
            {
                "signal_date": ds,
                "score": score,
                "became_base_trade": became_base,
                "base_entry_date": str(t[1]) if t else None,
                "filled": filled,
                "exit_reason": t[4] if t else None,
                "pnl_pct": t[3] if t else None,
                "skip_reason": skip_reason,
            }
        )

    return {
        "run_id": run_id,
        "symbol": symbol,
        "layer_run_trades": layer,
        "has_overlay_table": bool(overlay),
        "counts": {
            "signals": len(sig),
            "base_trades": (n_base if layer == "base" else None),
            "portfolio_trades": n_filled,
            "skipped": n_skipped,
            "no_base": n_no_base,
        },
        "skip_reasons": skip_hist,
        "rows": out,
    }


@router.get("/runs/{run_id:path}/skipped")
async def get_run_skipped(run_id: str, session: AsyncSession = Depends(get_db)) -> dict:
    """Opportunity-cost ledger: signals the portfolio DID NOT take (conv-skip / capacity) with the base
    backtest's REALIZED return — 'was skipping right?'. Empty when the run has no skipped ledger."""
    from sqlalchemy import text

    try:
        rows = (
            await session.execute(
                text(
                    "SELECT symbol, signal_date, entry_date, pnl_pct, conv, skip_reason "
                    "FROM run_skipped WHERE run_id=:rid ORDER BY signal_date DESC NULLS LAST"
                ),
                {"rid": run_id},
            )
        ).fetchall()
    except Exception:
        await session.rollback()
        return {"run_id": run_id, "skipped": []}
    return {
        "run_id": run_id,
        "skipped": [
            {
                "symbol": r[0],
                "signal_date": str(r[1]) if r[1] else None,
                "entry_date": str(r[2]) if r[2] else None,
                "pnl_pct": r[3],
                "conv": r[4],
                "skip_reason": r[5],
            }
            for r in rows
        ],
    }


_LIFECYCLE_STATES = ("trained", "pinned", "retired")


@router.post("/runs/bulk-state")
async def bulk_set_run_state(body: dict, session: AsyncSession = Depends(get_db)) -> dict:
    """Change lifecycle state for every run matching a current-state filter.

    Body: ``{"state": "<new>", "filter": {"current_state": "<current>"}}``.
    Registered before the ``/runs/{run_id:path}`` routes so ``bulk-state`` is not
    swallowed as a run id by the catch-all.
    """
    from sqlalchemy import update

    new_state = str(body.get("state", "")).lower()
    current_state = str((body.get("filter") or {}).get("current_state", "")).lower()
    if new_state not in _LIFECYCLE_STATES or current_state not in _LIFECYCLE_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"state and filter.current_state must each be one of: {', '.join(_LIFECYCLE_STATES)}",
        )

    from stock_ml.db.models.run import LeaderboardRunModel

    result = await session.execute(
        update(LeaderboardRunModel)
        .where(LeaderboardRunModel.state == current_state)
        .values(state=new_state)
    )
    await session.commit()
    return {"state": new_state, "updated": result.rowcount}


@router.delete("/runs/bulk")
async def bulk_delete_runs(body: dict, session: AsyncSession = Depends(get_db)) -> dict:
    """Delete every retired run from the leaderboard (DB-first: drops the rows).

    Body: ``{"state": "retired", "confirm": true}``. Restricted to ``retired`` so an
    errant call cannot mass-delete active models. Registered before
    ``/runs/{run_id:path}`` so ``bulk`` is not treated as a run id.
    """
    from sqlalchemy import delete

    if not body.get("confirm"):
        raise HTTPException(status_code=400, detail="confirm must be true for bulk delete")
    state = str(body.get("state", "")).lower()
    if state != "retired":
        raise HTTPException(status_code=400, detail="bulk delete is only allowed for state=retired")

    from stock_ml.db.models.run import LeaderboardRunModel

    result = await session.execute(
        delete(LeaderboardRunModel).where(LeaderboardRunModel.state == state)
    )
    await session.commit()
    # DB-first: this drops leaderboard rows only; on-disk artifacts/cache are
    # reclaimed separately by the GC sweep, so no bytes are freed here.
    return {"deleted": result.rowcount, "freed_mb": 0}


@router.patch("/runs/{run_id:path}/state")
async def set_run_state(run_id: str, body: dict, session: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import update

    row, _ = await _resolve(session, run_id)

    new_state = body.get("state", "").lower()
    if new_state not in _LIFECYCLE_STATES:
        raise HTTPException(
            status_code=400, detail=f"state must be one of: {', '.join(_LIFECYCLE_STATES)}"
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
