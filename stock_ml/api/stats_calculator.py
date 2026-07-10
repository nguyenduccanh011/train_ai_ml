"""Calculate and populate yearly/symbol stats from trades."""

from __future__ import annotations

from statistics import mean, median, stdev
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.models.symbol_stat import RunSymbolStatModel
from stock_ml.db.models.trade import RunTradeModel
from stock_ml.db.models.yearly_stat import RunYearlyStatModel


async def calculate_yearly_stats_from_trades(
    session: AsyncSession, run_id: str, trades: list[RunTradeModel]
) -> list[dict[str, Any]]:
    """Calculate yearly stats from trades, populate DB if empty."""
    if not trades:
        return []

    # Group trades by year
    yearly = {}
    for t in trades:
        year = t.entry_date.year if t.entry_date else 2020
        if year not in yearly:
            yearly[year] = []
        yearly[year].append(t)

    # Calculate stats for each year
    result = []
    for year in sorted(yearly.keys()):
        year_trades = yearly[year]
        pnls = [float(t.pnl_pct) for t in year_trades if t.pnl_pct is not None]

        if not pnls:
            continue

        wins = sum(1 for p in pnls if p >= 0)
        losses = sum(1 for p in pnls if p < 0)

        trade_count = len(year_trades)
        win_rate = wins / trade_count if trade_count > 0 else 0
        total_pnl = sum(pnls)
        avg_pnl = mean(pnls)
        med_pnl = median(pnls)
        std_pnl = stdev(pnls) if len(pnls) > 1 else 0.0
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0

        # Profit factor = sum of wins / abs(sum of losses)
        sum_wins = sum(p for p in pnls if p >= 0)
        sum_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else (1.0 if sum_wins > 0 else 0.0)

        # Average hold days
        hold_days = [float(t.holding_days) for t in year_trades if t.holding_days is not None]
        avg_hold = mean(hold_days) if hold_days else None

        # Drawdown calculation (simplified: max loss from peak)
        cumsum = []
        total = 0
        for p in pnls:
            total += p
            cumsum.append(total)

        max_drawdown = 0.0
        if cumsum:
            for i, cum in enumerate(cumsum):
                if i == 0:
                    peak = cum
                else:
                    peak = max(cumsum[: i + 1])
                dd = (cum - peak) / (peak + 1e-6) if peak != 0 else 0
                max_drawdown = min(max_drawdown, dd)

        stat = {
            "year": year,
            "trades": trade_count,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "avg_pnl": avg_pnl,
            "med_pnl": med_pnl,
            "std_pnl": std_pnl,
            "max_win": max_win,
            "max_loss": max_loss,
            "profit_factor": profit_factor,
            "avg_hold": avg_hold,
        }
        result.append(stat)

        # Update or insert into DB
        existing = await session.execute(
            select(RunYearlyStatModel).where(
                (RunYearlyStatModel.run_id == run_id) & (RunYearlyStatModel.year == year)
            )
        )
        db_stat = existing.scalar_one_or_none()

        if db_stat:
            # Update
            for key, value in stat.items():
                if key != "year":
                    setattr(db_stat, key, value)
        else:
            # Insert
            db_stat = RunYearlyStatModel(run_id=run_id, **stat)
            session.add(db_stat)

    await session.commit()
    return result


async def calculate_symbol_stats_from_trades(
    session: AsyncSession, run_id: str, trades: list[RunTradeModel]
) -> list[dict[str, Any]]:
    """Calculate symbol stats from trades, populate DB if empty."""
    if not trades:
        return []

    # Group trades by symbol
    symbols = {}
    for t in trades:
        symbol = t.symbol
        if symbol not in symbols:
            symbols[symbol] = []
        symbols[symbol].append(t)

    # Calculate stats for each symbol
    result = []
    for symbol in sorted(symbols.keys()):
        symbol_trades = symbols[symbol]
        pnls = [float(t.pnl_pct) for t in symbol_trades if t.pnl_pct is not None]

        if not pnls:
            continue

        wins = sum(1 for p in pnls if p >= 0)
        trade_count = len(symbol_trades)
        win_rate = wins / trade_count if trade_count > 0 else 0
        total_pnl = sum(pnls)
        avg_pnl = mean(pnls)
        med_pnl = median(pnls)
        std_pnl = stdev(pnls) if len(pnls) > 1 else 0.0
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0

        sum_wins = sum(p for p in pnls if p >= 0)
        sum_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = sum_wins / sum_losses if sum_losses > 0 else (1.0 if sum_wins > 0 else 0.0)

        hold_days = [float(t.holding_days) for t in symbol_trades if t.holding_days is not None]
        avg_hold = mean(hold_days) if hold_days else None

        stat = {
            "symbol": symbol,
            "trades": trade_count,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "med_pnl": med_pnl,
            "std_pnl": std_pnl,
            "max_win": max_win,
            "max_loss": max_loss,
            "profit_factor": profit_factor,
            "avg_hold": avg_hold,
        }
        result.append(stat)

        # Update or insert into DB
        existing = await session.execute(
            select(RunSymbolStatModel).where(
                (RunSymbolStatModel.run_id == run_id) & (RunSymbolStatModel.symbol == symbol)
            )
        )
        db_stat = existing.scalar_one_or_none()

        if db_stat:
            # Update
            for key, value in stat.items():
                if key != "symbol":
                    setattr(db_stat, key, value)
        else:
            # Insert
            db_stat = RunSymbolStatModel(run_id=run_id, **stat)
            session.add(db_stat)

    await session.commit()
    return result
