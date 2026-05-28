"""Trade statistics: aggregate, per-year, per-day.

All inputs are trades DataFrames produced by `trades_to_dataframe`.
All outputs are vanilla DataFrames so they round-trip to CSV/JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _basic_block(pnls: pd.Series, holds: pd.Series | None = None) -> dict:
    if len(pnls) == 0:
        return {
            "n_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "med_pnl": 0.0,
            "std_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "profit_factor": 0.0,
            "avg_hold_days": 0.0,
        }
    wins = int((pnls > 0).sum())
    losses = int((pnls <= 0).sum())
    gp = float(pnls[pnls > 0].sum())
    gl = float(-pnls[pnls < 0].sum())
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return {
        "n_trades": int(len(pnls)),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(pnls),
        "total_pnl": float(pnls.sum()),
        "avg_pnl": float(pnls.mean()),
        "med_pnl": float(pnls.median()),
        "std_pnl": float(pnls.std(ddof=0)),
        "max_win": float(pnls.max()),
        "max_loss": float(pnls.min()),
        "profit_factor": float(pf) if np.isfinite(pf) else 999.0,
        "avg_hold_days": float(holds.mean()) if holds is not None and len(holds) else 0.0,
    }


def aggregate_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return _basic_block(pd.Series(dtype=float))
    return _basic_block(trades["pnl_pct"], trades.get("holding_days"))


def per_year_stats(trades: pd.DataFrame, date_col: str = "exit_date") -> pd.DataFrame:
    """One row per (year). PnL attributed by trade exit year by default."""
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    rows = []
    for year, g in df.groupby("year"):
        rec = {"year": int(year), **_basic_block(g["pnl_pct"], g.get("holding_days"))}
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def per_day_stats(trades: pd.DataFrame, date_col: str = "exit_date") -> pd.DataFrame:
    """One row per calendar day on which at least one trade was attributed.

    Attribution is by `date_col` (default: exit_date — the day PnL is realized).
    Pass `entry_date` to see opening activity instead.
    """
    if trades.empty:
        return pd.DataFrame(
            columns=["date", "n_trades", "wins", "losses", "win_rate", "total_pnl", "avg_pnl"]
        )
    df = trades.copy()
    df["day"] = pd.to_datetime(df[date_col]).dt.normalize()
    rows = []
    for day, g in df.groupby("day"):
        block = _basic_block(g["pnl_pct"], g.get("holding_days"))
        rows.append(
            {
                "date": day.date().isoformat(),
                "n_trades": block["n_trades"],
                "wins": block["wins"],
                "losses": block["losses"],
                "win_rate": round(block["win_rate"], 4),
                "total_pnl": round(block["total_pnl"], 6),
                "avg_pnl": round(block["avg_pnl"], 6),
                "max_win": round(block["max_win"], 6),
                "max_loss": round(block["max_loss"], 6),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def per_symbol_stats(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for sym, g in trades.groupby("symbol"):
        rec = {"symbol": str(sym), **_basic_block(g["pnl_pct"], g.get("holding_days"))}
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
