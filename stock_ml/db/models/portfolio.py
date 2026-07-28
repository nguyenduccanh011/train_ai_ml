"""Portfolio-execution overlay models — daily equity/exposure timeline + per-day holdings with ACTUAL
weights. Populated by a portfolio-execution layer (e.g. the preempt + conviction-sizing + conv-skip combo
champion) so the detail-page 'Danh mục' tab can show the day-by-day portfolio, real weights, and the
entry/exit list. Separate from run_trades (which is per-symbol, equal-weight, no daily snapshot)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Date, DateTime, Double, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from stock_ml.db.base import Base


class RunEquityModel(Base):
    """Daily NAV / exposure / position-count timeline for a portfolio-execution run."""

    __tablename__ = "run_equity"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[str] = mapped_column(Date(), nullable=False)
    nav: Mapped[float] = mapped_column(Double(), nullable=False)  # NAV as multiple of start capital
    cash: Mapped[float | None] = mapped_column(Double(), nullable=True)
    exposure: Mapped[float | None] = mapped_column(Double(), nullable=True)  # invested / nav
    n_positions: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )


class RunPortfolioDailyModel(Base):
    """One row per (date, held position). weight = position value / NAV that day. Exit events are rows
    with is_exit=True (weight 0) carrying the exit_reason ('signal' | 'preempt' | ...)."""

    __tablename__ = "run_portfolio_daily"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[str] = mapped_column(Date(), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    weight: Mapped[float | None] = mapped_column(Double(), nullable=True)  # current market value / nav (drifts)
    entry_weight: Mapped[float | None] = mapped_column(Double(), nullable=True)  # FIXED allocation at entry (invested / nav_at_entry)
    unreal_pnl: Mapped[float | None] = mapped_column(Double(), nullable=True)  # mark-to-market P&L of this holding as of the day
    entry_date: Mapped[str | None] = mapped_column(Date(), nullable=True)
    days_held: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    is_new: Mapped[bool | None] = mapped_column(Boolean(), nullable=True)  # entered today
    is_exit: Mapped[bool | None] = mapped_column(Boolean(), nullable=True)  # closing today
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    conv: Mapped[float | None] = mapped_column(Double(), nullable=True)  # conviction (cs4) at entry
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )


class RunSkippedModel(Base):
    """Signals the portfolio DID NOT take (dropped by conv-skip / capacity) with the base trade's REALIZED
    outcome — the opportunity-cost ledger ('was skipping right?'). pnl_pct is the base backtest return the
    name would have earned."""

    __tablename__ = "run_skipped"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    signal_date: Mapped[str | None] = mapped_column(Date(), nullable=True)
    entry_date: Mapped[str | None] = mapped_column(Date(), nullable=True)
    pnl_pct: Mapped[float | None] = mapped_column(Double(), nullable=True)  # base backtest return if taken
    conv: Mapped[float | None] = mapped_column(Double(), nullable=True)
    skip_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 'conv_skip' | 'capacity'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )


class RunPendingModel(Base):
    """CAUSAL resting pullback order book: for each date, the buy signals still WAITING to fill (limit at
    close[signal]*(1-pct), within the pullback window, not yet filled, not held). outcome/result_date are
    POST-HOC (fill vs expire) shown as reference — the row itself is the causal state as of `date`."""

    __tablename__ = "run_pending"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[str] = mapped_column(Date(), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    signal_date: Mapped[str | None] = mapped_column(Date(), nullable=True)  # bar the buy signal fired
    days_waiting: Mapped[int | None] = mapped_column(Integer(), nullable=True)  # trading bars since signal
    limit_price: Mapped[float | None] = mapped_column(Double(), nullable=True)  # pullback limit target
    ref_price: Mapped[float | None] = mapped_column(Double(), nullable=True)  # close on `date`
    pct_to_limit: Mapped[float | None] = mapped_column(Double(), nullable=True)  # (limit/ref - 1); <=0 = at/through
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)  # 'fill' | 'expire' (post-hoc)
    result_date: Mapped[str | None] = mapped_column(Date(), nullable=True)  # fill/expire date (post-hoc)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )
