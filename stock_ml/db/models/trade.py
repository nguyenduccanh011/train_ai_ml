from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Double, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base

if TYPE_CHECKING:
    from stock_ml.db.models.run import LeaderboardRunModel


class RunTradeModel(Base):
    __tablename__ = "run_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Double, nullable=True)
    exit_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Double, nullable=True)
    # Intraday bar timestamps (NULL for daily-grain runs); let the chart place
    # buy/sell markers on the exact 15m candle for intraday strategies.
    entry_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    holding_days: Mapped[float | None] = mapped_column(Double, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Double, nullable=False)
    # Position side: 'long' (mua) / 'short' (bán khống) / NULL for legacy long-only daily runs.
    # Two-sided strategies (e.g. the VN30-futures intraday model) need this so the model-details
    # table can label a SHORT (sell-high entry, buy-back-low exit) instead of it reading as buy-high/sell-low.
    direction: Mapped[str | None] = mapped_column(String(8), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entry_signal_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped[LeaderboardRunModel] = relationship(
        "LeaderboardRunModel", back_populates="trades_list"
    )

    __table_args__ = (
        UniqueConstraint("run_id", "symbol", "entry_date", name="uq_run_symbol_entry"),
    )
