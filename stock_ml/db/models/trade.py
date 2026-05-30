from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, DateTime, Double, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base


class RunTradeModel(Base):
    __tablename__ = "run_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    entry_price: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    exit_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    holding_days: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    pnl_pct: Mapped[float] = mapped_column(Double, nullable=False)
    exit_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    entry_signal_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped["LeaderboardRunModel"] = relationship("LeaderboardRunModel", back_populates="trades_list")
