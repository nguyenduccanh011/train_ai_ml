"""Signal model — signal history per symbol and date."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Date, DateTime, Double, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base

if TYPE_CHECKING:
    from stock_ml.db.models.run import LeaderboardRunModel


class RunSignalModel(Base):
    __tablename__ = "run_signals"

    __table_args__ = (UniqueConstraint("run_id", "symbol", "date", name="uq_signal"),)

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    date: Mapped[str] = mapped_column(Date(), nullable=False)  # ISO date string
    signal: Mapped[int] = mapped_column(Integer(), nullable=False)  # 1=buy, -1=sell, 0=neutral
    score: Mapped[float | None] = mapped_column(Double(), nullable=True)  # entry alpha
    # exit model prediction (dual-ML only; NULL for single-model runs)
    exit_score: Mapped[float | None] = mapped_column(Double(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.now
    )

    run: Mapped[LeaderboardRunModel] = relationship(
        "LeaderboardRunModel", back_populates="signals_list"
    )
