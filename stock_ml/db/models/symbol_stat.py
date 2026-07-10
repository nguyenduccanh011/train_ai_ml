"""Symbol statistics model — per-symbol breakdown of run performance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Double, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base

if TYPE_CHECKING:
    from stock_ml.db.models.run import LeaderboardRunModel


class RunSymbolStatModel(Base):
    __tablename__ = "run_symbol_stats"

    __table_args__ = (UniqueConstraint("run_id", "symbol", name="uq_run_symbol"),)

    id: Mapped[int] = mapped_column(Integer(), primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    trades: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    win_rate: Mapped[float | None] = mapped_column(Double(), nullable=True)
    total_pnl: Mapped[float | None] = mapped_column(Double(), nullable=True)
    avg_pnl: Mapped[float | None] = mapped_column(Double(), nullable=True)
    med_pnl: Mapped[float | None] = mapped_column(Double(), nullable=True)
    std_pnl: Mapped[float | None] = mapped_column(Double(), nullable=True)
    max_win: Mapped[float | None] = mapped_column(Double(), nullable=True)
    max_loss: Mapped[float | None] = mapped_column(Double(), nullable=True)
    profit_factor: Mapped[float | None] = mapped_column(Double(), nullable=True)
    avg_hold: Mapped[float | None] = mapped_column(Double(), nullable=True)

    run: Mapped[LeaderboardRunModel] = relationship(
        "LeaderboardRunModel", back_populates="symbol_stats_list"
    )
