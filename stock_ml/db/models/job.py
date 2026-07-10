"""Job model for async task tracking."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base

if TYPE_CHECKING:
    from stock_ml.db.models.run import LeaderboardRunModel


class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)  # train, retrain, gc_sweep
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending"
    )  # pending, running, done, failed
    run_id: Mapped[str | None] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text(), nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON(), nullable=True)

    run: Mapped[LeaderboardRunModel] = relationship(
        "LeaderboardRunModel", back_populates="jobs_list"
    )
