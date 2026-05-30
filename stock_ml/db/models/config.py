from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Double, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base


class ExperimentConfigModel(Base):
    __tablename__ = "experiment_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(512), ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"), nullable=False, unique=True
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    strategy: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    market: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    feature_set: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    entry_model_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    entry_model_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    exit_model: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    split_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    engine_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    signal_threshold: Mapped[Optional[float]] = mapped_column(Double, nullable=True)
    hypothesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_yaml: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped["LeaderboardRunModel"] = relationship("LeaderboardRunModel", back_populates="experiment_config")
