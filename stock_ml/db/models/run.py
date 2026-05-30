from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base, TimestampMixin


class LeaderboardRunModel(Base, TimestampMixin):
    __tablename__ = "leaderboard_runs"

    __table_args__ = (
        CheckConstraint("state IN ('trained', 'pinned', 'retired')", name="ck_runs_state"),
        # Ranking query: score DESC, active only
        Index("idx_runs_score_state", "composite_score", "state", "superseded"),
        Index("idx_runs_market", "market"),
        Index("idx_runs_strategy", "strategy"),
        Index("idx_runs_feature_set", "feature_set"),
        Index("idx_runs_entry_model", "entry_model"),
        Index("idx_runs_timeframe", "timeframe"),
        Index("idx_runs_market_family", "market_family"),
        Index("idx_runs_fairness_group", "fairness_group_key"),
        Index("idx_runs_bundle_name", "bundle", "run_name", "generated_at"),
    )

    # --- Identity ---
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, index=True)
    bundle: Mapped[str] = mapped_column(String(255), nullable=False)
    run_name: Mapped[str] = mapped_column(String(255), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    superseded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # --- Lifecycle ---
    state: Mapped[str] = mapped_column(String(16), default="trained", nullable=False)
    cache_key_features: Mapped[str] = mapped_column(Text, default="", nullable=False)
    cache_key_predictions: Mapped[str] = mapped_column(Text, default="", nullable=False)
    artifact_trades_csv: Mapped[str] = mapped_column(Text, default="", nullable=False)
    artifact_meta_json: Mapped[str] = mapped_column(Text, default="", nullable=False)
    artifact_model_pkl: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # --- Strategy identity ---
    market: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    market_family: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    currency: Mapped[str] = mapped_column(String(16), default="unknown", nullable=False)
    pnl_mode: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    schema_ver: Mapped[str] = mapped_column(String(32), default="unknown", nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), default="unknown", nullable=False)
    strategy: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_set: Mapped[str] = mapped_column(String(255), nullable=False)
    entry_model: Mapped[str] = mapped_column(String(64), nullable=False)
    exit_model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    exit_model_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # --- Target (flat) ---
    target_type: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    target_forward_window: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    target_gain_threshold: Mapped[float | None] = mapped_column(Double, nullable=True)
    target_loss_threshold: Mapped[float | None] = mapped_column(Double, nullable=True)

    # --- Trading metrics ---
    trades: Mapped[int] = mapped_column(Integer, nullable=False)
    wr: Mapped[float] = mapped_column(Double, nullable=False)
    avg_pnl: Mapped[float] = mapped_column(Double, nullable=False)
    total_pnl: Mapped[float] = mapped_column(Double, nullable=False)
    pf: Mapped[float] = mapped_column(Double, nullable=False)
    avg_hold: Mapped[float] = mapped_column(Double, nullable=False)
    sharpe: Mapped[float] = mapped_column(Double, nullable=False)

    # --- Risk ---
    max_drawdown: Mapped[float] = mapped_column(Double, nullable=False)
    mdd_per_symbol: Mapped[float] = mapped_column(Double, nullable=False)
    yearly_consistency: Mapped[float] = mapped_column(Double, nullable=False)

    # --- Score ---
    composite_score: Mapped[float] = mapped_column(Double, nullable=False)
    score_mode: Mapped[str] = mapped_column(String(16), default="live", nullable=False)

    # --- Fairness ---
    n_symbols: Mapped[int] = mapped_column(Integer, nullable=False)
    first_test_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    last_test_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    backtest_window_key: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    cost_commission: Mapped[str] = mapped_column(Text, default="unknown", nullable=False)
    cost_tax: Mapped[str] = mapped_column(Text, default="unknown", nullable=False)
    cost_slippage: Mapped[str] = mapped_column(Text, default="unknown", nullable=False)
    fairness_group_key: Mapped[str] = mapped_column(String(40), nullable=False)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    same_symbols_as_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    same_window_as_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    same_cost_as_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    same_target_as_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    same_timeframe_as_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    same_market_family_as_baseline: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # --- Diagnostics ---
    warnings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # --- Versioning ---
    parent_run_id: Mapped[str | None] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # --- Relationships ---
    trades_list: Mapped[list[RunTradeModel]] = relationship(
        "RunTradeModel", back_populates="run", cascade="all, delete-orphan", lazy="noload"
    )
    experiment_config: Mapped[ExperimentConfigModel | None] = relationship(
        "ExperimentConfigModel",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="noload",
    )
    parent: Mapped[LeaderboardRunModel | None] = relationship(
        "LeaderboardRunModel",
        remote_side="LeaderboardRunModel.run_id",
        foreign_keys=[parent_run_id],
        lazy="noload",
    )
