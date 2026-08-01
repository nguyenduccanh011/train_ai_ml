from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
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

if TYPE_CHECKING:
    from stock_ml.db.models.job import JobModel
    from stock_ml.db.models.signal import RunSignalModel
    from stock_ml.db.models.symbol_stat import RunSymbolStatModel
    from stock_ml.db.models.template import StrategyTemplateModel
    from stock_ml.db.models.trade import RunTradeModel
    from stock_ml.db.models.yearly_stat import RunYearlyStatModel


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
    model_mode: Mapped[str] = mapped_column(String(32), default="ml_only", nullable=False)
    signal_mode: Mapped[str] = mapped_column(String(32), default="entry_first", nullable=False)
    direction: Mapped[str] = mapped_column(String(8), default="long", nullable=False)

    # --- Target (flat) ---
    target_type: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    target_forward_window: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # --- Trading metrics ---
    trades: Mapped[int] = mapped_column(Integer, nullable=False)
    wr: Mapped[float] = mapped_column(Double, nullable=False)
    avg_pnl: Mapped[float] = mapped_column(Double, nullable=False)
    total_pnl: Mapped[float] = mapped_column(Double, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    pf: Mapped[float] = mapped_column(Double, nullable=False)
    avg_hold: Mapped[float] = mapped_column(Double, nullable=False)
    max_win: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    max_loss: Mapped[float] = mapped_column(Double, nullable=False, default=0.0)
    sharpe: Mapped[float] = mapped_column(Double, nullable=False)

    # --- Risk ---
    max_drawdown: Mapped[float] = mapped_column(Double, nullable=False)
    mdd_per_symbol: Mapped[float] = mapped_column(Double, nullable=False)
    yearly_consistency: Mapped[float] = mapped_column(Double, nullable=False)

    # --- Score ---
    composite_score: Mapped[float] = mapped_column(Double, nullable=False)
    score_mode: Mapped[str] = mapped_column(String(16), default="live", nullable=False)

    # --- Backtest scope / cost ---
    n_symbols: Mapped[int] = mapped_column(Integer, nullable=False)
    first_test_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    last_test_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    backtest_window_key: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    cost_commission: Mapped[str] = mapped_column(Text, default="unknown", nullable=False)
    cost_tax: Mapped[str] = mapped_column(Text, default="unknown", nullable=False)
    cost_slippage: Mapped[str] = mapped_column(Text, default="unknown", nullable=False)

    # --- Diagnostics ---
    warnings: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    # --- Backtest Transparency (for fair comparison) ---
    test_start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    test_end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    train_start_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    train_end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # --- Universe Tracking ---
    universe_slug: Mapped[str | None] = mapped_column(String(128), nullable=True)
    universe_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Execution Costs (already exist as TEXT columns from migration 0004) ---
    # cost_commission, cost_tax, cost_slippage are stored in existing TEXT columns

    # --- Experiment tracking (Phase 0-2 research pipeline) ---
    experiment_group: Mapped[str] = mapped_column(
        String(255), default="ungrouped", nullable=False, index=True
    )
    variant_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    metadata_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Versioning ---
    parent_run_id: Mapped[str | None] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # --- Template Source (Phase 0-3: DB-First Experiments) ---
    template_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("strategy_templates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    run_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    template_config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_config: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Reproducibility (Phase 4): lock the exact code + data + libs of a run ---
    git_sha: Mapped[str | None] = mapped_column(String(40), nullable=True)
    data_snapshot_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    lib_versions: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # --- Relationships ---
    trades_list: Mapped[list[RunTradeModel]] = relationship(
        "RunTradeModel", back_populates="run", cascade="all, delete-orphan", lazy="noload"
    )
    signals_list: Mapped[list[RunSignalModel]] = relationship(
        "RunSignalModel", back_populates="run", cascade="all, delete-orphan", lazy="noload"
    )
    parent: Mapped[LeaderboardRunModel | None] = relationship(
        "LeaderboardRunModel",
        remote_side="LeaderboardRunModel.run_id",
        foreign_keys=[parent_run_id],
        lazy="noload",
    )
    jobs_list: Mapped[list[JobModel]] = relationship(
        "JobModel", back_populates="run", cascade="all, delete-orphan", lazy="noload"
    )
    yearly_stats_list: Mapped[list[RunYearlyStatModel]] = relationship(
        "RunYearlyStatModel", back_populates="run", cascade="all, delete-orphan", lazy="noload"
    )
    symbol_stats_list: Mapped[list[RunSymbolStatModel]] = relationship(
        "RunSymbolStatModel", back_populates="run", cascade="all, delete-orphan", lazy="noload"
    )
    template: Mapped[StrategyTemplateModel | None] = relationship(
        "StrategyTemplateModel", lazy="noload", foreign_keys=[template_id]
    )
