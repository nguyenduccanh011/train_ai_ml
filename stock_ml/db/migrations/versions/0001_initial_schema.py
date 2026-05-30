"""initial_schema

Revision ID: 0001
Revises:
Create Date: 2026-05-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # leaderboard_runs
    # ------------------------------------------------------------------ #
    op.create_table(
        "leaderboard_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(512), nullable=False),
        sa.Column("bundle", sa.String(255), nullable=False),
        sa.Column("run_name", sa.String(255), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("superseded", sa.Boolean(), nullable=False, server_default="false"),
        # Lifecycle
        sa.Column("state", sa.String(16), nullable=False, server_default="trained"),
        sa.Column("cache_key_features", sa.Text(), nullable=False, server_default=""),
        sa.Column("cache_key_predictions", sa.Text(), nullable=False, server_default=""),
        sa.Column("artifact_trades_csv", sa.Text(), nullable=False, server_default=""),
        sa.Column("artifact_meta_json", sa.Text(), nullable=False, server_default=""),
        sa.Column("artifact_model_pkl", sa.Text(), nullable=False, server_default=""),
        # Strategy identity
        sa.Column("market", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("market_family", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("currency", sa.String(16), nullable=False, server_default="unknown"),
        sa.Column("pnl_mode", sa.String(32), nullable=False, server_default="unknown"),
        sa.Column("schema_ver", sa.String(32), nullable=False, server_default="unknown"),
        sa.Column("timeframe", sa.String(8), nullable=False, server_default="unknown"),
        sa.Column("strategy", sa.String(255), nullable=False),
        sa.Column("feature_set", sa.String(255), nullable=False),
        sa.Column("entry_model", sa.String(64), nullable=False),
        sa.Column("exit_model_type", sa.String(64), nullable=False),
        sa.Column("exit_model_enabled", sa.Boolean(), nullable=False, server_default="false"),
        # Target (flat)
        sa.Column("target_type", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("target_forward_window", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("target_gain_threshold", sa.Double(), nullable=True),
        sa.Column("target_loss_threshold", sa.Double(), nullable=True),
        # Trading metrics
        sa.Column("trades", sa.Integer(), nullable=False),
        sa.Column("wr", sa.Double(), nullable=False),
        sa.Column("avg_pnl", sa.Double(), nullable=False),
        sa.Column("total_pnl", sa.Double(), nullable=False),
        sa.Column("pf", sa.Double(), nullable=False),
        sa.Column("avg_hold", sa.Double(), nullable=False),
        sa.Column("sharpe", sa.Double(), nullable=False),
        # Risk
        sa.Column("max_drawdown", sa.Double(), nullable=False),
        sa.Column("mdd_per_symbol", sa.Double(), nullable=False),
        sa.Column("yearly_consistency", sa.Double(), nullable=False),
        # Score
        sa.Column("composite_score", sa.Double(), nullable=False),
        sa.Column("score_mode", sa.String(16), nullable=False, server_default="live"),
        # Fairness
        sa.Column("n_symbols", sa.Integer(), nullable=False),
        sa.Column("first_test_year", sa.SmallInteger(), nullable=False),
        sa.Column("last_test_year", sa.SmallInteger(), nullable=False),
        sa.Column("backtest_window_key", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("cost_commission", sa.Text(), nullable=False, server_default="unknown"),
        sa.Column("cost_tax", sa.Text(), nullable=False, server_default="unknown"),
        sa.Column("cost_slippage", sa.Text(), nullable=False, server_default="unknown"),
        sa.Column("fairness_group_key", sa.String(40), nullable=False),
        sa.Column("is_baseline", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("same_symbols_as_baseline", sa.Boolean(), nullable=True),
        sa.Column("same_window_as_baseline", sa.Boolean(), nullable=True),
        sa.Column("same_cost_as_baseline", sa.Boolean(), nullable=True),
        sa.Column("same_target_as_baseline", sa.Boolean(), nullable=True),
        sa.Column("same_timeframe_as_baseline", sa.Boolean(), nullable=True),
        sa.Column("same_market_family_as_baseline", sa.Boolean(), nullable=True),
        # Diagnostics
        sa.Column("warnings", sa.JSON(), nullable=False, server_default="[]"),
        # Versioning
        sa.Column("parent_run_id", sa.String(512), nullable=True),
        # Timestamps
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_runs_run_id"),
        sa.CheckConstraint("state IN ('trained', 'pinned', 'retired')", name="ck_runs_state"),
        sa.ForeignKeyConstraint(
            ["parent_run_id"],
            ["leaderboard_runs.run_id"],
            ondelete="SET NULL",
            name="fk_runs_parent",
        ),
    )

    # Indexes for leaderboard_runs
    op.create_index("idx_runs_run_id", "leaderboard_runs", ["run_id"], unique=True)
    op.create_index(
        "idx_runs_score_state", "leaderboard_runs", ["composite_score", "state", "superseded"]
    )
    op.create_index("idx_runs_market", "leaderboard_runs", ["market"])
    op.create_index("idx_runs_strategy", "leaderboard_runs", ["strategy"])
    op.create_index("idx_runs_feature_set", "leaderboard_runs", ["feature_set"])
    op.create_index("idx_runs_entry_model", "leaderboard_runs", ["entry_model"])
    op.create_index("idx_runs_timeframe", "leaderboard_runs", ["timeframe"])
    op.create_index("idx_runs_market_family", "leaderboard_runs", ["market_family"])
    op.create_index("idx_runs_fairness_group", "leaderboard_runs", ["fairness_group_key"])
    op.create_index(
        "idx_runs_bundle_name", "leaderboard_runs", ["bundle", "run_name", "generated_at"]
    )
    op.create_index(
        "idx_runs_pinned_market",
        "leaderboard_runs",
        ["market", "composite_score"],
        postgresql_where=sa.text("state = 'pinned' AND superseded = false"),
    )
    op.create_index(
        "idx_runs_parent",
        "leaderboard_runs",
        ["parent_run_id"],
        postgresql_where=sa.text("parent_run_id IS NOT NULL"),
    )

    # ------------------------------------------------------------------ #
    # run_trades
    # ------------------------------------------------------------------ #
    op.create_table(
        "run_trades",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(512), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("entry_date", sa.Date(), nullable=True),
        sa.Column("entry_price", sa.Double(), nullable=True),
        sa.Column("exit_date", sa.Date(), nullable=True),
        sa.Column("exit_price", sa.Double(), nullable=True),
        sa.Column("holding_days", sa.Double(), nullable=True),
        sa.Column("pnl_pct", sa.Double(), nullable=False),
        sa.Column("exit_reason", sa.String(64), nullable=True),
        sa.Column("entry_signal_date", sa.Date(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["run_id"], ["leaderboard_runs.run_id"], ondelete="CASCADE", name="fk_trades_run"
        ),
    )
    op.create_index("idx_trades_run_id", "run_trades", ["run_id"])
    op.create_index("idx_trades_symbol_date", "run_trades", ["symbol", "entry_date"])

    # ------------------------------------------------------------------ #
    # experiment_configs
    # ------------------------------------------------------------------ #
    op.create_table(
        "experiment_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(512), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("strategy", sa.String(255), nullable=True),
        sa.Column("market", sa.String(64), nullable=True),
        sa.Column("feature_set", sa.String(255), nullable=True),
        sa.Column("entry_model_type", sa.String(64), nullable=True),
        sa.Column("entry_model_params", sa.JSON(), nullable=True),
        sa.Column("exit_model", sa.JSON(), nullable=True),
        sa.Column("split_config", sa.JSON(), nullable=True),
        sa.Column("engine_config", sa.JSON(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("signal_threshold", sa.Double(), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("raw_yaml", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_configs_run_id"),
        sa.ForeignKeyConstraint(
            ["run_id"], ["leaderboard_runs.run_id"], ondelete="CASCADE", name="fk_configs_run"
        ),
    )


def downgrade() -> None:
    op.drop_table("experiment_configs")
    op.drop_table("run_trades")
    op.drop_index("idx_runs_parent", table_name="leaderboard_runs")
    op.drop_index("idx_runs_pinned_market", table_name="leaderboard_runs")
    op.drop_index("idx_runs_bundle_name", table_name="leaderboard_runs")
    op.drop_index("idx_runs_fairness_group", table_name="leaderboard_runs")
    op.drop_index("idx_runs_market_family", table_name="leaderboard_runs")
    op.drop_index("idx_runs_timeframe", table_name="leaderboard_runs")
    op.drop_index("idx_runs_entry_model", table_name="leaderboard_runs")
    op.drop_index("idx_runs_feature_set", table_name="leaderboard_runs")
    op.drop_index("idx_runs_strategy", table_name="leaderboard_runs")
    op.drop_index("idx_runs_market", table_name="leaderboard_runs")
    op.drop_index("idx_runs_score_state", table_name="leaderboard_runs")
    op.drop_index("idx_runs_run_id", table_name="leaderboard_runs")
    op.drop_table("leaderboard_runs")
