"""add_jobs_and_stats_tables

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create jobs table for async task persistence (replaces in-memory _JOBS dict)
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(64), primary_key=True, nullable=False),
        sa.Column("type", sa.String(32), nullable=False),  # train, retrain, gc_sweep
        sa.Column(
            "status", sa.String(32), nullable=False, server_default="pending"
        ),  # pending, running, done, failed
        sa.Column(
            "run_id",
            sa.String(512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
    )
    op.create_index("idx_jobs_status", "jobs", ["status"])
    op.create_index("idx_jobs_run_id", "jobs", ["run_id"])

    # Create run_yearly_stats table for per-year breakdowns
    op.create_table(
        "run_yearly_stats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Double(), nullable=True),
        sa.Column("total_pnl", sa.Double(), nullable=True),
        sa.Column("max_drawdown", sa.Double(), nullable=True),
        sa.UniqueConstraint("run_id", "year", name="uq_run_yearly"),
    )
    op.create_index("idx_run_yearly_stats_run_id", "run_yearly_stats", ["run_id"])

    # Create run_symbol_stats table for per-symbol breakdowns
    op.create_table(
        "run_symbol_stats",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Double(), nullable=True),
        sa.Column("total_pnl", sa.Double(), nullable=True),
        sa.UniqueConstraint("run_id", "symbol", name="uq_run_symbol"),
    )
    op.create_index("idx_run_symbol_stats_run_id", "run_symbol_stats", ["run_id"])


def downgrade() -> None:
    # Drop run_symbol_stats
    op.drop_index("idx_run_symbol_stats_run_id", table_name="run_symbol_stats")
    op.drop_table("run_symbol_stats")

    # Drop run_yearly_stats
    op.drop_index("idx_run_yearly_stats_run_id", table_name="run_yearly_stats")
    op.drop_table("run_yearly_stats")

    # Drop jobs
    op.drop_index("idx_jobs_run_id", table_name="jobs")
    op.drop_index("idx_jobs_status", table_name="jobs")
    op.drop_table("jobs")
