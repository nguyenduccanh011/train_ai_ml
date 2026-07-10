"""Add backtest transparency fields to leaderboard_runs.

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-31 15:00:00.000000

Adds (cost_* already exist as TEXT from 0004):
- test_start_date, test_end_date: backtest window (new)
- train_start_date, train_end_date: training window (new)
- universe_slug, universe_version: universe tracking (new)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add backtest transparency columns (cost_* already exist)."""
    # Add date columns for backtest window
    op.add_column(
        "leaderboard_runs",
        sa.Column("test_start_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("test_end_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("train_start_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("train_end_date", sa.DateTime(timezone=True), nullable=True),
    )

    # Add universe tracking
    op.add_column(
        "leaderboard_runs",
        sa.Column("universe_slug", sa.String(128), nullable=True),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("universe_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Remove backtest transparency columns."""
    op.drop_column("leaderboard_runs", "universe_version")
    op.drop_column("leaderboard_runs", "universe_slug")
    op.drop_column("leaderboard_runs", "train_end_date")
    op.drop_column("leaderboard_runs", "train_start_date")
    op.drop_column("leaderboard_runs", "test_end_date")
    op.drop_column("leaderboard_runs", "test_start_date")
