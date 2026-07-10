"""Add missing trading metrics to leaderboard_runs.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-31 15:30:00.000000

Adds:
- pnl_pct: percentage return
- max_win: largest single trade win
- max_loss: largest single trade loss
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add missing trading metrics columns."""
    op.add_column(
        "leaderboard_runs",
        sa.Column("pnl_pct", sa.Double(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("max_win", sa.Double(), nullable=False, server_default="0.0"),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("max_loss", sa.Double(), nullable=False, server_default="0.0"),
    )


def downgrade() -> None:
    """Remove trading metrics columns."""
    op.drop_column("leaderboard_runs", "max_loss")
    op.drop_column("leaderboard_runs", "max_win")
    op.drop_column("leaderboard_runs", "pnl_pct")
