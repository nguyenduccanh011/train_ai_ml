"""add_experiment_tracking_fields

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add experiment tracking columns to leaderboard_runs (Phase 0-2 research pipeline)
    op.add_column(
        "leaderboard_runs",
        sa.Column("experiment_group", sa.String(255), nullable=False, server_default="ungrouped"),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("variant_type", sa.String(32), nullable=True),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("metadata_notes", sa.Text(), nullable=True),
    )

    # Add index for experiment_group filtering
    op.create_index("idx_runs_experiment_group", "leaderboard_runs", ["experiment_group"])


def downgrade() -> None:
    # Drop index
    op.drop_index("idx_runs_experiment_group", table_name="leaderboard_runs")

    # Drop columns
    op.drop_column("leaderboard_runs", "metadata_notes")
    op.drop_column("leaderboard_runs", "variant_type")
    op.drop_column("leaderboard_runs", "experiment_group")
