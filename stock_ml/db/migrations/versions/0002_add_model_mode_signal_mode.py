"""add_model_mode_signal_mode_to_leaderboard_and_config

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add new columns to leaderboard_runs
    op.add_column(
        "leaderboard_runs",
        sa.Column("model_mode", sa.String(32), nullable=False, server_default="ml_only"),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("signal_mode", sa.String(32), nullable=False, server_default="entry_first"),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("regime_model_type", sa.String(64), nullable=False, server_default="none"),
    )
    op.add_column(
        "leaderboard_runs",
        sa.Column("size_model_type", sa.String(64), nullable=False, server_default="none"),
    )

    # Add new columns to experiment_configs
    op.add_column(
        "experiment_configs",
        sa.Column("signal_mode", sa.String(32), nullable=True),
    )
    op.add_column(
        "experiment_configs",
        sa.Column("model_mode", sa.String(32), nullable=True),
    )
    op.add_column(
        "experiment_configs",
        sa.Column("regime_model", sa.JSON(), nullable=True),
    )
    op.add_column(
        "experiment_configs",
        sa.Column("size_model", sa.JSON(), nullable=True),
    )
    op.add_column(
        "experiment_configs",
        sa.Column("yaml_schema_version", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    # Drop columns from experiment_configs
    op.drop_column("experiment_configs", "yaml_schema_version")
    op.drop_column("experiment_configs", "size_model")
    op.drop_column("experiment_configs", "regime_model")
    op.drop_column("experiment_configs", "model_mode")
    op.drop_column("experiment_configs", "signal_mode")

    # Drop columns from leaderboard_runs
    op.drop_column("leaderboard_runs", "size_model_type")
    op.drop_column("leaderboard_runs", "regime_model_type")
    op.drop_column("leaderboard_runs", "signal_mode")
    op.drop_column("leaderboard_runs", "model_mode")
