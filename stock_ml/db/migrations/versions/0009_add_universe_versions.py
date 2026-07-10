"""Add universe_versions table for version history snapshots.

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-31 18:00:00.000000

When universe symbols change, version increments and symbol list
is saved to universe_versions for audit trail and reconstruction.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create universe_versions table."""
    op.create_table(
        "universe_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("universe_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("symbols_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("symbol_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["universe_id"],
            ["universe_sets.id"],
            ondelete="CASCADE",
            name="fk_universe_versions_universe_id",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_universe_versions"),
        sa.UniqueConstraint("universe_id", "version", name="uq_universe_version"),
    )
    op.create_index(
        "idx_universe_versions_universe_id",
        "universe_versions",
        ["universe_id"],
    )
    op.create_index(
        "idx_universe_versions_universe_version",
        "universe_versions",
        ["universe_id", "version"],
    )


def downgrade() -> None:
    """Drop universe_versions table."""
    op.drop_index(
        "idx_universe_versions_universe_version",
        table_name="universe_versions",
    )
    op.drop_index(
        "idx_universe_versions_universe_id",
        table_name="universe_versions",
    )
    op.drop_table("universe_versions")
