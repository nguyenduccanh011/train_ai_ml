"""Add universe management tables for symbol set CRUD.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-31 14:00:00.000000

Universe management: symbol sets (name, version, market, symbols),
with soft-delete (is_active flag) and locking (is_locked flag).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create universe_sets and universe_symbols tables."""
    op.create_table(
        "universe_sets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(128), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("market", sa.String(64), nullable=False),
        sa.Column("is_locked", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("symbol_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_universe_slug"),
    )
    op.create_index("idx_universe_slug", "universe_sets", ["slug"], unique=False)
    op.create_index("idx_universe_market", "universe_sets", ["market"], unique=False)
    op.create_index("idx_universe_is_active", "universe_sets", ["is_active"], unique=False)
    op.create_index(
        "idx_universe_market_active",
        "universe_sets",
        ["market", "is_active"],
        unique=False,
    )
    op.create_index(
        "idx_universe_market_locked",
        "universe_sets",
        ["market", "is_locked"],
        unique=False,
    )

    op.create_table(
        "universe_symbols",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("universe_id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("symbol_group", sa.String(64), nullable=True),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["universe_id"],
            ["universe_sets.id"],
            name="fk_universe_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("universe_id", "symbol", name="uq_universe_symbol"),
    )
    op.create_index(
        "idx_universe_symbols_universe_id", "universe_symbols", ["universe_id"], unique=False
    )
    op.create_index("idx_universe_symbols_symbol", "universe_symbols", ["symbol"], unique=False)


def downgrade() -> None:
    """Drop universe_symbols and universe_sets tables."""
    op.drop_index("idx_universe_symbols_symbol", table_name="universe_symbols")
    op.drop_index("idx_universe_symbols_universe_id", table_name="universe_symbols")
    op.drop_table("universe_symbols")

    op.drop_index("idx_universe_market_locked", table_name="universe_sets")
    op.drop_index("idx_universe_market_active", table_name="universe_sets")
    op.drop_index("idx_universe_is_active", table_name="universe_sets")
    op.drop_index("idx_universe_market", table_name="universe_sets")
    op.drop_index("idx_universe_slug", table_name="universe_sets")
    op.drop_table("universe_sets")
