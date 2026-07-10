"""Add template_id FK to leaderboard_runs for tracking template source.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-31 20:10:00.000000

Backward-compatible nullable FK allows YAML runs to coexist with DB-first template runs.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add template_id FK to leaderboard_runs."""
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("template_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_leaderboard_runs_template",
            "strategy_templates",
            ["template_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "idx_leaderboard_runs_template",
            ["template_id"],
        )


def downgrade() -> None:
    """Remove template_id FK from leaderboard_runs."""
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        batch_op.drop_index("idx_leaderboard_runs_template")
        batch_op.drop_constraint("fk_leaderboard_runs_template", type_="foreignkey")
        batch_op.drop_column("template_id")
