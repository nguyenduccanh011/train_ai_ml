"""Phase: Per-slot feature sets and targets.

Revision ID: 0014
Revises: 0013
Create Date: 2026-05-31 23:00:00.000000

Allows each component slot (entry/exit/regime/size) to override the global
feature_set and target config. New columns on component_slots are nullable —
NULL means use the global config (backward compat).

New columns:
  - feature_set_name: String(255) — registry key (e.g. 'leading_v4')
  - target_config: JSON — {type, horizon, gain_threshold, ...}
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add per-slot feature and target config columns."""
    with op.batch_alter_table("component_slots", schema=None) as batch_op:
        batch_op.add_column(sa.Column("feature_set_name", sa.String(255), nullable=True))
        batch_op.add_column(sa.Column("target_config", sa.JSON(), nullable=True))


def downgrade() -> None:
    """Remove per-slot feature and target config columns."""
    with op.batch_alter_table("component_slots", schema=None) as batch_op:
        batch_op.drop_column("target_config")
        batch_op.drop_column("feature_set_name")
