"""Add exit_score to run_signals (dual-ML exit prediction chain).

Revision ID: 0023
Revises: 0022
Create Date: 2026-06-03 00:00:00.000000

Dual-ML runs (separate entry + exit regressors) produce two prediction series
per (symbol, date): the entry alpha (already stored in ``score``) and the exit
model's prediction (forward drawdown / peak proximity), which was previously
discarded after being collapsed into the discrete signal. ``exit_score`` is
nullable: only dual-ML runs populate it; single-model runs leave it NULL.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("run_signals", schema=None) as batch_op:
        batch_op.add_column(sa.Column("exit_score", sa.Double(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_signals", schema=None) as batch_op:
        batch_op.drop_column("exit_score")
