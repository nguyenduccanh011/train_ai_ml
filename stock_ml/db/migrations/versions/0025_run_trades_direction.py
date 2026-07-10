"""Add direction (position side) to run_trades.

Revision ID: 0025
Revises: 0024
Create Date: 2026-06-21 00:00:00.000000

Two-sided strategies (the VN30-futures intraday model trades both long AND short)
need an explicit position side. Without it a profitable SHORT (open by SELLING high,
close by BUYING BACK low) reads on the model-details table as "buy high, sell low" yet
shows profit, which is confusing. 'long'/'short'; nullable so legacy long-only daily
runs leave it NULL.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("run_trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("direction", sa.String(length=8), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_trades", schema=None) as batch_op:
        batch_op.drop_column("direction")
