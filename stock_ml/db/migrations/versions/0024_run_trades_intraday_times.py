"""Add entry_time/exit_time to run_trades (intraday trade timestamps).

Revision ID: 0024
Revises: 0023
Create Date: 2026-06-21 00:00:00.000000

Daily-spot runs trade at the day grain (entry_date/exit_date suffice). Intraday
runs (e.g. the VN30-futures 15m strategy) enter and exit WITHIN a session, so the
model-details chart needs the bar-level timestamp to place buy/sell markers on the
right 15m candle. Both columns are nullable: daily runs leave them NULL and keep
rendering by date; intraday runs populate them.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("run_trades", schema=None) as batch_op:
        batch_op.add_column(sa.Column("entry_time", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("exit_time", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("run_trades", schema=None) as batch_op:
        batch_op.drop_column("exit_time")
        batch_op.drop_column("entry_time")
