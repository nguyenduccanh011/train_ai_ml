"""Sync run_symbol_stats / run_yearly_stats columns to the ORM model.

Revision ID: 0018
Revises: 0017
Create Date: 2026-06-01 13:00:00.000000

The ORM models for run_symbol_stats and run_yearly_stats declare per-bucket PnL
metrics (avg_pnl, med_pnl, std_pnl, max_win, max_loss, profit_factor, avg_hold)
that were never added by a migration — they only existed where the app called
metadata.create_all(). A migration-built database (CI, fresh Postgres) was
therefore missing them. Add them so model == migrations == database.

Idempotent: each column is added only if absent, so it is safe to run against a
database that already grew the columns out-of-band.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, column) pairs — all nullable Double metrics.
_COLUMNS = {
    "run_symbol_stats": [
        "avg_pnl",
        "med_pnl",
        "std_pnl",
        "max_win",
        "max_loss",
        "profit_factor",
        "avg_hold",
    ],
    "run_yearly_stats": [
        "avg_pnl",
        "med_pnl",
        "std_pnl",
        "max_win",
        "max_loss",
        "profit_factor",
        "avg_hold",
    ],
}


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    for table, cols in _COLUMNS.items():
        existing = {c["name"] for c in insp.get_columns(table)}
        to_add = [c for c in cols if c not in existing]
        if not to_add:
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            for col in to_add:
                batch_op.add_column(sa.Column(col, sa.Double(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    for table, cols in _COLUMNS.items():
        existing = {c["name"] for c in insp.get_columns(table)}
        to_drop = [c for c in cols if c in existing]
        if not to_drop:
            continue
        with op.batch_alter_table(table, schema=None) as batch_op:
            for col in to_drop:
                batch_op.drop_column(col)
