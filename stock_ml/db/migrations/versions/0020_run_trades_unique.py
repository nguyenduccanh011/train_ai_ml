"""Add uq_run_symbol_entry unique constraint to run_trades.

Revision ID: 0020
Revises: 0019
Create Date: 2026-06-01 15:00:00.000000

RunTradeModel declares UniqueConstraint(run_id, symbol, entry_date) but no
migration ever created it, so a migration-built database lacked it — meaning
on_conflict_do_nothing in the trade repo had no constraint to dedupe against and
re-running a template duplicated trades. Add the constraint so model == database
and the repo's idempotent upsert works at the DB level too.

Idempotent: created only if absent.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NAME = "uq_run_symbol_entry"


def _exists(bind) -> bool:
    insp = sa.inspect(bind)
    names = {uc["name"] for uc in insp.get_unique_constraints("run_trades")}
    # SQLite may surface it among indexes instead of constraints.
    names |= {ix["name"] for ix in insp.get_indexes("run_trades")}
    return _NAME in names


def upgrade() -> None:
    bind = op.get_bind()
    if _exists(bind):
        return
    with op.batch_alter_table("run_trades", schema=None) as batch_op:
        batch_op.create_unique_constraint(_NAME, ["run_id", "symbol", "entry_date"])


def downgrade() -> None:
    bind = op.get_bind()
    if not _exists(bind):
        return
    with op.batch_alter_table("run_trades", schema=None) as batch_op:
        batch_op.drop_constraint(_NAME, type_="unique")
