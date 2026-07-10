"""Phase 4: hysteresis thresholds + enable run_signals table.

Revision ID: 0017
Revises: 0016
Create Date: 2026-06-01 12:00:00.000000

Adds:
  - strategy_templates.entry_threshold / exit_threshold (nullable Float).
    NULL means "derive from signal_threshold" (+thr / -thr), so existing templates
    keep their exact legacy behavior with no backfill required.
  - run_signals table (migration 0016 was disabled): persists per (run, symbol, date)
    the discrete signal plus the continuous alpha score, which previously had nowhere
    to live. Created only if it does not already exist (it may have been made by hand).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.add_column(sa.Column("entry_threshold", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("exit_threshold", sa.Float(), nullable=True))

    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    if "run_signals" not in existing:
        op.create_table(
            "run_signals",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(512), nullable=False, index=True),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("date", sa.Date(), nullable=False),
            sa.Column("signal", sa.Integer(), nullable=False),
            sa.Column("score", sa.Double(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["run_id"], ["leaderboard_runs.run_id"], ondelete="CASCADE"),
            sa.UniqueConstraint("run_id", "symbol", "date", name="uq_signal"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    if "run_signals" in existing:
        op.drop_table("run_signals")

    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.drop_column("exit_threshold")
        batch_op.drop_column("entry_threshold")
