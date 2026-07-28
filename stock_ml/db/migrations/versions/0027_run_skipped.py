"""Add run_skipped: opportunity-cost ledger of signals the portfolio did NOT take.

Revision ID: 0027
Revises: 0026
Create Date: 2026-07-12 01:00:00.000000

Records signals dropped by the portfolio overlay (conv-skip / capacity) together with the base
backtest's REALIZED return for that name — so the detail page can answer "was skipping right?".
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0027"
down_revision: str | None = "0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "run_skipped" in set(insp.get_table_names()):
        op.drop_table("run_skipped")
    op.create_table(
        "run_skipped",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(length=512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=True),
        sa.Column("entry_date", sa.Date(), nullable=True),
        sa.Column("pnl_pct", sa.Double(), nullable=True),
        sa.Column("conv", sa.Double(), nullable=True),
        sa.Column("skip_reason", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_run_skipped_run", "run_skipped", ["run_id", "signal_date"])


def downgrade() -> None:
    op.drop_index("ix_run_skipped_run", table_name="run_skipped")
    op.drop_table("run_skipped")
