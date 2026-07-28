"""Add run_pending: causal resting pullback order book.

Revision ID: 0028
Revises: 0027
Create Date: 2026-07-12 02:00:00.000000

Replaces the look-ahead 'will-fill' pending list with the true causal resting book: for each date, the
buy signals still waiting to fill (limit at close[signal]*(1-pullback_pct), within the window, not filled,
not held). outcome/result_date are post-hoc reference (fill vs expire).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0028"
down_revision: str | None = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "run_pending" in set(insp.get_table_names()):
        op.drop_table("run_pending")
    op.create_table(
        "run_pending",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(length=512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("signal_date", sa.Date(), nullable=True),
        sa.Column("days_waiting", sa.Integer(), nullable=True),
        sa.Column("limit_price", sa.Double(), nullable=True),
        sa.Column("ref_price", sa.Double(), nullable=True),
        sa.Column("pct_to_limit", sa.Double(), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=True),
        sa.Column("result_date", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_run_pending_run_date", "run_pending", ["run_id", "date"])


def downgrade() -> None:
    op.drop_index("ix_run_pending_run_date", table_name="run_pending")
    op.drop_table("run_pending")
