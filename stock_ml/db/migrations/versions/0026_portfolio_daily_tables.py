"""Add portfolio-execution overlay tables: run_equity + run_portfolio_daily.

Revision ID: 0026
Revises: 0025
Create Date: 2026-07-12 00:00:00.000000

Backs the detail-page 'Danh mục' tab: a portfolio-execution run (e.g. the preempt +
conviction-sizing + conv-skip combo champion) persists a daily NAV/exposure timeline
(run_equity) and per-day holdings with ACTUAL weights + entry/exit events
(run_portfolio_daily). Distinct from run_trades (per-symbol, equal-weight, no daily
snapshot). Idempotent create so a DB that already has the raw tables re-aligns cleanly.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    existing = set(insp.get_table_names())

    # Drop any pre-existing raw (pre-migration) versions so the schema is authoritative.
    for tbl in ("run_portfolio_daily", "run_equity"):
        if tbl in existing:
            op.drop_table(tbl)

    op.create_table(
        "run_equity",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(length=512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("nav", sa.Double(), nullable=False),
        sa.Column("cash", sa.Double(), nullable=True),
        sa.Column("exposure", sa.Double(), nullable=True),
        sa.Column("n_positions", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_run_equity_run_date", "run_equity", ["run_id", "date"])

    op.create_table(
        "run_portfolio_daily",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(length=512),
            sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("weight", sa.Double(), nullable=True),
        sa.Column("entry_weight", sa.Double(), nullable=True),
        sa.Column("unreal_pnl", sa.Double(), nullable=True),
        sa.Column("entry_date", sa.Date(), nullable=True),
        sa.Column("days_held", sa.Integer(), nullable=True),
        sa.Column("is_new", sa.Boolean(), nullable=True),
        sa.Column("is_exit", sa.Boolean(), nullable=True),
        sa.Column("exit_reason", sa.String(length=32), nullable=True),
        sa.Column("conv", sa.Double(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_run_portfolio_daily_run_date", "run_portfolio_daily", ["run_id", "date"])


def downgrade() -> None:
    op.drop_index("ix_run_portfolio_daily_run_date", table_name="run_portfolio_daily")
    op.drop_table("run_portfolio_daily")
    op.drop_index("ix_run_equity_run_date", table_name="run_equity")
    op.drop_table("run_equity")
