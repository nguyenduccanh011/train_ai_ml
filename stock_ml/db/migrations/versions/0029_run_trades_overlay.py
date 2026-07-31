"""Add run_trades_overlay: Stage-2 (portfolio-overlaid) trades get their own table.

Revision ID: 0029
Revises: 0028
Create Date: 2026-07-29

Establishes the BASE-only invariant on run_trades (docs/refactor/
PORTFOLIO_REGISTRATION_PIPELINE.md): engine-level trades stay in run_trades;
anything that went through the Stage-2 overlay (preempt / green_trail /
early_cut, conviction sizing) lives here. Also adds overlay_config_hash to
leaderboard_nav (script-created table — guarded, may not exist on fresh DBs)
so official registrations are idempotent and traceable.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0029"
down_revision: str | None = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    if "run_trades_overlay" not in tables:
        op.create_table(
            "run_trades_overlay",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "run_id",
                sa.String(length=512),
                sa.ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("symbol", sa.String(length=32), nullable=False),
            sa.Column("entry_date", sa.Date(), nullable=True),
            sa.Column("entry_price", sa.Double(), nullable=True),
            sa.Column("exit_date", sa.Date(), nullable=True),
            sa.Column("exit_price", sa.Double(), nullable=True),
            sa.Column("holding_days", sa.Integer(), nullable=True),
            sa.Column("pnl_pct", sa.Double(), nullable=True),
            sa.Column("exit_reason", sa.String(length=32), nullable=True),
            sa.Column("conv", sa.Double(), nullable=True),
            sa.Column("prio", sa.Double(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_run_trades_overlay_run", "run_trades_overlay", ["run_id"])
    # leaderboard_nav is created by ops/score_nav_leaderboard.py (no ORM); only ALTER if present.
    if "leaderboard_nav" in tables:
        cols = {c["name"] for c in insp.get_columns("leaderboard_nav")}
        if "overlay_config_hash" not in cols:
            op.add_column("leaderboard_nav", sa.Column("overlay_config_hash", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_index("ix_run_trades_overlay_run", table_name="run_trades_overlay")
    op.drop_table("run_trades_overlay")
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "leaderboard_nav" in set(insp.get_table_names()):
        cols = {c["name"] for c in insp.get_columns("leaderboard_nav")}
        if "overlay_config_hash" in cols:
            op.drop_column("leaderboard_nav", "overlay_config_hash")
