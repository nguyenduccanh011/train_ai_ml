"""run_overlay: a Stage-2 book is (run_id, overlay_key), not run_id alone.

Revision ID: 0035
Revises: 0034
Create Date: 2026-08-03

Every overlay table was keyed by ``run_id`` alone, so scoring one run under a second strategy
DELETEd the first one's book and overwrote its number — measured on the live board: 2551 runs
carried an overlay number but only 30 had a book, and every one of those books predated the panel
its number was scored on. See docs/refactor/OVERLAY_IDENTITY_RESTRUCTURE.md §1-2.

This migration adds the missing dimension:
  - ``run_overlay``  one row per (run, strategy): full config JSON (not just a hash), metrics,
    the panel it was scored on, and whether its detail tables are populated.
  - ``overlay_key`` on the 5 detail tables, so two strategies of one run keep two books.

ADDITIVE ONLY. Existing rows are stamped ``overlay_key='legacy'`` rather than deleted or guessed:
they were written by an unknown config on an unknown panel, and inventing a key for them would
launder exactly the uncertainty this table exists to expose. Ops step B6
(``scripts/ops/rescore_books.py``) re-scores them under a named strategy and drops what is left.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0035"
down_revision: str | None = "0034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The books that must now carry a strategy key. run_trades stays BASE-only (0033).
_DETAIL_TABLES = (
    "run_equity",
    "run_portfolio_daily",
    "run_trades_overlay",
    "run_skipped",
    "run_pending",
)
# Sentinel for rows that predate the key. NOT a real md5 — it must be impossible for a scoring to
# collide with it, and obvious in a query result that this row's provenance is unknown.
LEGACY_KEY = "legacy"


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)

    if "run_overlay" not in insp.get_table_names():
        op.create_table(
            "run_overlay",
            sa.Column("run_id", sa.String(length=512), nullable=False),
            sa.Column("overlay_key", sa.String(length=32), nullable=False),
            # human-facing name of the strategy ("K10 · sàn 5 tỷ"); the hash is not a label
            sa.Column("label", sa.Text(), nullable=True),
            # FULL PortfolioConstants overrides — a hash proves identity but cannot be read back
            # into a config, and a book nobody can re-run is a number nobody can check
            # sa.JSON (not JSONB): alembic.ini defaults to SQLite, so every migration must render
            # on both backends — a Postgres-only type here breaks the local dev DB path.
            sa.Column("config", sa.JSON(), nullable=True),
            # BASE trades borrowed from this run (the overlay/* mechanism, now a column not a run)
            sa.Column("base_run", sa.String(length=512), nullable=True),
            sa.Column("cagr", sa.Double(), nullable=True),
            sa.Column("maxdd", sa.Double(), nullable=True),
            sa.Column("nav", sa.Double(), nullable=True),
            sa.Column("years", sa.Double(), nullable=True),
            sa.Column("n_trades", sa.Integer(), nullable=True),
            sa.Column("k", sa.Integer(), nullable=True),
            # identity of THIS scoring (config + panel) vs the strategy key above (config only)
            sa.Column("scoring_hash", sa.String(length=32), nullable=True),
            sa.Column("panel_fp", sa.String(length=64), nullable=True),
            sa.Column("conv_miss_frac", sa.Double(), nullable=True),
            sa.Column("offpanel_frac", sa.Double(), nullable=True),
            sa.Column("has_detail", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column(
                "computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
            ),
            sa.ForeignKeyConstraint(["run_id"], ["leaderboard_runs.run_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("run_id", "overlay_key"),
        )
        op.create_index("ix_run_overlay_run", "run_overlay", ["run_id"])

    for tbl in _DETAIL_TABLES:
        if tbl not in insp.get_table_names():
            continue
        have = {c["name"] for c in insp.get_columns(tbl)}
        if "overlay_key" not in have:
            op.add_column(
                tbl,
                sa.Column(
                    "overlay_key",
                    sa.String(length=32),
                    nullable=False,
                    server_default=LEGACY_KEY,
                ),
            )
        idx = f"ix_{tbl}_run_overlay"
        if idx not in {i["name"] for i in insp.get_indexes(tbl)}:
            op.create_index(idx, tbl, ["run_id", "overlay_key"])


def downgrade() -> None:
    # Non-destructive forward migration; downgrade is a deliberate no-op (mirrors 0030/0034).
    pass
