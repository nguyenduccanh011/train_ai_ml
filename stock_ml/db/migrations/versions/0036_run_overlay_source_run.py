"""run_overlay.source_run: which retired placeholder a book was folded from.

Revision ID: 0036
Revises: 0035
Create Date: 2026-08-03

Migration 0035 moved the ``overlay/*`` placeholder runs' books onto their parent under a real
``overlay_key``. What it did not record is the reverse pointer: opening the old link
(``overlay/_dyn300_k6_liqcol``) we know the parent (``parent_run_id``) but not WHICH of its six
strategies that placeholder was — so the page could only drop the reader on the parent's default
book, which is a different strategy than the one they clicked.

Deriving it from ``overlay_note`` would work today and rot tomorrow: that note is a
human-readable string, and "policy encoded in a string" is the exact antipattern this restructure
removes. So the mapping becomes a column.

Additive; backfilled by ``scripts/ops/rescore_books.py --backfill-source`` (one-off, reads the
note that the tier registrar wrote) and set going forward by ``persist_overlay``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0036"
down_revision: str | None = "0035"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if "run_overlay" not in insp.get_table_names():
        return
    have = {c["name"] for c in insp.get_columns("run_overlay")}
    if "source_run" not in have:
        op.add_column("run_overlay", sa.Column("source_run", sa.String(length=512), nullable=True))
        op.create_index("ix_run_overlay_source", "run_overlay", ["source_run"])


def downgrade() -> None:
    # Non-destructive forward migration; downgrade is a deliberate no-op (mirrors 0034/0035).
    pass
