"""Add run_signals table and expand yearly/symbol stats columns.

Revision ID: 0016
Revises: 0015
Create Date: 2026-06-01 00:00:00.000000

DISABLED: Migration causing startup issues. Tables created manually if needed.
"""

from collections.abc import Sequence

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Migration disabled - skip."""
    pass


def downgrade() -> None:
    """Migration disabled - skip."""
    pass
