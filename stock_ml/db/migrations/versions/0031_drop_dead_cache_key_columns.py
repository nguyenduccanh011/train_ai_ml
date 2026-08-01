"""Drop the dead cache_key_features / cache_key_predictions columns — §4.8.

Revision ID: 0031
Revises: 0030
Create Date: 2026-08-01

``leaderboard_runs.cache_key_features`` and ``cache_key_predictions`` were carried since the
initial schema (0001) to attribute feature/prediction cache files to a run, but the DB-first
write path never populated them: both are empty ('') on 3612/3612 live rows. The cache GC
attributes files from each run's ``predictions_meta.json`` (the file source), not these columns,
so nothing reads them. The Pydantic ``CacheKeys`` model + all ORM/adapter/aggregator wiring were
removed in the same change; this migration drops the now-unused columns.

Guarded (drop only if present) so it is safe on a fresh DB built past 0001 and on the live DB.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0031"
down_revision: str | None = "0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEAD_COLUMNS = ("cache_key_features", "cache_key_predictions")


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    have = {c["name"] for c in insp.get_columns("leaderboard_runs")}
    for name in _DEAD_COLUMNS:
        if name in have:
            op.drop_column("leaderboard_runs", name)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    have = {c["name"] for c in insp.get_columns("leaderboard_runs")}
    for name in _DEAD_COLUMNS:
        if name not in have:
            op.add_column(
                "leaderboard_runs",
                sa.Column(name, sa.Text(), nullable=False, server_default=""),
            )
