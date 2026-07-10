"""Add reproducibility fields to leaderboard_runs.

Revision ID: 0019
Revises: 0018
Create Date: 2026-06-01 14:00:00.000000

Locks the exact code + data + library versions behind every run so a result can
be reproduced later (alongside the existing config_hash / run_seed):
  - git_sha            VARCHAR(40) — commit the run was produced from
  - data_snapshot_date DATE        — max OHLCV date in the DuckDB store at run time
  - lib_versions       JSON        — {"numpy": "1.26.3", "lightgbm": "4.6.0", ...}

Idempotent: each column is added only if absent.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_COLUMNS = {
    "git_sha": sa.String(length=40),
    "data_snapshot_date": sa.Date(),
    "lib_versions": sa.JSON(),
}


def upgrade() -> None:
    bind = op.get_bind()
    existing = {c["name"] for c in sa.inspect(bind).get_columns("leaderboard_runs")}
    to_add = {name: typ for name, typ in _COLUMNS.items() if name not in existing}
    if not to_add:
        return
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        for name, typ in to_add.items():
            batch_op.add_column(sa.Column(name, typ, nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    existing = {c["name"] for c in sa.inspect(bind).get_columns("leaderboard_runs")}
    to_drop = [name for name in _COLUMNS if name in existing]
    if not to_drop:
        return
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        for name in to_drop:
            batch_op.drop_column(name)
