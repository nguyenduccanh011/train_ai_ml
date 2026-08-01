"""Drop the dead fairness columns + index — §4.4.

Revision ID: 0032
Revises: 0031
Create Date: 2026-08-01

The leaderboard fairness mechanism (baseline comparison) was dead: the six
``same_*_as_baseline`` flags were NULL on 3612/3612 rows, ``is_baseline`` was false on all
3612, and ``fairness_group_key`` — though populated — held only 3 distinct values (degenerate)
and was never surfaced by the API (the dashboard "fair mode" read undefined and failed closed).
The fairness.py module, the CacheKeys/fairness Pydantic fields, the aggregator baseline
annotation, and RunRepository.get_by_fairness_group were removed in the same change; this
migration drops the now-unused columns and the fairness_group_key index.

Guarded (drop only if present) so it is safe on the live DB and a fresh one built past 0001.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0032"
down_revision: str | None = "0031"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEAD_COLUMNS = (
    "fairness_group_key",
    "is_baseline",
    "same_symbols_as_baseline",
    "same_window_as_baseline",
    "same_cost_as_baseline",
    "same_target_as_baseline",
    "same_timeframe_as_baseline",
    "same_market_family_as_baseline",
)


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    indexes = {ix["name"] for ix in insp.get_indexes("leaderboard_runs")}
    if "idx_runs_fairness_group" in indexes:
        op.drop_index("idx_runs_fairness_group", table_name="leaderboard_runs")
    have = {c["name"] for c in insp.get_columns("leaderboard_runs")}
    for name in _DEAD_COLUMNS:
        if name in have:
            op.drop_column("leaderboard_runs", name)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    have = {c["name"] for c in insp.get_columns("leaderboard_runs")}
    if "fairness_group_key" not in have:
        op.add_column(
            "leaderboard_runs",
            sa.Column("fairness_group_key", sa.String(40), nullable=False, server_default=""),
        )
    if "is_baseline" not in have:
        op.add_column(
            "leaderboard_runs",
            sa.Column("is_baseline", sa.Boolean(), nullable=False, server_default="false"),
        )
    for name in _DEAD_COLUMNS[2:]:
        if name not in have:
            op.add_column("leaderboard_runs", sa.Column(name, sa.Boolean(), nullable=True))
    indexes = {ix["name"] for ix in insp.get_indexes("leaderboard_runs")}
    if "idx_runs_fairness_group" not in indexes:
        op.create_index("idx_runs_fairness_group", "leaderboard_runs", ["fairness_group_key"])
