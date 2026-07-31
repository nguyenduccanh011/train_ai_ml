"""Codify the full leaderboard_nav shape (18 live cols) — §4.1.

Revision ID: 0030
Revises: 0029
Create Date: 2026-08-01

``leaderboard_nav`` is script-created by ops/score_nav_leaderboard.py, whose DDL declares
only 11 columns; the LIVE table has drifted to 18 (cagr_t2/maxdd_t2 via an ad-hoc ALTER,
cagr_overlay/maxdd_overlay/overlay_k/overlay_note/overlay_config_hash added over time) with
NO migration capturing them. A fresh DB rebuilt from the script would silently miss 7 columns
and every overlay/T2 writer would fail. This migration makes the 18-col shape authoritative:
CREATE TABLE IF NOT EXISTS (fresh DB) + ADD COLUMN IF NOT EXISTS per column (no-op on live).

Additive only — no column is dropped or retyped, so it is safe to run on the live 99GB DB and
on a fresh one alike. The dead-column cleanup and the config_hash re-identity are separate,
ORM-coordinated migrations (they carry orphaning risk and are handled apart from this one).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0030"
down_revision: str | None = "0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The authoritative 18-column shape (name -> SQLAlchemy type), in live order. run_id is the PK;
# computed_at is NOT NULL with a server default so a fresh insert without it still lands.
_NAV_COLUMNS: list[tuple[str, sa.types.TypeEngine]] = [
    ("nav_adv", sa.Double()),
    ("nav_noadv", sa.Double()),
    ("cagr_adv", sa.Double()),
    ("cagr_noadv", sa.Double()),
    ("maxdd_nav", sa.Double()),
    ("nav_f22_adv", sa.Double()),
    ("years", sa.Double()),
    ("n_trades_sim", sa.Integer()),
    ("config_hash", sa.String(length=64)),
    ("cagr_t2", sa.Double()),
    ("maxdd_t2", sa.Double()),
    ("cagr_overlay", sa.Double()),
    ("maxdd_overlay", sa.Double()),
    ("overlay_k", sa.Integer()),
    ("overlay_note", sa.Text()),
    ("overlay_config_hash", sa.String(length=32)),
]


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())
    if "leaderboard_nav" not in tables:
        op.create_table(
            "leaderboard_nav",
            sa.Column("run_id", sa.String(length=512), primary_key=True, nullable=False),
            *[sa.Column(name, typ, nullable=True) for name, typ in _NAV_COLUMNS],
            sa.Column(
                "computed_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        return
    # Live table exists: add only the columns it is missing (idempotent — no-op where present).
    have = {c["name"] for c in insp.get_columns("leaderboard_nav")}
    for name, typ in _NAV_COLUMNS:
        if name not in have:
            op.add_column("leaderboard_nav", sa.Column(name, typ, nullable=True))
    if "computed_at" not in have:
        op.add_column(
            "leaderboard_nav",
            sa.Column(
                "computed_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )


def downgrade() -> None:
    # Non-destructive forward migration; downgrade is a deliberate no-op. Dropping these columns
    # would delete published NAV/overlay curves that predate any migration — never automatic.
    pass
