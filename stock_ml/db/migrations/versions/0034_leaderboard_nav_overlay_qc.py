"""Add overlay data-coverage QC columns to leaderboard_nav.

Revision ID: 0034
Revises: 0033
Create Date: 2026-08-02

The Stage-2 overlay now scores against the SERVING-DECLARED market panel artifact
(SERVING_PANEL_TASKS.md T1). To make the backtest≡serving band traceable, persist per-run:
  - overlay_panel_fp: the declared panel_fingerprint (rows:max_date:Σclose:Σvolume) the run was
    scored on — so a board row records WHICH universe produced its cagr_overlay.
  - conv_miss_frac / offpanel_frac: the data-coverage QC from run_portfolio — how many trades hit
    a missing conviction bar (halt) vs an off-panel symbol (data gap). offpanel_frac>0 flags a run
    whose traded universe is not fully covered by the declared panel (a STALE dyn run trading
    ICB-excluded fund-certs, pre-universe_resolver-fix) — surfaced instead of silently fabricated.

Additive only (ADD COLUMN IF NOT EXISTS semantics via inspect) — safe on the live DB.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0034"
down_revision: str | None = "0033"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_QC_COLUMNS: list[tuple[str, sa.types.TypeEngine]] = [
    ("overlay_panel_fp", sa.String(length=64)),
    ("conv_miss_frac", sa.Double()),
    ("offpanel_frac", sa.Double()),
]


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    have = {c["name"] for c in insp.get_columns("leaderboard_nav")}
    for name, typ in _QC_COLUMNS:
        if name not in have:
            op.add_column("leaderboard_nav", sa.Column(name, typ, nullable=True))


def downgrade() -> None:
    # Non-destructive forward migration; downgrade is a deliberate no-op (mirrors 0030).
    pass
