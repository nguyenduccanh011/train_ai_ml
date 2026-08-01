"""Enforce the run_trades = BASE-only invariant — migrate the 19 OUTPUT runs (§B2).

Revision ID: 0033
Revises: 0032
Create Date: 2026-08-01

PORTFOLIO_REGISTRATION_PIPELINE.md §1/§B2: `run_trades` must hold ENGINE (BASE) trades only;
Stage-2 OUTPUT (post-overlay) lives in `run_trades_overlay`. Migration 0029 created the table +
invariant but did NOT move the legacy OUTPUT rows. This does that one-time move.

An OUTPUT run is identified structurally: it has rows whose exit_reason is overlay-only
(preempt / green_trail / early_cut) — the engine BASE never produces those. ALL rows of such a
run are its overlaid book, so the whole run moves (not just the overlay-exit rows). conv/prio are
NULL (the legacy rows predate conviction/priority capture).

Atomic (single migration transaction: INSERT then DELETE) and guarded (0 overlay-exit rows may
remain in run_trades afterwards). A CSV backup of the moved rows was taken before applying
(/f/pg_backups/run_trades_19_output_preB2.csv). The trades API already prefers run_trades_overlay
(runs.py get_run_trades, "B4"), so these runs' trades tab is unaffected.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0033"
down_revision: str | None = "0032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OVERLAY_REASONS = ("preempt", "green_trail", "early_cut")


def _output_run_ids(bind) -> list[str]:
    rows = bind.execute(
        sa.text(
            "SELECT DISTINCT run_id FROM run_trades WHERE exit_reason IN :ov"
        ).bindparams(sa.bindparam("ov", expanding=True)),
        {"ov": list(_OVERLAY_REASONS)},
    ).fetchall()
    return [r[0] for r in rows]


def upgrade() -> None:
    bind = op.get_bind()
    rids = _output_run_ids(bind)
    if not rids:
        return  # already BASE-only (idempotent no-op)

    bind.execute(
        sa.text(
            "INSERT INTO run_trades_overlay "
            "(run_id, symbol, entry_date, entry_price, exit_date, exit_price, "
            " holding_days, pnl_pct, exit_reason, conv, prio, created_at) "
            "SELECT run_id, symbol, entry_date, entry_price, exit_date, exit_price, "
            " holding_days, pnl_pct, exit_reason, NULL, NULL, now() "
            "FROM run_trades WHERE run_id IN :rids"
        ).bindparams(sa.bindparam("rids", expanding=True)),
        {"rids": rids},
    )
    bind.execute(
        sa.text("DELETE FROM run_trades WHERE run_id IN :rids").bindparams(
            sa.bindparam("rids", expanding=True)
        ),
        {"rids": rids},
    )

    remaining = bind.execute(
        sa.text(
            "SELECT count(*) FROM run_trades WHERE exit_reason IN :ov"
        ).bindparams(sa.bindparam("ov", expanding=True)),
        {"ov": list(_OVERLAY_REASONS)},
    ).scalar()
    if remaining:
        raise RuntimeError(
            f"BASE-only invariant not achieved: {remaining} overlay-exit rows remain in run_trades"
        )


def downgrade() -> None:
    # Deliberate no-op: the moved rows cannot be distinguished from natively-registered overlay
    # rows without a marker, so an automatic reverse would over-delete. Restore from the CSV
    # backup (/f/pg_backups/run_trades_19_output_preB2.csv) if a rollback is ever needed.
    pass
