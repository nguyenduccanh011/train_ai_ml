"""Seed the vn_stock_default universe (symbol set moved out of market YAML).

Revision ID: 0021
Revises: 0020
Create Date: 2026-06-01 16:00:00.000000

P5: the market default symbol list used to live in config/markets/vn_stock.yaml
(symbols.default_list). It now lives in the DB (universe_sets / universe_symbols)
under slug "vn_stock_default" and is resolved at runtime by get_pipeline_symbols.
This migration seeds that universe so a fresh database reproduces the exact 61
symbols without the YAML hardcode.

Idempotent: skips if the slug already exists. Dialect-agnostic (PG + SQLite).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SLUG = "vn_stock_default"
# The 61-symbol default set previously hardcoded in vn_stock.yaml:symbols.default_list.
_SYMBOLS = [
    "ACB",
    "AAS",
    "AAV",
    "ACV",
    "BCG",
    "BCM",
    "BID",
    "BSR",
    "BVH",
    "CTG",
    "DCM",
    "DGC",
    "DIG",
    "DPM",
    "EIB",
    "FPT",
    "FRT",
    "GAS",
    "GEX",
    "GMD",
    "HCM",
    "HDB",
    "HDG",
    "HPG",
    "HSG",
    "KBC",
    "KDH",
    "LPB",
    "MBB",
    "MSN",
    "MWG",
    "NKG",
    "NLG",
    "NT2",
    "NVL",
    "OCB",
    "PC1",
    "PDR",
    "PLX",
    "PNJ",
    "POW",
    "PVD",
    "PVS",
    "REE",
    "SAB",
    "SBT",
    "SHB",
    "SSI",
    "STB",
    "TCB",
    "TPB",
    "VCB",
    "VCI",
    "VDS",
    "VHM",
    "VIC",
    "VJC",
    "VND",
    "VNM",
    "VPB",
    "VTP",
]


def upgrade() -> None:
    bind = op.get_bind()
    existing = bind.execute(
        sa.text("SELECT id FROM universe_sets WHERE slug = :slug"), {"slug": _SLUG}
    ).scalar()
    if existing is not None:
        return  # already seeded (e.g. via seed_universes.py)

    bind.execute(
        sa.text(
            "INSERT INTO universe_sets "
            "(slug, name, description, market, is_locked, is_active, symbol_count, version) "
            "VALUES (:slug, :name, :desc, :market, :locked, :active, :count, :version)"
        ),
        {
            "slug": _SLUG,
            "name": "VN Stock Default",
            "desc": "Default VN equity universe (migrated from vn_stock.yaml, P5).",
            "market": "vn_stock",
            "locked": False,
            "active": True,
            "count": len(_SYMBOLS),
            "version": 1,
        },
    )
    universe_id = bind.execute(
        sa.text("SELECT id FROM universe_sets WHERE slug = :slug"), {"slug": _SLUG}
    ).scalar()
    for symbol in _SYMBOLS:
        bind.execute(
            sa.text("INSERT INTO universe_symbols (universe_id, symbol) VALUES (:uid, :sym)"),
            {"uid": universe_id, "sym": symbol},
        )


def downgrade() -> None:
    bind = op.get_bind()
    universe_id = bind.execute(
        sa.text("SELECT id FROM universe_sets WHERE slug = :slug"), {"slug": _SLUG}
    ).scalar()
    if universe_id is None:
        return
    bind.execute(
        sa.text("DELETE FROM universe_symbols WHERE universe_id = :uid"), {"uid": universe_id}
    )
    bind.execute(sa.text("DELETE FROM universe_sets WHERE id = :uid"), {"uid": universe_id})
