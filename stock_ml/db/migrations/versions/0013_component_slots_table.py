"""Refactor: Migrate 8 dual-component FK columns to component_slots join table.

Revision ID: 0013
Revises: 0012
Create Date: 2026-05-31 22:00:00.000000

Replaces the 12 FK columns on strategy_templates (4 old + 8 dual) with a
dedicated component_slots table (1 row per slot: entry/exit/regime/size).

Data migration: Each template → 4 rows in component_slots, one per slot type.
ML/rule component FKs copied from the old 8 dual FK columns.

After migration: Drop all 12 FK columns from strategy_templates.
Result: Cleaner schema, easier to query, supports arbitrary slot extensions.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create component_slots table, migrate data, drop old FK columns."""

    # Step 1: Create component_slots table
    op.create_table(
        "component_slots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("slot_type", sa.String(8), nullable=False),
        sa.Column("ml_component_id", sa.Integer(), nullable=True),
        sa.Column("rule_component_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["template_id"], ["strategy_templates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["ml_component_id"], ["model_components.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["rule_component_id"], ["model_components.id"], ondelete="SET NULL"
        ),
    )
    op.create_index("idx_component_slots_template", "component_slots", ["template_id"])
    op.create_index("idx_component_slots_slot_type", "component_slots", ["slot_type"])

    # Step 2: Data migration — copy from old dual FK columns
    conn = op.get_bind()

    templates = conn.execute(
        text("""
        SELECT id,
               entry_ml_component_id, entry_rule_component_id,
               exit_ml_component_id, exit_rule_component_id,
               regime_ml_component_id, regime_rule_component_id,
               size_ml_component_id, size_rule_component_id
        FROM strategy_templates
        """)
    )

    for template in templates:
        tmpl_id = template[0]
        slots_data = [
            ("entry", template[1], template[2]),  # ml, rule
            ("exit", template[3], template[4]),
            ("regime", template[5], template[6]),
            ("size", template[7], template[8]),
        ]

        for slot_type, ml_id, rule_id in slots_data:
            if ml_id is not None or rule_id is not None:
                conn.execute(
                    text("""
                    INSERT INTO component_slots (template_id, slot_type, ml_component_id, rule_component_id)
                    VALUES (:template_id, :slot_type, :ml_component_id, :rule_component_id)
                    """),
                    {
                        "template_id": tmpl_id,
                        "slot_type": slot_type,
                        "ml_component_id": ml_id,
                        "rule_component_id": rule_id,
                    },
                )

    # Step 3: Drop old FK columns from strategy_templates
    is_sqlite = op.get_bind().dialect.name == "sqlite"
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        # Drop the 8 dual-component FK columns
        batch_op.drop_column("entry_ml_component_id")
        batch_op.drop_column("entry_rule_component_id")
        batch_op.drop_column("exit_ml_component_id")
        batch_op.drop_column("exit_rule_component_id")
        batch_op.drop_column("regime_ml_component_id")
        batch_op.drop_column("regime_rule_component_id")
        batch_op.drop_column("size_ml_component_id")
        batch_op.drop_column("size_rule_component_id")

        # Drop the 4 old backward-compat FK columns
        batch_op.drop_column("entry_component_id")
        batch_op.drop_column("exit_component_id")
        batch_op.drop_column("regime_component_id")
        batch_op.drop_column("size_component_id")

        if is_sqlite:
            # SQLite batch mode rebuilds the table; the index on the now-dropped
            # entry_component_id must be removed here, or the rebuild fails trying
            # to recreate it ("no such column: entry_component_id").
            batch_op.drop_index("idx_template_entry_component")

    if not is_sqlite:
        # On Postgres the index was dropped automatically with its column; guard
        # with IF EXISTS so this stays a safe no-op.
        op.execute("DROP INDEX IF EXISTS idx_template_entry_component")


def downgrade() -> None:
    """Recreate old dual-component columns, drop component_slots table."""

    # Step 1: Recreate 8 dual-component FK columns
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.add_column(sa.Column("entry_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("entry_rule_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("exit_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("exit_rule_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("regime_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("regime_rule_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("size_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("size_rule_component_id", sa.Integer(), nullable=True))

        # Create FKs
        batch_op.create_foreign_key(
            "fk_template_entry_ml_component",
            "model_components",
            ["entry_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_entry_rule_component",
            "model_components",
            ["entry_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_exit_ml_component",
            "model_components",
            ["exit_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_exit_rule_component",
            "model_components",
            ["exit_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_regime_ml_component",
            "model_components",
            ["regime_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_regime_rule_component",
            "model_components",
            ["regime_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_size_ml_component",
            "model_components",
            ["size_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_template_size_rule_component",
            "model_components",
            ["size_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # Step 2: Data reverse migration (from component_slots back to FK columns)
    conn = op.get_bind()

    templates = conn.execute(text("SELECT DISTINCT template_id FROM component_slots"))

    for (tmpl_id,) in templates:
        slots = conn.execute(
            text("""
            SELECT slot_type, ml_component_id, rule_component_id
            FROM component_slots
            WHERE template_id = :template_id
            """),
            {"template_id": tmpl_id},
        )

        for slot_type, ml_id, rule_id in slots:
            col_prefix = f"{slot_type}"
            conn.execute(
                text(f"""
                UPDATE strategy_templates
                SET {col_prefix}_ml_component_id = :ml_id,
                    {col_prefix}_rule_component_id = :rule_id
                WHERE id = :tmpl_id
                """),
                {"ml_id": ml_id, "rule_id": rule_id, "tmpl_id": tmpl_id},
            )

    # Step 3: Drop component_slots table
    op.drop_table("component_slots")
