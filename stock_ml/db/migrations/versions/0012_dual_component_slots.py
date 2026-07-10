"""Add dual-component slots (ML + Rule per slot).

Revision ID: 0012
Revises: 0011
Create Date: 2026-05-31 21:00:00.000000

Introduces flexible dual-component model: each slot (entry/exit/regime/size)
can now have separate ML and Rule components working together.

Data migration: backfills new FK columns from old single-component FKs based on
component algorithm type. Old columns kept nullable for backward compat.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ML_ALGORITHMS = {"lightgbm", "xgboost", "random_forest", "mlp"}


def upgrade() -> None:
    """Add component_type to model_components, add 8 dual FKs to strategy_templates."""
    # Step 1: Add component_type to model_components
    with op.batch_alter_table("model_components", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("component_type", sa.String(8), nullable=False, server_default="ml")
        )
        batch_op.create_index("idx_model_components_type", ["component_type"])

    # Step 2: Backfill component_type based on algorithm
    conn = op.get_bind()
    conn.execute(text("UPDATE model_components SET component_type='rule' WHERE algorithm='rule'"))

    # Step 3: Add 8 new FK columns to strategy_templates
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.add_column(sa.Column("entry_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("entry_rule_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("exit_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("exit_rule_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("regime_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("regime_rule_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("size_ml_component_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("size_rule_component_id", sa.Integer(), nullable=True))

    # Step 4: Create foreign keys for new columns
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_templates_entry_ml_component",
            "model_components",
            ["entry_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_entry_rule_component",
            "model_components",
            ["entry_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_exit_ml_component",
            "model_components",
            ["exit_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_exit_rule_component",
            "model_components",
            ["exit_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_regime_ml_component",
            "model_components",
            ["regime_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_regime_rule_component",
            "model_components",
            ["regime_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_size_ml_component",
            "model_components",
            ["size_ml_component_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_templates_size_rule_component",
            "model_components",
            ["size_rule_component_id"],
            ["id"],
            ondelete="SET NULL",
        )

    # Step 5: Data migration - backfill new FKs from old single-component FKs
    _migrate_data_from_single_to_dual_component(conn)

    # Step 6: Make entry_component_id nullable for backward compat with new templates
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.alter_column("entry_component_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    """Revert dual-component slots."""
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.alter_column("entry_component_id", existing_type=sa.Integer(), nullable=False)

        # Drop foreign keys for new columns
        batch_op.drop_constraint("fk_templates_entry_ml_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_entry_rule_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_exit_ml_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_exit_rule_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_regime_ml_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_regime_rule_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_size_ml_component", type_="foreignkey")
        batch_op.drop_constraint("fk_templates_size_rule_component", type_="foreignkey")

        # Drop new columns
        batch_op.drop_column("entry_ml_component_id")
        batch_op.drop_column("entry_rule_component_id")
        batch_op.drop_column("exit_ml_component_id")
        batch_op.drop_column("exit_rule_component_id")
        batch_op.drop_column("regime_ml_component_id")
        batch_op.drop_column("regime_rule_component_id")
        batch_op.drop_column("size_ml_component_id")
        batch_op.drop_column("size_rule_component_id")

    with op.batch_alter_table("model_components", schema=None) as batch_op:
        batch_op.drop_index("idx_model_components_type")
        batch_op.drop_column("component_type")


def _migrate_data_from_single_to_dual_component(conn) -> None:
    """Backfill new dual FK columns from old single-component FKs.

    Logic:
    - Read old {slot}_component_id
    - Join with model_components to check algorithm type
    - If algorithm in ML_ALGORITHMS → {slot}_ml_component_id = component_id
    - If algorithm == 'rule' → {slot}_rule_component_id = component_id
    """
    slots = ["entry", "exit", "regime", "size"]

    for slot in slots:
        old_id_col = f"{slot}_component_id"
        ml_col = f"{slot}_ml_component_id"
        rule_col = f"{slot}_rule_component_id"

        # Query: fetch templates with old {slot}_component_id and its algorithm
        result = conn.execute(
            text(f"""
                SELECT st.id, st.{old_id_col}, mc.algorithm
                FROM strategy_templates st
                LEFT JOIN model_components mc ON mc.id = st.{old_id_col}
                WHERE st.{old_id_col} IS NOT NULL
            """)
        )

        for row in result:
            template_id, component_id, algorithm = row
            if algorithm in ML_ALGORITHMS:
                conn.execute(
                    text(f"UPDATE strategy_templates SET {ml_col} = :cid WHERE id = :tid"),
                    {"cid": component_id, "tid": template_id},
                )
            elif algorithm == "rule":
                conn.execute(
                    text(f"UPDATE strategy_templates SET {rule_col} = :cid WHERE id = :tid"),
                    {"cid": component_id, "tid": template_id},
                )
