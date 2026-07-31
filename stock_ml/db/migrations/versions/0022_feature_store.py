"""Feature Store + Expression DSL: normalised feature tables, drop feature_set_catalog.

Revision ID: 0022
Revises: 0021
Create Date: 2026-06-01 18:00:00.000000

Replaces the single ``feature_set_catalog`` table with five normalised tables
(see docs/FEATURE_STORE_DSL_DESIGN.md §3):

  feature_def, feature_dep, feature_set, feature_set_member, feature_materialization

Existing ``feature_set_catalog`` rows are copied into ``feature_set`` (preserving
ids), ``strategy_templates.feature_set_id`` is repointed to ``feature_set.id``, and
the old catalog is dropped. Member rows + expressions are filled afterwards by
``python -m stock_ml.scripts.seed_features``.

Dialect-agnostic (PG + SQLite). Column counts are derived (COUNT(member)) — there
is no hardcoded ``column_count`` in the new schema.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _create_new_tables() -> None:
    op.create_table(
        "feature_def",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("expr", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("output_dtype", sa.String(16), nullable=False, server_default="float"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("expr_hash", sa.String(40), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feature_def"),
    )
    op.create_index("idx_feature_def_name", "feature_def", ["name"])
    op.create_index("idx_feature_def_kind", "feature_def", ["kind"])
    op.create_index("idx_feature_def_expr_hash", "feature_def", ["expr_hash"])

    op.create_table(
        "feature_dep",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.Integer(), nullable=False),
        sa.Column("depends_on_feature_id", sa.Integer(), nullable=True),
        sa.Column("depends_on_raw", sa.String(64), nullable=True),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_def.id"], ondelete="CASCADE", name="fk_feature_dep_feature"
        ),
        sa.ForeignKeyConstraint(
            ["depends_on_feature_id"],
            ["feature_def.id"],
            ondelete="CASCADE",
            name="fk_feature_dep_depends_on",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feature_dep"),
    )
    op.create_index("idx_feature_dep_feature", "feature_dep", ["feature_id"])

    op.create_table(
        "feature_set",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feature_set"),
    )
    op.create_index("idx_feature_set_name", "feature_set", ["name"])

    op.create_table(
        "feature_set_member",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_set_id", sa.Integer(), nullable=False),
        sa.Column("feature_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.ForeignKeyConstraint(
            ["feature_set_id"], ["feature_set.id"], ondelete="CASCADE", name="fk_fsm_set"
        ),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_def.id"], ondelete="CASCADE", name="fk_fsm_feature"
        ),
        sa.UniqueConstraint("feature_set_id", "feature_id", name="uq_feature_set_member"),
        sa.PrimaryKeyConstraint("id", name="pk_feature_set_member"),
    )
    op.create_index("idx_feature_set_member_set", "feature_set_member", ["feature_set_id"])

    op.create_table(
        "feature_materialization",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("feature_id", sa.Integer(), nullable=False),
        sa.Column("expr_hash", sa.String(40), nullable=False),
        sa.Column("data_version", sa.String(64), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("engine_version", sa.String(32), nullable=False, server_default=""),
        sa.Column(
            "computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature_def.id"], ondelete="CASCADE", name="fk_feature_mat_feature"
        ),
        sa.UniqueConstraint(
            "feature_id", "expr_hash", "data_version", name="uq_feature_materialization"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feature_materialization"),
    )
    op.create_index("idx_feature_mat_feature", "feature_materialization", ["feature_id"])


def upgrade() -> None:
    bind = op.get_bind()
    _create_new_tables()

    # Migrate existing feature_set_catalog rows into feature_set (preserve ids).
    bind.execute(
        sa.text(
            "INSERT INTO feature_set (id, name, version, description, is_active, created_at) "
            "SELECT id, name, 1, description, is_active, created_at FROM feature_set_catalog"
        )
    )
    if bind.dialect.name == "postgresql":
        bind.execute(
            sa.text(
                "SELECT setval(pg_get_serial_sequence('feature_set', 'id'), "
                "(SELECT COALESCE(MAX(id), 1) FROM feature_set))"
            )
        )

    # Repoint strategy_templates.feature_set_id FK to the new feature_set table.
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.drop_constraint("fk_template_feature_set", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_template_feature_set",
            "feature_set",
            ["feature_set_id"],
            ["id"],
            ondelete="RESTRICT",
        )

    op.drop_index("idx_feature_set_catalog_name", table_name="feature_set_catalog")
    op.drop_table("feature_set_catalog")


def downgrade() -> None:
    bind = op.get_bind()

    op.create_table(
        "feature_set_catalog",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("column_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("columns", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feature_set_catalog"),
    )
    op.create_index("idx_feature_set_catalog_name", "feature_set_catalog", ["name"])

    bind.execute(
        sa.text(
            "INSERT INTO feature_set_catalog (id, name, column_count, description, is_active, created_at) "
            "SELECT id, name, 0, description, is_active, created_at FROM feature_set"
        )
    )
    if bind.dialect.name == "postgresql":
        bind.execute(
            sa.text(
                "SELECT setval(pg_get_serial_sequence('feature_set_catalog', 'id'), "
                "(SELECT COALESCE(MAX(id), 1) FROM feature_set_catalog))"
            )
        )

    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.drop_constraint("fk_template_feature_set", type_="foreignkey")
        batch_op.create_foreign_key(
            "fk_template_feature_set",
            "feature_set_catalog",
            ["feature_set_id"],
            ["id"],
            ondelete="RESTRICT",
        )

    op.drop_index("idx_feature_mat_feature", table_name="feature_materialization")
    op.drop_table("feature_materialization")
    op.drop_index("idx_feature_set_member_set", table_name="feature_set_member")
    op.drop_table("feature_set_member")
    op.drop_index("idx_feature_set_name", table_name="feature_set")
    op.drop_table("feature_set")
    op.drop_index("idx_feature_dep_feature", table_name="feature_dep")
    op.drop_table("feature_dep")
    op.drop_index("idx_feature_def_expr_hash", table_name="feature_def")
    op.drop_index("idx_feature_def_kind", table_name="feature_def")
    op.drop_index("idx_feature_def_name", table_name="feature_def")
    op.drop_table("feature_def")
