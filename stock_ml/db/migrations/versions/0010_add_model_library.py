"""Add model library and strategy templates for DB-first experiment management.

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-31 20:00:00.000000

Phase: DB-First Model Library
- feature_set_catalog: register available feature sets
- target_catalog: reusable target definitions
- model_components: model library (entry/exit/regime/size slots)
- strategy_templates: experiment configs (replaces YAML)
- template_runs: tracks which template each run came from
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create model library and strategy template tables."""
    # feature_set_catalog: static registry of feature sets
    op.create_table(
        "feature_set_catalog",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("column_count", sa.Integer(), nullable=False),
        sa.Column("columns", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feature_set_catalog"),
    )
    op.create_index(
        "idx_feature_set_catalog_name",
        "feature_set_catalog",
        ["name"],
    )

    # target_catalog: reusable target definitions
    op.create_table(
        "target_catalog",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("type", sa.String(64), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("output_dtype", sa.String(16), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_target_catalog"),
    )
    op.create_index("idx_target_catalog_name", "target_catalog", ["name"])
    op.create_index("idx_target_catalog_type", "target_catalog", ["type"])

    # model_components: model library with entry/exit/regime/size slots
    op.create_table(
        "model_components",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("algorithm", sa.String(64), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_model_components"),
    )
    op.create_index("idx_model_components_name", "model_components", ["name"])
    op.create_index("idx_model_components_role", "model_components", ["role"])
    op.create_index("idx_model_components_algorithm", "model_components", ["algorithm"])
    op.create_index(
        "idx_model_components_role_algo",
        "model_components",
        ["role", "algorithm"],
    )

    # strategy_templates: experiment configs (replaces YAML)
    op.create_table(
        "strategy_templates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("market", sa.String(64), nullable=False),
        sa.Column("strategy", sa.String(255), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False, server_default="long"),
        sa.Column("universe_slug", sa.String(128), nullable=True),
        # Components (FK)
        sa.Column("feature_set_id", sa.Integer(), nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("entry_component_id", sa.Integer(), nullable=False),
        sa.Column("exit_component_id", sa.Integer(), nullable=True),
        sa.Column("regime_component_id", sa.Integer(), nullable=True),
        sa.Column("size_component_id", sa.Integer(), nullable=True),
        # Signal config
        sa.Column("signal_mode", sa.String(32), nullable=False, server_default="entry_first"),
        sa.Column("signal_threshold", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("model_mode", sa.String(32), nullable=False, server_default="ml_only"),
        # Execution configs (JSON)
        sa.Column("split_config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("engine_config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("validation_config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("seed", sa.Integer(), nullable=False, server_default="42"),
        # Status
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default="2"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["feature_set_id"],
            ["feature_set_catalog.id"],
            ondelete="RESTRICT",
            name="fk_template_feature_set",
        ),
        sa.ForeignKeyConstraint(
            ["target_id"],
            ["target_catalog.id"],
            ondelete="RESTRICT",
            name="fk_template_target",
        ),
        sa.ForeignKeyConstraint(
            ["entry_component_id"],
            ["model_components.id"],
            ondelete="RESTRICT",
            name="fk_template_entry_component",
        ),
        sa.ForeignKeyConstraint(
            ["exit_component_id"],
            ["model_components.id"],
            ondelete="SET NULL",
            name="fk_template_exit_component",
        ),
        sa.ForeignKeyConstraint(
            ["regime_component_id"],
            ["model_components.id"],
            ondelete="SET NULL",
            name="fk_template_regime_component",
        ),
        sa.ForeignKeyConstraint(
            ["size_component_id"],
            ["model_components.id"],
            ondelete="SET NULL",
            name="fk_template_size_component",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_strategy_templates"),
    )
    op.create_index("idx_template_name", "strategy_templates", ["name"])
    op.create_index("idx_template_market", "strategy_templates", ["market"])
    op.create_index("idx_template_feature_set", "strategy_templates", ["feature_set_id"])
    op.create_index("idx_template_entry_component", "strategy_templates", ["entry_component_id"])

    # template_runs: join table tracking which template each run came from
    op.create_table(
        "template_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(512), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["template_id"],
            ["strategy_templates.id"],
            ondelete="CASCADE",
            name="fk_template_runs_template",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["leaderboard_runs.run_id"],
            ondelete="CASCADE",
            name="fk_template_runs_run",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_template_runs"),
    )
    op.create_index("idx_template_runs_template", "template_runs", ["template_id"])
    op.create_index("idx_template_runs_run", "template_runs", ["run_id"])


def downgrade() -> None:
    """Drop model library and strategy template tables."""
    op.drop_index("idx_template_runs_run", table_name="template_runs")
    op.drop_index("idx_template_runs_template", table_name="template_runs")
    op.drop_table("template_runs")

    op.drop_index("idx_template_entry_component", table_name="strategy_templates")
    op.drop_index("idx_template_feature_set", table_name="strategy_templates")
    op.drop_index("idx_template_market", table_name="strategy_templates")
    op.drop_index("idx_template_name", table_name="strategy_templates")
    op.drop_table("strategy_templates")

    op.drop_index("idx_model_components_role_algo", table_name="model_components")
    op.drop_index("idx_model_components_algorithm", table_name="model_components")
    op.drop_index("idx_model_components_role", table_name="model_components")
    op.drop_index("idx_model_components_name", table_name="model_components")
    op.drop_table("model_components")

    op.drop_index("idx_target_catalog_type", table_name="target_catalog")
    op.drop_index("idx_target_catalog_name", table_name="target_catalog")
    op.drop_table("target_catalog")

    op.drop_index("idx_feature_set_catalog_name", table_name="feature_set_catalog")
    op.drop_table("feature_set_catalog")
