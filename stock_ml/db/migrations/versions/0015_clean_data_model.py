"""Phase: Clean data model — eliminate redundant tables and denormalized columns.

Revision ID: 0015
Revises: 0014
Create Date: 2026-05-31 23:30:00.000000

Removes:
  - experiment_configs table (redundant with strategy_templates for DB-first runs)
  - template_runs table (redundant with leaderboard_runs.template_id FK)

Adds:
  - strategy_templates: config_hash, version (for config versioning)
  - leaderboard_runs: run_seed, raw_config, template_config_hash (from removed tables)

Removes from leaderboard_runs:
  - target_gain_threshold, target_loss_threshold
  - exit_model_type, exit_model_enabled
  - regime_model_type, size_model_type
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Clean data model."""
    # Step 1: Add config versioning to strategy_templates
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("config_hash", sa.String(64), nullable=False, server_default="")
        )
        batch_op.add_column(sa.Column("version", sa.Integer(), nullable=False, server_default="1"))

    # Step 2: Add new columns to leaderboard_runs
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("run_seed", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("template_config_hash", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("raw_config", sa.Text(), nullable=True))

    # Step 3: Drop denormalized columns from leaderboard_runs
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        batch_op.drop_column("target_gain_threshold")
        batch_op.drop_column("target_loss_threshold")
        batch_op.drop_column("exit_model_type")
        batch_op.drop_column("exit_model_enabled")
        batch_op.drop_column("regime_model_type")
        batch_op.drop_column("size_model_type")

    # Step 4: Drop template_runs table
    op.drop_table("template_runs")

    # Step 5: Drop experiment_configs table
    op.drop_table("experiment_configs")


def downgrade() -> None:
    """Restore tables and columns (minimal recovery)."""
    # Step 1: Recreate experiment_configs (minimal schema)
    op.create_table(
        "experiment_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(512), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("strategy", sa.String(255), nullable=True),
        sa.Column("market", sa.String(64), nullable=True),
        sa.Column("feature_set", sa.String(255), nullable=True),
        sa.Column("entry_model_type", sa.String(64), nullable=True),
        sa.Column("entry_model_params", sa.JSON(), nullable=True),
        sa.Column("exit_model", sa.JSON(), nullable=True),
        sa.Column("split_config", sa.JSON(), nullable=True),
        sa.Column("engine_config", sa.JSON(), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("signal_threshold", sa.Float(), nullable=True),
        sa.Column("signal_mode", sa.String(32), nullable=True),
        sa.Column("model_mode", sa.String(32), nullable=True),
        sa.Column("direction", sa.String(8), nullable=True),
        sa.Column("regime_model", sa.JSON(), nullable=True),
        sa.Column("size_model", sa.JSON(), nullable=True),
        sa.Column("yaml_schema_version", sa.Integer(), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=True),
        sa.Column("raw_yaml", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )

    # Step 2: Recreate template_runs (minimal schema)
    op.create_table(
        "template_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(512), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["template_id"], ["strategy_templates.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["leaderboard_runs.run_id"], ondelete="CASCADE"),
    )
    op.create_index("idx_template_runs_template", "template_runs", ["template_id"])
    op.create_index("idx_template_runs_run", "template_runs", ["run_id"])

    # Step 3: Restore columns to leaderboard_runs
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("target_gain_threshold", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("target_loss_threshold", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("exit_model_type", sa.String(64), nullable=False, server_default="none")
        )
        batch_op.add_column(
            sa.Column("exit_model_enabled", sa.Boolean(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column("regime_model_type", sa.String(64), nullable=False, server_default="none")
        )
        batch_op.add_column(
            sa.Column("size_model_type", sa.String(64), nullable=False, server_default="none")
        )

    # Step 4: Remove new columns from leaderboard_runs
    with op.batch_alter_table("leaderboard_runs", schema=None) as batch_op:
        batch_op.drop_column("raw_config")
        batch_op.drop_column("template_config_hash")
        batch_op.drop_column("run_seed")

    # Step 5: Remove versioning from strategy_templates
    with op.batch_alter_table("strategy_templates", schema=None) as batch_op:
        batch_op.drop_column("version")
        batch_op.drop_column("config_hash")
