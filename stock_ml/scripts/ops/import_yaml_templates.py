"""Import existing YAML experiments into DB template library (Phase 0-3).

This script migrates experiments from YAML files to the strategy_templates table,
creating model components and target catalog entries as needed. It enables backward
compatibility while establishing the DB-first workflow.

Usage:
    python stock_ml/scripts/import_yaml_templates.py \\
        --yaml-dir config/experiments/done \\
        --dry-run  # preview changes without committing

    python stock_ml/scripts/import_yaml_templates.py \\
        --yaml-dir config/experiments/done \\
        --commit   # actually insert into DB
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

import click
import yaml
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from stock_ml.db.models import (
    ModelComponentModel,
    StrategyTemplateModel,
    TargetCatalogModel,
)
from stock_ml.db.repositories.feature_repo import FeatureSetRepository
from stock_ml.db.repositories.template_repo import (
    ModelComponentRepository,
    StrategyTemplateRepository,
    TargetCatalogRepository,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def config_hash(cfg_dict: dict) -> str:
    """Compute config hash for matching with leaderboard_runs."""
    s = yaml.dump(cfg_dict, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(s.encode()).hexdigest()


async def init_feature_sets(session: AsyncSession) -> dict[str, int]:
    """Resolve known feature sets to their IDs (seeded by seed_features.py).

    Feature sets are now defined in the normalised feature_set/feature_def tables
    via ``python -m stock_ml.scripts.seed_features``. This only maps names → IDs.

    Returns:
        mapping from feature set name to ID
    """
    repo = FeatureSetRepository(session)
    names = ["basic_v1", "leading_v2", "leading_v3"]
    result: dict[str, int] = {}
    for name in names:
        fs = await repo.get_by_name(name)
        if fs is None:
            raise RuntimeError(
                f"Feature set '{name}' not found. Run "
                "`python -m stock_ml.scripts.seed_features` before importing templates."
            )
        result[name] = fs.id
        logger.info(f"Feature set '{name}' resolved (id={fs.id})")
    return result


async def get_or_create_target(session: AsyncSession, target_cfg: dict) -> TargetCatalogModel:
    """Get or create target catalog entry."""
    repo = TargetCatalogRepository(session)
    target_type = target_cfg.get("type")

    # Create a unique name based on type + params
    params_key = "_".join(f"{k}={v}" for k, v in sorted(target_cfg.items()) if k != "type")
    name = f"{target_type}_{params_key}" if params_key else target_type
    name = name.replace(".", "_").replace("/", "_")[:255]  # Truncate to 255 chars

    existing = await repo.get_by_name(name)
    if existing:
        return existing

    # Infer output_dtype
    output_dtype = "regression" if "regression" in target_type else "classification"
    params = {k: v for k, v in target_cfg.items() if k != "type"}

    target = await repo.create(
        name=name,
        type=target_type,
        params=params,
        output_dtype=output_dtype,
        description=f"Target {target_type} from YAML import",
    )
    logger.info(f"Created target '{name}' (id={target.id}, type={target_type})")
    return target


async def get_or_create_component(
    session: AsyncSession, role: str, algorithm: str, params: dict
) -> ModelComponentModel:
    """Get or create model component."""
    repo = ModelComponentRepository(session)

    # Create unique name from role, algorithm, and param hash
    params_key = hashlib.md5(yaml.dump(params, sort_keys=True).encode()).hexdigest()[:8]
    name = f"{role}_{algorithm}_{params_key}"

    existing = await repo.get_by_name(name)
    if existing:
        return existing

    component = await repo.create(
        name=name,
        role=role,
        algorithm=algorithm,
        params=params,
        description=f"{role.capitalize()} model: {algorithm} (YAML import)",
    )
    logger.info(f"Created component '{name}' (id={component.id})")
    return component


async def import_yaml_file(
    session: AsyncSession,
    yaml_path: Path,
    feature_set_ids: dict[str, int],
) -> StrategyTemplateModel | None:
    """Import a single YAML experiment file."""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to parse {yaml_path}: {e}")
        return None

    # Validate required fields
    required = ["name", "strategy", "market", "components", "split", "engine"]
    if not all(k in cfg for k in required):
        logger.warning(f"Skipping {yaml_path}: missing required fields")
        return None

    comp = cfg.get("components", {})
    feature_name = comp.get("features", "basic_v1")
    if feature_name not in feature_set_ids:
        logger.warning(f"Skipping {yaml_path}: unknown feature set '{feature_name}'")
        return None

    try:
        # Get/create target
        target_cfg = comp.get("target", {})
        target = await get_or_create_target(session, target_cfg)

        # Get/create entry component
        entry_cfg = comp.get("entry_model", {})
        entry_algo = entry_cfg.get("type", "lightgbm")
        entry_params = entry_cfg.get("params", {})
        entry_component = await get_or_create_component(session, "entry", entry_algo, entry_params)

        # Get/create exit component (optional)
        exit_component = None
        exit_cfg = comp.get("exit_model", {})
        if exit_cfg and exit_cfg.get("enabled", False):
            exit_algo = exit_cfg.get("type", "lightgbm")
            exit_params = exit_cfg.get("params", {})
            exit_component = await get_or_create_component(session, "exit", exit_algo, exit_params)

        # Get/create regime component (optional, usually stub)
        regime_component = None
        regime_cfg = comp.get("regime_model", {})
        if regime_cfg and regime_cfg.get("enabled", False):
            regime_algo = regime_cfg.get("type", "none")
            if regime_algo != "none":
                regime_params = regime_cfg.get("params", {})
                regime_component = await get_or_create_component(
                    session, "regime", regime_algo, regime_params
                )

        # Get/create size component (optional, usually stub)
        size_component = None
        size_cfg = comp.get("size_model", {})
        if size_cfg and size_cfg.get("enabled", False):
            size_algo = size_cfg.get("type", "none")
            if size_algo != "none":
                size_params = size_cfg.get("params", {})
                size_component = await get_or_create_component(
                    session, "size", size_algo, size_params
                )

        # Check if template already exists
        repo = StrategyTemplateRepository(session)
        existing = await repo.get_by_name(cfg["name"])
        if existing:
            logger.info(f"Template '{cfg['name']}' already exists (id={existing.id}), skipping")
            return existing

        # Build component_slots list (all YAML components = ML type by default)
        component_slots = []
        if entry_component:
            component_slots.append(
                {
                    "slot_type": "entry",
                    "ml_component_id": entry_component.id,
                    "rule_component_id": None,
                }
            )
        if exit_component:
            component_slots.append(
                {
                    "slot_type": "exit",
                    "ml_component_id": exit_component.id,
                    "rule_component_id": None,
                }
            )
        if regime_component:
            component_slots.append(
                {
                    "slot_type": "regime",
                    "ml_component_id": regime_component.id,
                    "rule_component_id": None,
                }
            )
        if size_component:
            component_slots.append(
                {
                    "slot_type": "size",
                    "ml_component_id": size_component.id,
                    "rule_component_id": None,
                }
            )

        # Create strategy template
        template = await repo.create(
            name=cfg["name"],
            market=cfg["market"],
            strategy=cfg["strategy"],
            feature_set_id=feature_set_ids[feature_name],
            target_id=target.id,
            component_slots=component_slots,
            direction=cfg.get("direction", "long"),
            signal_mode=comp.get("signal_mode", "entry_first"),
            signal_threshold=cfg.get("signal_threshold", 0.0),
            model_mode=cfg.get("model_mode", "ml_only"),
            split_config=cfg.get("split", {}),
            engine_config=cfg.get("engine", {}),
            validation_config=cfg.get("validation"),
            seed=cfg.get("seed", 42),
            description="Auto-imported from YAML",
            hypothesis=cfg.get("hypothesis", ""),
            universe_slug=comp.get("universe", {}).get("slug") if comp.get("universe") else None,
        )
        logger.info(f"Created template '{cfg['name']}' from {yaml_path.name} (id={template.id})")
        return template

    except Exception as e:
        logger.error(f"Failed to import {yaml_path}: {e}", exc_info=True)
        return None


@click.command()
@click.option(
    "--yaml-dir",
    type=click.Path(exists=True),
    default="config/experiments/done",
    help="Directory containing YAML experiment configs",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without committing",
)
@click.option(
    "--commit",
    is_flag=True,
    help="Actually insert into DB",
)
@click.option(
    "--db-url",
    type=str,
    default=None,
    help="Database URL (defaults to STOCK_ML_DB_URL env var)",
)
def main(yaml_dir: str, dry_run: bool, commit: bool, db_url: str):
    """Import YAML experiments into DB template library."""
    asyncio.run(_main_async(yaml_dir, dry_run, commit, db_url))


async def _main_async(yaml_dir: str, dry_run: bool, commit: bool, db_url: str):
    """Async implementation of main."""
    if dry_run and commit:
        click.echo("Error: cannot specify both --dry-run and --commit")
        return

    # Get DB URL
    if not db_url:
        import os

        db_url = os.getenv("STOCK_ML_DB_URL", "sqlite+aiosqlite:///./stock_ml.db")

    click.echo(f"Database: {db_url}")
    click.echo(f"YAML directory: {yaml_dir}")
    click.echo(f"Mode: {'DRY RUN' if dry_run else 'COMMIT' if commit else 'PREVIEW'}")
    click.echo()

    # Connect to DB
    engine = create_async_engine(db_url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # Initialize feature sets
        feature_set_ids = await init_feature_sets(session)

        if dry_run or commit:
            await session.commit()

        # Import YAML files
        yaml_path = Path(yaml_dir)
        if not yaml_path.exists():
            logger.error(f"YAML directory not found: {yaml_path}")
            return

        yaml_files = sorted(yaml_path.glob("*.yaml"))
        logger.info(f"Found {len(yaml_files)} YAML files to import")

        imported_count = 0
        for yaml_file in yaml_files:
            template = await import_yaml_file(session, yaml_file, feature_set_ids)
            if template:
                imported_count += 1

        click.echo()
        click.echo(f"Imported {imported_count} / {len(yaml_files)} YAML files")

        if dry_run:
            click.echo("DRY RUN: rolling back changes")
            await session.rollback()
        elif commit:
            click.echo("Committing to database...")
            await session.commit()
            click.echo("Import complete!")
        else:
            click.echo("Changes not saved (use --commit to save)")
            await session.rollback()

    await engine.dispose()


if __name__ == "__main__":
    main()
