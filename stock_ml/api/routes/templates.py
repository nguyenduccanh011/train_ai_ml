"""API routes for strategy templates."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.dependencies import get_db
from stock_ml.db.repositories.run_repo import LeaderboardRunRepository
from stock_ml.db.repositories.template_repo import StrategyTemplateRepository

router = APIRouter(prefix="/api/v1/templates", tags=["templates"])


def _template_to_dict(template):
    """Convert template model to dict."""
    # Build componentSlots array (Phase 0.4: per-slot features/targets)
    component_slots = []
    for slot in template.component_slots:
        component_slots.append(
            {
                "slotType": slot.slot_type,
                "mlComponentId": slot.ml_component_id,
                "mlComponent": {"id": slot.ml_component.id, "name": slot.ml_component.name}
                if slot.ml_component
                else None,
                "ruleComponentId": slot.rule_component_id,
                "ruleComponent": {"id": slot.rule_component.id, "name": slot.rule_component.name}
                if slot.rule_component
                else None,
                "featureSetName": slot.feature_set_name,  # NEW
                "targetConfig": slot.target_config,  # NEW
            }
        )

    result = {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "hypothesis": template.hypothesis,
        "market": template.market,
        "strategy": template.strategy,
        "direction": template.direction,
        "universeSlug": template.universe_slug,
        "featureSetId": template.feature_set_id,
        "targetId": template.target_id,
        "componentSlots": component_slots,
        "featureSet": {"id": template.feature_set.id, "name": template.feature_set.name}
        if template.feature_set
        else None,
        "target": {"id": template.target.id, "name": template.target.name}
        if template.target
        else None,
        "signalMode": template.signal_mode,
        "signalThreshold": template.signal_threshold,
        "entryThreshold": template.entry_threshold,
        "exitThreshold": template.exit_threshold,
        "modelMode": template.model_mode,
        "splitConfig": template.split_config,
        "engineConfig": template.engine_config,
        "validationConfig": template.validation_config,
        "seed": template.seed,
        "isActive": template.is_active,
        "createdAt": template.created_at.isoformat() if template.created_at else None,
        "updatedAt": template.updated_at.isoformat() if template.updated_at else None,
    }
    return result


@router.get("/")
async def list_templates(
    market: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    """List strategy templates."""
    repo = StrategyTemplateRepository(session)
    if market:
        templates = await repo.list_by_market(market)
    else:
        templates = await repo.list_all(is_active=True)
    return [_template_to_dict(t) for t in templates]


@router.post("/")
async def create_template(
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Create a new strategy template with component slots."""
    required = {
        "name",
        "market",
        "strategy",
        "featureSetId",
        "targetId",
        "componentSlots",
    }
    if not required.issubset(body.keys()):
        raise HTTPException(status_code=400, detail=f"Missing required fields: {required}")

    repo = StrategyTemplateRepository(session)
    existing = await repo.get_by_name(body["name"])
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Template '{body['name']}' already exists",
        )

    # Validate and build component_slots
    component_slots = body.get("componentSlots", [])
    if not isinstance(component_slots, list):
        raise HTTPException(status_code=400, detail="componentSlots must be a list")

    # Create template with component slots
    try:
        template = await repo.create(
            name=body["name"],
            market=body["market"],
            strategy=body["strategy"],
            feature_set_id=body["featureSetId"],
            target_id=body["targetId"],
            component_slots=component_slots,
            direction=body.get("direction", "long"),
            signal_mode=body.get("signalMode", "entry_first"),
            signal_threshold=body.get("signalThreshold", 0.0),
            entry_threshold=body.get("entryThreshold"),
            exit_threshold=body.get("exitThreshold"),
            model_mode=body.get("modelMode"),  # None = auto-infer from components
            split_config=body.get("splitConfig", {}),
            engine_config=body.get("engineConfig", {}),
            validation_config=body.get("validationConfig", {}),
            seed=body.get("seed", 42),
            description=body.get("description"),
            hypothesis=body.get("hypothesis"),
            universe_slug=body.get("universeSlug"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    await session.commit()

    # Reload to get updated relationships
    template = await repo.get_by_id(template.id)
    return _template_to_dict(template)


@router.get("/{template_id}")
async def get_template(template_id: int, session: AsyncSession = Depends(get_db)):
    """Get template by ID with all relationships."""
    repo = StrategyTemplateRepository(session)
    template = await repo.get_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return _template_to_dict(template)


@router.put("/{template_id}")
async def update_template(
    template_id: int,
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Update template."""
    repo = StrategyTemplateRepository(session)
    template = await repo.update(
        template_id,
        description=body.get("description"),
        hypothesis=body.get("hypothesis"),
        direction=body.get("direction"),
        signal_mode=body.get("signalMode"),
        signal_threshold=body.get("signalThreshold"),
        entry_threshold=body.get("entryThreshold"),
        exit_threshold=body.get("exitThreshold"),
        model_mode=body.get("modelMode"),
        split_config=body.get("splitConfig"),
        engine_config=body.get("engineConfig"),
        validation_config=body.get("validationConfig"),
        seed=body.get("seed"),
    )
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    await session.commit()
    return _template_to_dict(template)


@router.delete("/{template_id}")
async def delete_template(template_id: int, session: AsyncSession = Depends(get_db)):
    """Soft delete template."""
    repo = StrategyTemplateRepository(session)
    success = await repo.soft_delete(template_id)
    if not success:
        raise HTTPException(status_code=404, detail="Template not found")
    await session.commit()
    return {"deleted": True}


@router.get("/{template_id}/runs")
async def list_template_runs(template_id: int, session: AsyncSession = Depends(get_db)):
    """List all runs from a template."""
    # First check template exists
    template_repo = StrategyTemplateRepository(session)
    template = await template_repo.get_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # List runs from leaderboard_runs
    runs_repo = LeaderboardRunRepository(session)
    runs = await runs_repo.list_by_template(template_id)
    return [
        {
            "runId": r.run_id,
            "state": r.state,
            "composite_score": r.composite_score,
            "market": r.market,
            "strategy": r.strategy,
            "feature_set": r.feature_set,
            "entry_model": r.entry_model,
            "total_pnl": r.total_pnl,
            "max_drawdown": r.max_drawdown,
            "sharpe": r.sharpe,
            "n_symbols": r.n_symbols,
            "run_seed": r.run_seed,
            "generated_at": r.generated_at.isoformat() if r.generated_at else None,
        }
        for r in runs
    ]


@router.post("/{template_id}/submit")
async def submit_template_run(
    template_id: int,
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Submit a template to experiment queue (DB-First Phase 0-3).

    Converts template to experiment config and queues for execution.
    Returns job_id for polling progress.

    Request body (optional):
    {
        "override_seed": 42,  # Override template seed
        "override_symbols": "AAA,SSI"  # Override universe
    }
    """
    import subprocess
    import sys

    from stock_ml.src.pipeline.experiment import ExperimentConfig

    template_repo = StrategyTemplateRepository(session)
    template = await template_repo.get_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    try:
        # Convert template to experiment config
        config: ExperimentConfig = await ExperimentConfig.from_template_id_async(
            template_id, session
        )

        # Apply overrides from request body
        if body and body.get("override_seed"):
            config.seed = int(body["override_seed"])

        # Convert to dict for subprocess execution
        config_dict = {
            "name": config.name,
            "strategy": config.strategy,
            "market": config.market,
            "seed": config.seed,
            "direction": config.direction,
            "signal_threshold": config.signal_threshold,
            "entry_threshold": config.entry_threshold,
            "exit_threshold": config.exit_threshold,
            "model_mode": config.model_mode,
            "components": {
                "features": config.feature_set,
                "target": config.target,
                "entry_model": config.entry_model,
                "exit_model": config.exit_model,
                "signal_mode": config.signal_mode,
            },
            "split": config.split,
            "engine": config.engine,
        }

        if config.validation:
            config_dict["validation"] = config.validation

        if config.universe:
            config_dict["universe"] = config.universe

        # Queue job via subprocess (async)
        from .experiments import _stock_ml_root
        from .jobs import attach_process, register_job

        stock_ml_root = _stock_ml_root()
        job_id = f"tmpl_{template_id}_{template.name}"
        log_path = stock_ml_root / "logs" / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        register_job(
            job_id,
            {
                "job_id": job_id,
                "type": "template",
                "template_id": template_id,
                "template_name": template.name,
                "status": "queued",
                "log": str(log_path),
            },
        )

        # Start experiment runner subprocess
        try:
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Starting template run: stock_ml_root={stock_ml_root}, cwd={stock_ml_root.parent}"
            )

            log_fh = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "stock_ml.scripts.run_template",
                    "--template-id",
                    str(template_id),
                    "--seed",
                    str(config.seed),
                ],
                cwd=str(stock_ml_root.parent),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            logger.info(f"Subprocess started: pid={proc.pid}")
            attach_process(job_id, proc, log_path)
        except Exception as e:
            log_path.unlink(missing_ok=True)
            logger.error(f"Failed to start subprocess: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to start template run: {e}") from e

        return {
            "job_id": job_id,
            "template_id": template_id,
            "template_name": template.name,
            "status": "queued",
            "log_path": str(log_path),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid template config: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process template: {e}") from e


@router.post("/{template_id}/export-yaml")
async def export_template_yaml(template_id: int, session: AsyncSession = Depends(get_db)):
    """Export template as YAML for backward compatibility."""
    repo = StrategyTemplateRepository(session)
    template = await repo.get_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Get entry and exit components from slots
    def _get_slot(slot_type):
        return next((s for s in template.component_slots if s.slot_type == slot_type), None)

    entry_slot = _get_slot("entry")
    if not entry_slot or (not entry_slot.ml_component and not entry_slot.rule_component):
        raise HTTPException(status_code=400, detail="Template has no entry component")

    # Build entry_model dict
    entry_model = {}
    if entry_slot.ml_component:
        entry_model["type"] = entry_slot.ml_component.algorithm
        entry_model["params"] = entry_slot.ml_component.params
    if entry_slot.rule_component:
        entry_model["rule_conditions"] = entry_slot.rule_component.params.get("conditions", [])
        entry_model["rule_logic"] = entry_slot.rule_component.params.get("logic", "AND")

    yaml_dict = {
        "name": template.name,
        "hypothesis": template.hypothesis or "",
        "strategy": template.strategy,
        "market": template.market,
        "seed": template.seed,
        "direction": template.direction,
        "signal_threshold": template.signal_threshold,
        **({"entry_threshold": template.entry_threshold} if template.entry_threshold is not None else {}),
        **({"exit_threshold": template.exit_threshold} if template.exit_threshold is not None else {}),
        "model_mode": template.model_mode,
        "components": {
            "features": template.feature_set.name,
            "signal_mode": template.signal_mode,
            "target": {
                "type": template.target.type,
                **template.target.params,
            },
            "entry_model": entry_model,
        },
        "split": template.split_config,
        "engine": template.engine_config,
    }

    # Add exit model if exists
    exit_slot = _get_slot("exit")
    if exit_slot and (exit_slot.ml_component or exit_slot.rule_component):
        exit_model = {}
        if exit_slot.ml_component:
            exit_model["type"] = exit_slot.ml_component.algorithm
            exit_model["params"] = exit_slot.ml_component.params
        if exit_slot.rule_component:
            exit_model["rule_conditions"] = exit_slot.rule_component.params.get("conditions", [])
            exit_model["rule_logic"] = exit_slot.rule_component.params.get("logic", "AND")
        exit_model["enabled"] = True
        yaml_dict["components"]["exit_model"] = exit_model

    if template.validation_config:
        yaml_dict["validation"] = template.validation_config

    if template.universe_slug:
        yaml_dict["components"]["universe"] = {
            "mode": "db",
            "slug": template.universe_slug,
        }

    import yaml

    yaml_str = yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)
    return {"yaml": yaml_str}
