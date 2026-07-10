"""API routes for model library (components, catalog)."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from stock_ml.db.dependencies import get_db
from stock_ml.db.repositories.feature_repo import FeatureSetRepository
from stock_ml.db.repositories.template_repo import (
    ModelComponentRepository,
    TargetCatalogRepository,
)

router = APIRouter(prefix="/api/v1/model-library", tags=["model-library"])


# --- Feature Sets (read-only) ---


@router.get("/feature-sets")
async def list_feature_sets(session: AsyncSession = Depends(get_db)):
    """List all feature sets (column count derived from members)."""
    repo = FeatureSetRepository(session)
    feature_sets = await repo.list_all(is_active=True)
    return [
        {
            "id": fs.id,
            "name": fs.name,
            "columnCount": len(fs.members),
            "columns": [m.feature.name for m in fs.members] if fs.members else [],
            "description": fs.description,
        }
        for fs in feature_sets
    ]


@router.get("/feature-sets/{feature_set_id}")
async def get_feature_set(feature_set_id: int, session: AsyncSession = Depends(get_db)):
    """Get feature set by ID (members ordered by position)."""
    repo = FeatureSetRepository(session)
    fs = await repo.get_by_id(feature_set_id)
    if not fs:
        raise HTTPException(status_code=404, detail="Feature set not found")
    return {
        "id": fs.id,
        "name": fs.name,
        "columnCount": len(fs.members),
        "columns": [m.feature.name for m in fs.members],
        "description": fs.description,
    }


# --- Target Catalog ---


@router.get("/targets")
async def list_targets(type: str | None = None, session: AsyncSession = Depends(get_db)):
    """List targets, optionally filtered by type."""
    repo = TargetCatalogRepository(session)
    targets = await repo.list_by_type(type) if type else await repo.list_all(is_active=True)
    return [
        {
            "id": t.id,
            "name": t.name,
            "type": t.type,
            "params": t.params,
            "outputDtype": t.output_dtype,
            "description": t.description,
        }
        for t in targets
    ]


@router.post("/targets")
async def create_target(
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Create a new target definition."""
    required = {"name", "type", "params", "outputDtype"}
    if not required.issubset(body.keys()):
        raise HTTPException(status_code=400, detail=f"Missing required fields: {required}")

    repo = TargetCatalogRepository(session)
    existing = await repo.get_by_name(body["name"])
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Target '{body['name']}' already exists",
        )

    target = await repo.create(
        name=body["name"],
        type=body["type"],
        params=body["params"],
        output_dtype=body["outputDtype"],
        description=body.get("description"),
    )
    await session.commit()
    return {
        "id": target.id,
        "name": target.name,
        "type": target.type,
        "params": target.params,
        "outputDtype": target.output_dtype,
        "description": target.description,
    }


@router.get("/targets/{target_id}")
async def get_target(target_id: int, session: AsyncSession = Depends(get_db)):
    """Get target by ID."""
    repo = TargetCatalogRepository(session)
    target = await repo.get_by_id(target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    return {
        "id": target.id,
        "name": target.name,
        "type": target.type,
        "params": target.params,
        "outputDtype": target.output_dtype,
        "description": target.description,
    }


@router.put("/targets/{target_id}")
async def update_target(
    target_id: int,
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Update target."""
    repo = TargetCatalogRepository(session)
    target = await repo.update(
        target_id,
        params=body.get("params"),
        description=body.get("description"),
    )
    if not target:
        raise HTTPException(status_code=404, detail="Target not found")
    await session.commit()
    return {
        "id": target.id,
        "name": target.name,
        "type": target.type,
        "params": target.params,
        "outputDtype": target.output_dtype,
        "description": target.description,
    }


# --- Model Components ---


@router.get("/components")
async def list_components(
    role: str | None = None,
    algorithm: str | None = None,
    component_type: str | None = None,
    session: AsyncSession = Depends(get_db),
):
    """List model components, optionally filtered by role, algorithm, or component_type.

    Args:
        role: entry | exit | regime | size
        algorithm: lightgbm | xgboost | random_forest | mlp | rule
        component_type: ml | rule (NEW: filter by component classification)
    """
    repo = ModelComponentRepository(session)
    if role and component_type:
        components = await repo.list_by_role_and_component_type(
            role, component_type, is_active=True
        )
    elif role and algorithm:
        components = await repo.list_by_role_and_algorithm(role, algorithm, is_active=True)
    elif role:
        components = await repo.list_by_role(role, is_active=True)
    else:
        # List all
        components = []
        for r in ["entry", "exit", "regime", "size"]:
            components.extend(await repo.list_by_role(r, is_active=True))

    return [
        {
            "id": c.id,
            "name": c.name,
            "role": c.role,
            "algorithm": c.algorithm,
            "componentType": c.component_type,
            "params": c.params,
            "description": c.description,
            "createdAt": c.created_at.isoformat() if c.created_at else None,
        }
        for c in components
    ]


@router.post("/components")
async def create_component(
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Create a new model component."""
    required = {"name", "role", "algorithm"}
    if not required.issubset(body.keys()):
        raise HTTPException(status_code=400, detail=f"Missing required fields: {required}")

    repo = ModelComponentRepository(session)
    existing = await repo.get_by_name(body["name"])
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Component '{body['name']}' already exists",
        )

    component = await repo.create(
        name=body["name"],
        role=body["role"],
        algorithm=body["algorithm"],
        params=body.get("params", {}),
        description=body.get("description"),
    )
    await session.commit()
    return {
        "id": component.id,
        "name": component.name,
        "role": component.role,
        "algorithm": component.algorithm,
        "componentType": component.component_type,
        "params": component.params,
        "description": component.description,
        "createdAt": component.created_at.isoformat() if component.created_at else None,
    }


@router.get("/components/{component_id}")
async def get_component(component_id: int, session: AsyncSession = Depends(get_db)):
    """Get component by ID."""
    repo = ModelComponentRepository(session)
    component = await repo.get_by_id(component_id)
    if not component:
        raise HTTPException(status_code=404, detail="Component not found")
    return {
        "id": component.id,
        "name": component.name,
        "role": component.role,
        "algorithm": component.algorithm,
        "componentType": component.component_type,
        "params": component.params,
        "description": component.description,
        "createdAt": component.created_at.isoformat() if component.created_at else None,
    }


@router.put("/components/{component_id}")
async def update_component(
    component_id: int,
    body: dict,
    session: AsyncSession = Depends(get_db),
):
    """Update component params."""
    repo = ModelComponentRepository(session)
    component = await repo.update(
        component_id,
        params=body.get("params"),
        description=body.get("description"),
    )
    if not component:
        raise HTTPException(status_code=404, detail="Component not found")
    await session.commit()
    return {
        "id": component.id,
        "name": component.name,
        "role": component.role,
        "algorithm": component.algorithm,
        "componentType": component.component_type,
        "params": component.params,
        "description": component.description,
        "createdAt": component.created_at.isoformat() if component.created_at else None,
    }


@router.delete("/components/{component_id}")
async def delete_component(
    component_id: int,
    session: AsyncSession = Depends(get_db),
):
    """Soft delete component."""
    repo = ModelComponentRepository(session)
    success = await repo.soft_delete(component_id)
    if not success:
        raise HTTPException(status_code=404, detail="Component not found")
    await session.commit()
    return {"deleted": True}
