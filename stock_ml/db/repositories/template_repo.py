"""Repository for strategy templates and model components."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from stock_ml.db.models.template import (
    ComponentSlotModel,
    ModelComponentModel,
    StrategyTemplateModel,
    TargetCatalogModel,
)


class ModelComponentRepository:
    """CRUD for model components."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str,
        role: str,
        algorithm: str,
        params: dict,
        description: str | None = None,
    ) -> ModelComponentModel:
        """Create a new model component."""
        component = ModelComponentModel(
            name=name,
            role=role,
            algorithm=algorithm,
            component_type="rule" if algorithm == "rule" else "ml",
            params=params,
            description=description,
            is_active=True,
        )
        self._session.add(component)
        await self._session.flush()
        return component

    async def get_by_id(self, component_id: int) -> ModelComponentModel | None:
        """Get component by ID."""
        result = await self._session.execute(
            select(ModelComponentModel).where(ModelComponentModel.id == component_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> ModelComponentModel | None:
        """Get component by name."""
        result = await self._session.execute(
            select(ModelComponentModel).where(ModelComponentModel.name == name)
        )
        return result.scalar_one_or_none()

    async def list_by_role(self, role: str, is_active: bool = True) -> list[ModelComponentModel]:
        """List components by role."""
        q = select(ModelComponentModel).where(ModelComponentModel.role == role)
        if is_active:
            q = q.where(ModelComponentModel.is_active.is_(True))
        q = q.order_by(ModelComponentModel.name)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def list_by_role_and_algorithm(
        self, role: str, algorithm: str, is_active: bool = True
    ) -> list[ModelComponentModel]:
        """List components by role and algorithm."""
        q = select(ModelComponentModel).where(
            and_(
                ModelComponentModel.role == role,
                ModelComponentModel.algorithm == algorithm,
            )
        )
        if is_active:
            q = q.where(ModelComponentModel.is_active.is_(True))
        q = q.order_by(ModelComponentModel.name)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def list_by_role_and_component_type(
        self, role: str, component_type: str, is_active: bool = True
    ) -> list[ModelComponentModel]:
        """List components by role and component_type (ml | rule)."""
        q = select(ModelComponentModel).where(
            and_(
                ModelComponentModel.role == role,
                ModelComponentModel.component_type == component_type,
            )
        )
        if is_active:
            q = q.where(ModelComponentModel.is_active.is_(True))
        q = q.order_by(ModelComponentModel.name)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def update(
        self,
        component_id: int,
        params: dict | None = None,
        description: str | None = None,
    ) -> ModelComponentModel | None:
        """Update component."""
        updates = {}
        if params is not None:
            updates["params"] = params
        if description is not None:
            updates["description"] = description
        if not updates:
            return await self.get_by_id(component_id)

        await self._session.execute(
            update(ModelComponentModel)
            .where(ModelComponentModel.id == component_id)
            .values(**updates, updated_at=datetime.utcnow())
        )
        await self._session.flush()
        return await self.get_by_id(component_id)

    async def soft_delete(self, component_id: int) -> bool:
        """Soft delete (set is_active=False)."""
        result = await self._session.execute(
            update(ModelComponentModel)
            .where(ModelComponentModel.id == component_id)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0  # type: ignore[return-value]


class TargetCatalogRepository:
    """CRUD for target definitions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        name: str,
        type: str,
        params: dict,
        output_dtype: str,
        description: str | None = None,
    ) -> TargetCatalogModel:
        """Create a new target definition."""
        target = TargetCatalogModel(
            name=name,
            type=type,
            params=params,
            output_dtype=output_dtype,
            description=description,
            is_active=True,
        )
        self._session.add(target)
        await self._session.flush()
        return target

    async def get_by_id(self, target_id: int) -> TargetCatalogModel | None:
        """Get target by ID."""
        result = await self._session.execute(
            select(TargetCatalogModel).where(TargetCatalogModel.id == target_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> TargetCatalogModel | None:
        """Get target by name."""
        result = await self._session.execute(
            select(TargetCatalogModel).where(TargetCatalogModel.name == name)
        )
        return result.scalar_one_or_none()

    async def list_all(self, is_active: bool = True) -> list[TargetCatalogModel]:
        """List all targets."""
        q = select(TargetCatalogModel)
        if is_active:
            q = q.where(TargetCatalogModel.is_active.is_(True))
        q = q.order_by(TargetCatalogModel.name)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def list_by_type(self, type: str) -> list[TargetCatalogModel]:
        """List targets by type."""
        result = await self._session.execute(
            select(TargetCatalogModel)
            .where(
                and_(
                    TargetCatalogModel.type == type,
                    TargetCatalogModel.is_active.is_(True),
                )
            )
            .order_by(TargetCatalogModel.name)
        )
        return list(result.scalars().all())

    async def update(
        self,
        target_id: int,
        params: dict | None = None,
        description: str | None = None,
    ) -> TargetCatalogModel | None:
        """Update target."""
        updates = {}
        if params is not None:
            updates["params"] = params
        if description is not None:
            updates["description"] = description
        if not updates:
            return await self.get_by_id(target_id)

        await self._session.execute(
            update(TargetCatalogModel)
            .where(TargetCatalogModel.id == target_id)
            .values(**updates, updated_at=datetime.utcnow())
        )
        await self._session.flush()
        return await self.get_by_id(target_id)

    async def soft_delete(self, target_id: int) -> bool:
        """Soft delete."""
        result = await self._session.execute(
            update(TargetCatalogModel)
            .where(TargetCatalogModel.id == target_id)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0  # type: ignore[return-value]


class StrategyTemplateRepository:
    """CRUD for strategy templates."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _get_slot(template: StrategyTemplateModel, slot_type: str) -> ComponentSlotModel | None:
        """Get component slot by type (entry|exit|regime|size)."""
        return next((s for s in template.component_slots if s.slot_type == slot_type), None)

    @staticmethod
    def _compute_config_hash(
        feature_set_id: int,
        target_id: int,
        split_config: dict,
        engine_config: dict,
        seed: int,
        signal_threshold: float,
        model_mode: str,
        direction: str = "long",
        signal_mode: str = "entry_first",
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
    ) -> str:
        """Compute config hash from strategy parameters.

        entry/exit_threshold are only folded into the hash when explicitly set, so
        templates that leave them NULL (derive from signal_threshold) keep a stable hash.
        """
        payload = {
            "feature_set_id": feature_set_id,
            "target_id": target_id,
            "split_config": split_config,
            "engine_config": engine_config,
            "seed": seed,
            "signal_threshold": signal_threshold,
            "model_mode": model_mode,
            "direction": direction,
            "signal_mode": signal_mode,
        }
        if entry_threshold is not None:
            payload["entry_threshold"] = entry_threshold
        if exit_threshold is not None:
            payload["exit_threshold"] = exit_threshold
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def _infer_model_mode(self, entry_slot: dict) -> str:
        """Infer model_mode from entry slot component IDs.

        Returns:
            "ml_only": entry has ML component only
            "rule_only": entry has Rule component only
            "ml_rule_hybrid": entry has both ML and Rule components
        """
        has_ml = entry_slot.get("ml_component_id") is not None
        has_rule = entry_slot.get("rule_component_id") is not None

        if has_ml and has_rule:
            return "ml_rule_hybrid"
        elif has_ml:
            return "ml_only"
        elif has_rule:
            return "rule_only"
        else:
            return "ml_only"

    async def create(
        self,
        name: str,
        market: str,
        strategy: str,
        feature_set_id: int,
        target_id: int,
        component_slots: list[dict],
        direction: str = "long",
        signal_mode: str = "entry_first",
        signal_threshold: float = 0.0,
        entry_threshold: float | None = None,
        exit_threshold: float | None = None,
        model_mode: str | None = None,
        split_config: dict | None = None,
        engine_config: dict | None = None,
        validation_config: dict | None = None,
        seed: int = 42,
        description: str | None = None,
        hypothesis: str | None = None,
        universe_slug: str | None = None,
    ) -> StrategyTemplateModel:
        """Create a new strategy template with component slots.

        Args:
            component_slots: list of dicts with keys:
                {slot_type, ml_component_id, rule_component_id}
                where slot_type in [entry, exit, regime, size]
            model_mode: If None, auto-infer from entry slot components.
                        If provided, use explicitly (for backward compatibility).
        """
        # Validate: must have entry slot with at least one component
        entry_slots = [s for s in component_slots if s.get("slot_type") == "entry"]
        if not entry_slots:
            raise ValueError("Must provide at least one entry slot")
        entry_slot = entry_slots[0]
        if not entry_slot.get("ml_component_id") and not entry_slot.get("rule_component_id"):
            raise ValueError("Entry slot must have ml_component_id or rule_component_id")

        # Auto-infer model_mode if not provided
        if model_mode is None:
            model_mode = self._infer_model_mode(entry_slot)

        config_hash = self._compute_config_hash(
            feature_set_id=feature_set_id,
            target_id=target_id,
            split_config=split_config or {},
            engine_config=engine_config or {},
            seed=seed,
            signal_threshold=signal_threshold,
            model_mode=model_mode,
            direction=direction,
            signal_mode=signal_mode,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )

        template = StrategyTemplateModel(
            name=name,
            market=market,
            strategy=strategy,
            feature_set_id=feature_set_id,
            target_id=target_id,
            direction=direction,
            signal_mode=signal_mode,
            signal_threshold=signal_threshold,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            model_mode=model_mode,
            split_config=split_config or {},
            engine_config=engine_config or {},
            validation_config=validation_config or {},
            seed=seed,
            description=description,
            hypothesis=hypothesis,
            universe_slug=universe_slug,
            is_active=True,
            config_hash=config_hash,
            version=1,
        )

        # Create component slots (Phase 0.4: per-slot features/targets)
        for slot_data in component_slots:
            slot = ComponentSlotModel(
                slot_type=slot_data.get("slot_type"),
                ml_component_id=slot_data.get("ml_component_id"),
                rule_component_id=slot_data.get("rule_component_id"),
                feature_set_name=slot_data.get("feature_set_name"),  # NEW
                target_config=slot_data.get("target_config"),  # NEW
            )
            template.component_slots.append(slot)

        self._session.add(template)
        await self._session.flush()
        return template

    async def get_by_id(self, template_id: int) -> StrategyTemplateModel | None:
        """Get template by ID with all relationships."""
        result = await self._session.execute(
            select(StrategyTemplateModel)
            .where(StrategyTemplateModel.id == template_id)
            .options(
                selectinload(StrategyTemplateModel.feature_set),
                selectinload(StrategyTemplateModel.target),
                selectinload(StrategyTemplateModel.component_slots).selectinload(
                    ComponentSlotModel.ml_component
                ),
                selectinload(StrategyTemplateModel.component_slots).selectinload(
                    ComponentSlotModel.rule_component
                ),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> StrategyTemplateModel | None:
        """Get template by name."""
        result = await self._session.execute(
            select(StrategyTemplateModel)
            .where(StrategyTemplateModel.name == name)
            .options(
                selectinload(StrategyTemplateModel.feature_set),
                selectinload(StrategyTemplateModel.target),
                selectinload(StrategyTemplateModel.component_slots).selectinload(
                    ComponentSlotModel.ml_component
                ),
                selectinload(StrategyTemplateModel.component_slots).selectinload(
                    ComponentSlotModel.rule_component
                ),
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self, is_active: bool = True) -> list[StrategyTemplateModel]:
        """List all templates with relationships loaded."""
        q = select(StrategyTemplateModel)
        if is_active:
            q = q.where(StrategyTemplateModel.is_active.is_(True))
        q = q.order_by(StrategyTemplateModel.name)
        q = q.options(
            selectinload(StrategyTemplateModel.feature_set),
            selectinload(StrategyTemplateModel.target),
            selectinload(StrategyTemplateModel.component_slots).selectinload(
                ComponentSlotModel.ml_component
            ),
            selectinload(StrategyTemplateModel.component_slots).selectinload(
                ComponentSlotModel.rule_component
            ),
        )
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def list_by_market(self, market: str) -> list[StrategyTemplateModel]:
        """List templates by market."""
        result = await self._session.execute(
            select(StrategyTemplateModel)
            .where(
                and_(
                    StrategyTemplateModel.market == market,
                    StrategyTemplateModel.is_active.is_(True),
                )
            )
            .order_by(StrategyTemplateModel.name)
        )
        return list(result.scalars().all())

    async def list_by_feature_set(self, feature_set_id: int) -> list[StrategyTemplateModel]:
        """List templates using a feature set."""
        result = await self._session.execute(
            select(StrategyTemplateModel)
            .where(
                and_(
                    StrategyTemplateModel.feature_set_id == feature_set_id,
                    StrategyTemplateModel.is_active.is_(True),
                )
            )
            .order_by(StrategyTemplateModel.name)
        )
        return list(result.scalars().all())

    async def update(
        self,
        template_id: int,
        **kwargs,
    ) -> StrategyTemplateModel | None:
        """Update template fields. Recomputes config_hash and increments version if config changes.

        If model_mode not provided and config changes, auto-infers from current entry slot.
        """
        updates = {k: v for k, v in kwargs.items() if v is not None}
        if not updates:
            return await self.get_by_id(template_id)

        # Check if config-related fields are being updated
        config_fields = {
            "split_config",
            "engine_config",
            "seed",
            "signal_threshold",
            "model_mode",
            "direction",
            "signal_mode",
            "entry_threshold",
            "exit_threshold",
        }
        if config_fields & set(updates.keys()):
            template = await self.get_by_id(template_id)
            if template:
                # Auto-infer model_mode if config changes but model_mode not explicitly provided
                model_mode = updates.get("model_mode")
                if model_mode is None:
                    model_mode = template.infer_model_mode()

                config_hash = self._compute_config_hash(
                    feature_set_id=template.feature_set_id,
                    target_id=template.target_id,
                    split_config=updates.get("split_config") or template.split_config,
                    engine_config=updates.get("engine_config") or template.engine_config,
                    seed=updates.get("seed") or template.seed,
                    signal_threshold=updates.get("signal_threshold", template.signal_threshold),
                    model_mode=model_mode,
                    direction=updates.get("direction", template.direction),
                    signal_mode=updates.get("signal_mode", template.signal_mode),
                    entry_threshold=updates.get("entry_threshold", template.entry_threshold),
                    exit_threshold=updates.get("exit_threshold", template.exit_threshold),
                )
                updates["config_hash"] = config_hash
                updates["model_mode"] = model_mode
                updates["version"] = template.version + 1

        await self._session.execute(
            update(StrategyTemplateModel)
            .where(StrategyTemplateModel.id == template_id)
            .values(**updates, updated_at=datetime.utcnow())
        )
        await self._session.flush()
        return await self.get_by_id(template_id)

    async def soft_delete(self, template_id: int) -> bool:
        """Soft delete."""
        result = await self._session.execute(
            update(StrategyTemplateModel)
            .where(StrategyTemplateModel.id == template_id)
            .values(is_active=False, updated_at=datetime.utcnow())
        )
        return result.rowcount > 0  # type: ignore[return-value]
