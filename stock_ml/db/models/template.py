"""ORM models for model library and strategy templates (Phase 0-3: DB-First Experiments).

Tables:
  - target_catalog: reusable target definitions
  - model_components: model library (entry/exit/regime/size slots)
  - component_slots: join table (template → (ml_component, rule_component) per slot)
  - strategy_templates: experiment configs (replaces YAML)
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base

if TYPE_CHECKING:
    from stock_ml.db.models.feature import FeatureSetModel


class TargetCatalogModel(Base):
    """Reusable target definitions."""

    __tablename__ = "target_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    output_dtype: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    templates: Mapped[list[StrategyTemplateModel]] = relationship(
        "StrategyTemplateModel",
        back_populates="target",
        foreign_keys="StrategyTemplateModel.target_id",
    )


class ModelComponentModel(Base):
    """Model library: entry/exit/regime/size components with params."""

    __tablename__ = "model_components"

    __table_args__ = (Index("idx_model_components_role_algo", "role", "algorithm"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    component_type: Mapped[str] = mapped_column(String(8), nullable=False, default="ml", index=True)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    slots_as_ml: Mapped[list[ComponentSlotModel]] = relationship(
        "ComponentSlotModel",
        back_populates="ml_component",
        foreign_keys="ComponentSlotModel.ml_component_id",
    )
    slots_as_rule: Mapped[list[ComponentSlotModel]] = relationship(
        "ComponentSlotModel",
        back_populates="rule_component",
        foreign_keys="ComponentSlotModel.rule_component_id",
    )


class ComponentSlotModel(Base):
    """Component slot: join table for strategy → (ml_component, rule_component) per slot.

    Phase 0.4: Per-slot feature sets and targets.
    - feature_set_name: None → use strategy's global feature_set
    - target_config: None → use strategy's global target
    """

    __tablename__ = "component_slots"

    __table_args__ = (
        Index("idx_component_slots_template", "template_id"),
        Index("idx_component_slots_slot_type", "slot_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    template_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("strategy_templates.id", ondelete="CASCADE"), nullable=False
    )
    slot_type: Mapped[str] = mapped_column(String(8), nullable=False)
    ml_component_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_components.id", ondelete="SET NULL"), nullable=True
    )
    rule_component_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("model_components.id", ondelete="SET NULL"), nullable=True
    )
    feature_set_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    template: Mapped[StrategyTemplateModel] = relationship(
        "StrategyTemplateModel", back_populates="component_slots"
    )
    ml_component: Mapped[ModelComponentModel | None] = relationship(
        "ModelComponentModel",
        back_populates="slots_as_ml",
        foreign_keys=[ml_component_id],
    )
    rule_component: Mapped[ModelComponentModel | None] = relationship(
        "ModelComponentModel",
        back_populates="slots_as_rule",
        foreign_keys=[rule_component_id],
    )


class StrategyTemplateModel(Base):
    """Strategy template: experiment config (replaces YAML file)."""

    __tablename__ = "strategy_templates"

    __table_args__ = (
        Index("idx_template_market", "market"),
        Index("idx_template_feature_set", "feature_set_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Strategy identity
    market: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(255), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False, default="long")
    universe_slug: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Components (FK)
    feature_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_set.id", ondelete="RESTRICT"), nullable=False
    )
    target_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("target_catalog.id", ondelete="RESTRICT"), nullable=False
    )

    # Signal config
    signal_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="entry_first")
    signal_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Hysteresis band (Phase 1): score > entry_threshold → long, score < exit_threshold → exit.
    # NULL means "derive from signal_threshold" (+thr / -thr), reproducing the legacy rule.
    entry_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="ml_only")

    # Execution configs (JSON)
    split_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    engine_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    validation_config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    seed: Mapped[int] = mapped_column(Integer, nullable=False, default=42)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    feature_set: Mapped[FeatureSetModel] = relationship(
        "FeatureSetModel", back_populates="templates", foreign_keys=[feature_set_id]
    )
    target: Mapped[TargetCatalogModel] = relationship(
        "TargetCatalogModel", back_populates="templates", foreign_keys=[target_id]
    )
    component_slots: Mapped[list[ComponentSlotModel]] = relationship(
        "ComponentSlotModel", back_populates="template", cascade="all, delete-orphan"
    )

    # Config versioning
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    def infer_model_mode(self) -> str:
        """Auto-infer model_mode from entry slot components.

        Returns:
            "ml_only": entry has ML component only
            "rule_only": entry has Rule component only
            "ml_rule_hybrid": entry has both ML and Rule components
            "none": entry slot is missing or has no components
        """
        entry_slot = next((s for s in self.component_slots if s.slot_type == "entry"), None)

        if not entry_slot:
            return "none"

        has_ml = entry_slot.ml_component is not None
        has_rule = entry_slot.rule_component is not None

        if has_ml and has_rule:
            return "ml_rule_hybrid"
        elif has_ml:
            return "ml_only"
        elif has_rule:
            return "rule_only"
        else:
            return "none"
