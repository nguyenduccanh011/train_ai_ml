"""ORM models for the Feature Store + Expression DSL (replaces feature_set_catalog).

Five normalised tables (see docs/FEATURE_STORE_DSL_DESIGN.md §3):

  - feature_def           : one atomic feature = one DSL formula (expr, kind, hash)
  - feature_dep           : DAG edges extracted from the parser (topo-sort/impact)
  - feature_set           : a named collection of features
  - feature_set_member    : M:N feature_set ↔ feature_def (DEDUP of *definitions*)
  - feature_materialization : index of the on-disk feature store (DEDUP of *values*)

Column count is derived (COUNT(member)) — there is no hardcoded ``column_count``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base

if TYPE_CHECKING:
    from stock_ml.db.models.template import StrategyTemplateModel


class FeatureDefModel(Base):
    """An atomic feature: a single DSL expression with inferred kind + content hash."""

    __tablename__ = "feature_def"

    __table_args__ = (
        Index("idx_feature_def_kind", "kind"),
        Index("idx_feature_def_expr_hash", "expr_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    expr: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # per_symbol|cross_sectional|market
    output_dtype: Mapped[str] = mapped_column(String(16), nullable=False, default="float")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    expr_hash: Mapped[str] = mapped_column(String(40), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    deps: Mapped[list[FeatureDepModel]] = relationship(
        "FeatureDepModel",
        back_populates="feature",
        foreign_keys="FeatureDepModel.feature_id",
        cascade="all, delete-orphan",
    )
    members: Mapped[list[FeatureSetMemberModel]] = relationship(
        "FeatureSetMemberModel", back_populates="feature", cascade="all, delete-orphan"
    )
    materializations: Mapped[list[FeatureMaterializationModel]] = relationship(
        "FeatureMaterializationModel", back_populates="feature", cascade="all, delete-orphan"
    )


class FeatureDepModel(Base):
    """A dependency edge: feature → (another feature | a raw field)."""

    __tablename__ = "feature_dep"

    __table_args__ = (Index("idx_feature_dep_feature", "feature_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_def.id", ondelete="CASCADE"), nullable=False
    )
    depends_on_feature_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("feature_def.id", ondelete="CASCADE"), nullable=True
    )
    depends_on_raw: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 'close' etc.

    feature: Mapped[FeatureDefModel] = relationship(
        "FeatureDefModel", back_populates="deps", foreign_keys=[feature_id]
    )


class FeatureSetModel(Base):
    """A named feature set (replaces feature_set_catalog)."""

    __tablename__ = "feature_set"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    members: Mapped[list[FeatureSetMemberModel]] = relationship(
        "FeatureSetMemberModel",
        back_populates="feature_set",
        cascade="all, delete-orphan",
        order_by="FeatureSetMemberModel.position",
    )
    templates: Mapped[list[StrategyTemplateModel]] = relationship(
        "StrategyTemplateModel",
        back_populates="feature_set",
        foreign_keys="StrategyTemplateModel.feature_set_id",
    )


class FeatureSetMemberModel(Base):
    """M:N join: which features belong to a set, in order."""

    __tablename__ = "feature_set_member"

    __table_args__ = (
        UniqueConstraint("feature_set_id", "feature_id", name="uq_feature_set_member"),
        Index("idx_feature_set_member_set", "feature_set_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_set_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_set.id", ondelete="CASCADE"), nullable=False
    )
    feature_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_def.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    feature_set: Mapped[FeatureSetModel] = relationship(
        "FeatureSetModel", back_populates="members"
    )
    feature: Mapped[FeatureDefModel] = relationship("FeatureDefModel", back_populates="members")


class FeatureMaterializationModel(Base):
    """Index of materialised feature values in the on-disk feature store."""

    __tablename__ = "feature_materialization"

    __table_args__ = (
        UniqueConstraint(
            "feature_id", "expr_hash", "data_version", name="uq_feature_materialization"
        ),
        Index("idx_feature_mat_feature", "feature_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feature_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("feature_def.id", ondelete="CASCADE"), nullable=False
    )
    expr_hash: Mapped[str] = mapped_column(String(40), nullable=False)
    data_version: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_uri: Mapped[str] = mapped_column(Text, nullable=False)
    rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    feature: Mapped[FeatureDefModel] = relationship(
        "FeatureDefModel", back_populates="materializations"
    )
