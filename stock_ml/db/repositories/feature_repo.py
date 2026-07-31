"""Repositories for the Feature Store + DSL tables (feature_def / feature_set / …)."""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from stock_ml.db.models.feature import (
    FeatureDefModel,
    FeatureDepModel,
    FeatureSetMemberModel,
    FeatureSetModel,
)


class FeatureDefRepository:
    """CRUD for atomic feature definitions (DSL expressions)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, feature_id: int) -> FeatureDefModel | None:
        result = await self._session.execute(
            select(FeatureDefModel)
            .where(FeatureDefModel.id == feature_id)
            .options(selectinload(FeatureDefModel.deps))
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> FeatureDefModel | None:
        result = await self._session.execute(
            select(FeatureDefModel)
            .where(FeatureDefModel.name == name)
            .options(selectinload(FeatureDefModel.deps))
        )
        return result.scalar_one_or_none()

    async def list(
        self, kind: str | None = None, search: str | None = None, is_active: bool = True
    ) -> list[FeatureDefModel]:
        q = select(FeatureDefModel)
        if is_active:
            q = q.where(FeatureDefModel.is_active.is_(True))
        if kind:
            q = q.where(FeatureDefModel.kind == kind)
        if search:
            q = q.where(FeatureDefModel.name.ilike(f"%{search}%"))
        q = q.order_by(FeatureDefModel.name)
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def usage_counts(self) -> dict[int, int]:
        """feature_id -> number of sets that include it."""
        result = await self._session.execute(
            select(FeatureSetMemberModel.feature_id, func.count()).group_by(
                FeatureSetMemberModel.feature_id
            )
        )
        return {row[0]: row[1] for row in result.all()}

    async def create(
        self,
        *,
        name: str,
        expr: str,
        kind: str,
        expr_hash: str,
        output_dtype: str = "float",
        description: str | None = None,
        version: int = 1,
        feature_deps: list[str] | None = None,
        raw_deps: list[str] | None = None,
    ) -> FeatureDefModel:
        feature = FeatureDefModel(
            name=name,
            expr=expr,
            kind=kind,
            expr_hash=expr_hash,
            output_dtype=output_dtype,
            description=description,
            version=version,
            is_active=True,
        )
        self._session.add(feature)
        await self._session.flush()
        await self._set_deps(feature, feature_deps or [], raw_deps or [])
        return feature

    async def upsert(
        self,
        *,
        name: str,
        expr: str,
        kind: str,
        expr_hash: str,
        output_dtype: str = "float",
        description: str | None = None,
        version: int = 1,
        feature_deps: list[str] | None = None,
        raw_deps: list[str] | None = None,
    ) -> FeatureDefModel:
        """Create or update a feature_def by name (idempotent for seeding)."""
        existing = await self.get_by_name(name)
        if existing is None:
            return await self.create(
                name=name,
                expr=expr,
                kind=kind,
                expr_hash=expr_hash,
                output_dtype=output_dtype,
                description=description,
                version=version,
                feature_deps=feature_deps,
                raw_deps=raw_deps,
            )
        existing.expr = expr
        existing.kind = kind
        existing.expr_hash = expr_hash
        existing.output_dtype = output_dtype
        existing.description = description
        existing.version = version
        existing.is_active = True
        await self._set_deps(existing, feature_deps or [], raw_deps or [])
        await self._session.flush()
        return existing

    async def _set_deps(
        self, feature: FeatureDefModel, feature_deps: list[str], raw_deps: list[str]
    ) -> None:
        await self._session.execute(
            delete(FeatureDepModel).where(FeatureDepModel.feature_id == feature.id)
        )
        for dep_name in feature_deps:
            dep = await self.get_by_name(dep_name)
            self._session.add(
                FeatureDepModel(
                    feature_id=feature.id,
                    depends_on_feature_id=dep.id if dep else None,
                    depends_on_raw=None if dep else dep_name,
                )
            )
        for raw in raw_deps:
            self._session.add(
                FeatureDepModel(
                    feature_id=feature.id, depends_on_feature_id=None, depends_on_raw=raw
                )
            )
        await self._session.flush()


class FeatureSetRepository:
    """CRUD for named feature sets and their membership."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, set_id: int) -> FeatureSetModel | None:
        result = await self._session.execute(
            select(FeatureSetModel)
            .where(FeatureSetModel.id == set_id)
            .options(
                selectinload(FeatureSetModel.members).selectinload(FeatureSetMemberModel.feature)
            )
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> FeatureSetModel | None:
        result = await self._session.execute(
            select(FeatureSetModel)
            .where(FeatureSetModel.name == name)
            .options(
                selectinload(FeatureSetModel.members).selectinload(FeatureSetMemberModel.feature)
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self, is_active: bool = True) -> list[FeatureSetModel]:
        q = select(FeatureSetModel)
        if is_active:
            q = q.where(FeatureSetModel.is_active.is_(True))
        q = q.order_by(FeatureSetModel.name)
        q = q.options(
            selectinload(FeatureSetModel.members).selectinload(FeatureSetMemberModel.feature)
        )
        result = await self._session.execute(q)
        return list(result.scalars().all())

    async def member_counts(self) -> dict[int, int]:
        """feature_set_id -> member count."""
        result = await self._session.execute(
            select(FeatureSetMemberModel.feature_set_id, func.count()).group_by(
                FeatureSetMemberModel.feature_set_id
            )
        )
        return {row[0]: row[1] for row in result.all()}

    async def create(
        self, *, name: str, description: str | None = None, version: int = 1
    ) -> FeatureSetModel:
        fs = FeatureSetModel(name=name, description=description, version=version, is_active=True)
        self._session.add(fs)
        await self._session.flush()
        return fs

    async def upsert(
        self, *, name: str, description: str | None = None, version: int = 1
    ) -> FeatureSetModel:
        existing = await self.get_by_name(name)
        if existing is None:
            return await self.create(name=name, description=description, version=version)
        existing.description = description
        existing.version = version
        existing.is_active = True
        await self._session.flush()
        return existing

    async def replace_members(self, set_id: int, feature_ids: list[int]) -> None:
        """Set the exact ordered membership of a set (idempotent)."""
        await self._session.execute(
            delete(FeatureSetMemberModel).where(FeatureSetMemberModel.feature_set_id == set_id)
        )
        for pos, fid in enumerate(feature_ids):
            self._session.add(
                FeatureSetMemberModel(feature_set_id=set_id, feature_id=fid, position=pos)
            )
        await self._session.flush()
