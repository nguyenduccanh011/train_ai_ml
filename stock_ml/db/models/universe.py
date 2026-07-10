"""Universe (symbol set) management models."""

from __future__ import annotations

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stock_ml.db.base import Base, TimestampMixin


class UniverseSetModel(Base, TimestampMixin):
    """A named, versioned set of symbols for backtesting/training.

    Supports: explicit lists, sector/group definitions, dynamic sets.
    """

    __tablename__ = "universe_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    market: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    symbol_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    symbols: Mapped[list[UniverseSymbolModel]] = relationship(
        "UniverseSymbolModel",
        back_populates="universe",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    __table_args__ = (
        Index("idx_universe_market_active", "market", "is_active"),
        Index("idx_universe_market_locked", "market", "is_locked"),
    )


class UniverseSymbolModel(Base):
    """A symbol within a universe."""

    __tablename__ = "universe_symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    universe_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("universe_sets.id", ondelete="CASCADE"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol_group: Mapped[str | None] = mapped_column(String(64), nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    universe: Mapped[UniverseSetModel] = relationship(
        "UniverseSetModel", back_populates="symbols", lazy="noload"
    )

    __table_args__ = (
        UniqueConstraint("universe_id", "symbol", name="uq_universe_symbol"),
        Index("idx_universe_id", "universe_id"),
        Index("idx_symbol", "symbol"),
    )
