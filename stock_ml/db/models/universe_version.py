"""Universe version snapshots — preserves symbol history when universe updates."""

from datetime import UTC, datetime

from sqlalchemy import ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from stock_ml.db.base import Base


class UniverseVersionModel(Base):
    """Snapshot of universe symbols at each version change.

    When universe.symbols change, version increments and we record
    the complete symbol list. Enables auditing and reconstruction of
    historical universes.
    """

    __tablename__ = "universe_versions"

    __table_args__ = (UniqueConstraint("universe_id", "version", name="uq_universe_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    universe_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("universe_sets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    symbols_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="[]",
    )
    symbol_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        name="created_at", nullable=False, default=lambda: datetime.now(UTC)
    )
