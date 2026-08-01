"""leaderboard_nav ORM — NAV/overlay metrics per run (companion to leaderboard_runs).

Historically script-created (ops/score_nav_leaderboard.py) with NO ORM, read via raw
text() SQL. This model makes the 18-column shape (codified in migration 0030) an
authoritative ORM entity so the API reads it type-safely and a schema drift fails loudly
instead of silently returning '—'.

Column families:
- nav_*/cagr_adv/cagr_noadv/maxdd_nav/nav_f22_adv/years/n_trades_sim/config_hash:
  LEGACY nh_nav2 shuffle yardstick (K=25 permutation-mean). Being retired in favour of
  the overlay column — kept for backward read while coverage grows.
- cagr_t2/maxdd_t2: LEGACY T+2 nh_nav2 variant (ad-hoc, few rows).
- cagr_overlay/maxdd_overlay/overlay_k/overlay_note/overlay_config_hash: the OFFICIAL
  Stage-2 number from stock_ml.portfolio (register_overlay.py). This is canonical.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Double, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from stock_ml.db.base import Base


class LeaderboardNavModel(Base):
    """NAV-sim + Stage-2 overlay metrics for one leaderboard run (PK == run_id)."""

    __tablename__ = "leaderboard_nav"

    run_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("leaderboard_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    # --- legacy nh_nav2 shuffle yardstick (K=25) — being retired ---
    nav_adv: Mapped[float | None] = mapped_column(Double(), nullable=True)
    nav_noadv: Mapped[float | None] = mapped_column(Double(), nullable=True)
    cagr_adv: Mapped[float | None] = mapped_column(Double(), nullable=True)
    cagr_noadv: Mapped[float | None] = mapped_column(Double(), nullable=True)
    maxdd_nav: Mapped[float | None] = mapped_column(Double(), nullable=True)
    nav_f22_adv: Mapped[float | None] = mapped_column(Double(), nullable=True)
    years: Mapped[float | None] = mapped_column(Double(), nullable=True)
    n_trades_sim: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cagr_t2: Mapped[float | None] = mapped_column(Double(), nullable=True)
    maxdd_t2: Mapped[float | None] = mapped_column(Double(), nullable=True)
    # --- official Stage-2 overlay (stock_ml.portfolio) — canonical ---
    cagr_overlay: Mapped[float | None] = mapped_column(Double(), nullable=True)
    maxdd_overlay: Mapped[float | None] = mapped_column(Double(), nullable=True)
    overlay_k: Mapped[int | None] = mapped_column(Integer(), nullable=True)
    overlay_note: Mapped[str | None] = mapped_column(Text(), nullable=True)
    overlay_config_hash: Mapped[str | None] = mapped_column(String(32), nullable=True)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
