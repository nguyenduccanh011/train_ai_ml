"""Identity of a Stage-2 overlay — WHICH strategy, and WHICH scoring of it.

Two hashes that must not be confused (docs/refactor/OVERLAY_IDENTITY_RESTRUCTURE.md §3.1):

``overlay_key(C)``          = md5 over the PortfolioConstants alone.
    WHAT the strategy IS. Stable across every data sync, so it is the primary key of a book and
    the thing a strategy dropdown lists. Panel deliberately EXCLUDED: panel_fingerprint moves
    every time the store syncs, so folding it in here would mint a new "strategy" daily, fill the
    picker with duplicates, and orphan yesterday's book.

``scoring_hash(C, panel_fp)`` = md5 over the constants AND the declared panel.
    WHAT this particular scoring WAS. Idempotency key: already scored under this exact config on
    this exact panel -> skip. A re-declared panel changes it, which is what forces a re-score.

Lives in the wheel (not stock_ml.db) on purpose: serving computes the SAME key for its
tiers.yaml strategies, so the two repos agree on identity without a side agreement.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from stock_ml.portfolio.constants import PortfolioConstants

__all__ = ["overlay_key", "scoring_hash"]


def _digest(C: PortfolioConstants, panel_fp: str | None) -> str:
    """md5 over ALL dataclass fields (so a new knob is captured automatically — no hand-kept
    field list to drift), plus the panel when one is given."""
    d = dataclasses.asdict(C)
    if panel_fp is not None:
        d["__panel_fp__"] = panel_fp
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def overlay_key(C: PortfolioConstants) -> str:
    """32-char identity of the STRATEGY (config only). Primary key of a book."""
    return _digest(C, None)


def scoring_hash(C: PortfolioConstants, panel_fp: str) -> str:
    """32-char identity of one SCORING (config + declared panel). Idempotency key."""
    if not panel_fp:
        raise ValueError(
            "scoring_hash needs the declared panel_fingerprint — an overlay scored against an "
            "unnamed panel cannot be told apart from one scored on a different universe. Use "
            "overlay_key(C) when you want the panel-independent strategy identity."
        )
    return _digest(C, panel_fp)
