"""Catalog of the historical hb_deploy_* research variants as PortfolioConstants
overrides — replaces the 10 per-lever script copies deleted in step 4 of
docs/refactor/PORTFOLIO_LAYER_UNIFICATION.md.

ONLY 'gtos' (the champion default) is golden-verified byte-exact
(test_portfolio_golden.py). The other mappings were extracted from the script
diffs vs hb_deploy_gtos.py; before relying on one for a registered number,
verify against the original script at git 9e37efd7
(stock_ml/analysis/serving_blindspot/r3line/hb_deploy_<name>.py).

Levers with no constants mapping (dl63 family) were NOT ported: they add a
~150-line dist-low-63 entry-gate (+ time-stop for dl63ts) — recover from git
history if that line is ever revived.
"""

from __future__ import annotations

from dataclasses import replace

from stock_ml.portfolio.constants import PortfolioConstants

# name -> (overrides dict | None if not portable, note)
VARIANTS: dict[str, tuple[dict | None, str]] = {
    "gtos": (
        {},
        "champion default: early-cut s2 + green-trail 8% + ret7 + overshoot ON (golden-verified)",
    ),
    "gt": (
        dict(os_pct=100.0),
        "gtos minus overshoot (os_pct=100 -> threshold=max -> filter no-op)",
    ),
    "gt1": (
        dict(os_pct=100.0, ec_check_bar=1),
        "early-cut checks red@s1, cuts s2; no overshoot. UNVERIFIED: green-trail start bar may also differ",
    ),
    "ec": (
        dict(os_pct=100.0, gt=1.0),
        "early-cut only (gt=1.0 disables green-trail); no overshoot",
    ),
    "osdef": (dict(gt=1.0), "early-cut only, overshoot ON"),
    "ret7g": (
        dict(os_pct=100.0, rewrite_on=False),
        "ret7 gate only, no exit rewrite, no overshoot",
    ),
    "ret5g": (
        dict(os_pct=100.0, rewrite_on=False, ret_win=5, r5thr=0.03),
        "ret5 gate (window 5, thr 0.03), no exit rewrite, no overshoot",
    ),
    "dl63": (None, "dist-low-63 entry-gate family — NOT ported, see git 9e37efd7"),
    "dl63ts": (None, "dl63 + time-stop — NOT ported, see git 9e37efd7"),
    "dl63opt": (None, "dl63 optimized — NOT ported, see git 9e37efd7"),
}


def variant_constants(name: str, base: PortfolioConstants | None = None) -> PortfolioConstants:
    overrides, note = VARIANTS[name]
    if overrides is None:
        raise ValueError(f"variant '{name}' has no constants mapping: {note}")
    return replace(base or PortfolioConstants(), **overrides)
