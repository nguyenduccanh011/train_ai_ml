"""Build ``resolved.json`` — the self-sufficient export record (ENGINE_UPGRADE §11.5.1 / R1).

A bundle's ``config.json`` records what was CHOSEN (39-65 pinned fields) but not what it takes to REGENERATE
the signal: 162-188 of the 227 ``EngineConfig`` nodes are inherited from wheel defaults, the causal z-window
(``z_norm_window``/``z_norm_min_periods``) comes from hardcoded literals, the feature *semantics* live in the
DSL source, and the wheel version itself is not recorded. ``resolved.json`` captures those so a diff between
two wheels on the same bundle points at exactly the field that moved.

G1 scope (§11.5.1 table): WRITE-ONLY. Export writes this; load checks nothing (that is §11.5.2's job, a
different mechanism comparing the *measuring stick*, not the config). Data is DECLARED, never embedded
(§13.3 Q1): the list of series/symbols + expected version, resolved from the live source at serve time.
"""

from __future__ import annotations

import dataclasses
from importlib.metadata import PackageNotFoundError, version
from typing import Any

# The causal z-window that defines every buy/sell band, read from cfg.engine with these defaults in
# recombine_signals (experiment.py). Recorded resolved so a literal change is visible in the diff.
_ZWIN_DEFAULT = 252
_ZMIN_DEFAULT = 60


def _wheel_version() -> str:
    """Real installed wheel version (NOT core.__version__, which is frozen at the inference-contract
    number). Falls back to the frozen number with a marker if the wheel isn't pip-installed (dev tree)."""
    for name in ("stock_ml_core", "stock-ml-core"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    try:
        from stock_ml.core import __version__ as _v

        return f"{_v}+uninstalled"
    except Exception:
        return "unknown"


def _catalog_fingerprint() -> str:
    """sha over the DSL operator source — the feature SEMANTICS a bundle's column names don't capture."""
    from stock_ml.src.features.dsl.engine import engine_code_fingerprint

    return engine_code_fingerprint()


def build_resolved_config(
    cfg: Any,
    engine: Any,
    *,
    universe: list[str] | None = None,
    universe_slug: str | None = None,
    as_of: str | None = None,
) -> dict:
    """Assemble the ``resolved.json`` payload for a bundle export.

    Args:
        cfg: the resolved ``ExperimentConfig``.
        engine: the ``EngineConfig`` built by ``engine_config_from_dict(cfg.engine)`` — its 227 fields
            after defaults are the record's core (the 162-188 nodes no config file pins).
        universe/universe_slug/as_of: the DATA DECLARATION (§13.3 Q1) — which symbols, which set, as of
            when. NOT the bytes; production re-resolves them from the live source, fail-loud if absent.
    """
    engine_dict = dataclasses.asdict(engine)  # 227 fields incl. nested cost -> dict

    # Recombine layer: the keys this config sets explicitly, PLUS the critical hardcoded-default z-window
    # resolved. (Full resolution of all 35 recombine defaults is a follow-up; the DSL/feature semantics
    # are covered by catalog_fingerprint, the engine nodes by engine_dict.)
    engine_block = cfg.engine if isinstance(cfg.engine, dict) else {}
    from stock_ml.src.backtest.engine import _RECOMBINE_KEYS

    recombine = {k: engine_block[k] for k in _RECOMBINE_KEYS if k in engine_block}
    recombine["z_norm_window"] = int(engine_block.get("z_norm_window", _ZWIN_DEFAULT))
    recombine["z_norm_min_periods"] = int(engine_block.get("z_norm_min_periods", _ZMIN_DEFAULT))

    # Data DECLARATION (not bytes). rs_ metrics need VNINDEX; the run-score modulator + breadth are
    # market-context series the serving host resolves from the live source (fail-loud, §1.1/§11.5.1 Q1).
    data_dependencies = {
        "universe": {
            "slug": universe_slug,
            "as_of": as_of,
            "n_symbols": len(universe) if universe else None,
        },
        "market_context": ["VNINDEX", "runscore", "breadth"],
        "note": "declared only — resolved from the live source (sieutinhieu) at serve time, fail-loud if absent",
    }

    return {
        "schema": "resolved.json/1",
        # This file records the resolve AS OF THIS EXPORT under the wheel below — not a runtime input
        # (§11.4). A backfilled file must overwrite this marker with 'backfill@... <date>' (§R1 note).
        "provenance": f"export@{_wheel_version()}",
        "wheel_version": _wheel_version(),
        "catalog_fingerprint": _catalog_fingerprint(),
        "engine": engine_dict,
        "recombine": recombine,
        "feature_sets": {
            "entry": getattr(cfg, "entry_features", None) or getattr(cfg, "feature_set", None),
            "exit": getattr(cfg, "exit_features", None) or getattr(cfg, "feature_set", None),
        },
        "data_dependencies": data_dependencies,
    }
