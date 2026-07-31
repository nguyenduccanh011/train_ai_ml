"""R3 partition guard for ``engine_config_from_dict`` (ENGINE_UPGRADE §11 R3 / §662).

The one shared deserializer splits a config's ``engine:`` block THREE ways: EngineConfig fields /
recombine-pipeline keys (``_RECOMBINE_KEYS``) / cost keys. Historically three hand-maintained pop-lists
did this and "matched by luck"; these tests assert the partition is actually sound so train and serving
strip identically (§10.3) and a stray key fails LOUD instead of silently reaching either side.
"""

from __future__ import annotations

import dataclasses

import pytest

from stock_ml.src.backtest.engine import (
    _RECOMBINE_KEYS,
    CostModel,
    EngineConfig,
    engine_config_from_dict,
)

_ENGINE_FIELDS = {f.name for f in dataclasses.fields(EngineConfig)}


def test_recombine_keys_disjoint_from_engine_fields() -> None:
    """A recombine key must never also be an EngineConfig field — else stripping it would drop a real
    engine setting (or keeping it would feed the signal layer an engine field). This is the invariant
    the old 'match by luck' pop-lists never checked."""
    overlap = _RECOMBINE_KEYS & _ENGINE_FIELDS
    assert not overlap, f"recombine keys overlap EngineConfig fields: {sorted(overlap)}"
    assert "cost" not in _RECOMBINE_KEYS  # cost is built separately, not stripped


def test_recombine_keys_are_stripped_not_passed_to_engine() -> None:
    """Every recombine key present in the block is consumed here (EngineConfig would reject them)."""
    block = {
        "max_hold_bars": 20,
        "hard_stop_pct": -0.08,
        **{k: 1 for k in _RECOMBINE_KEYS},  # all recombine keys, arbitrary values
        "portfolio": {"enabled": False},
    }
    engine, portfolio_cfg = engine_config_from_dict(block)
    assert isinstance(engine, EngineConfig)
    assert engine.max_hold_bars == 20 and engine.hard_stop_pct == -0.08
    assert portfolio_cfg == {"enabled": False}


def test_cost_keys_flat_and_nested_reach_costmodel() -> None:
    """Cost keys — flat or nested under ``costs:`` — build the CostModel, not EngineConfig fields."""
    flat, _ = engine_config_from_dict(
        {
            "max_hold_bars": 20,
            "hard_stop_pct": None,
            "commission": 0.0025,
            "tax": 0.001,
            "slippage": 0.0015,
        }
    )
    nested, _ = engine_config_from_dict(
        {
            "max_hold_bars": 20,
            "hard_stop_pct": None,
            "costs": {
                "commission": 0.0025,
                "tax": 0.001,
                "slippage": 0.0015,
                "slippage_model": "doc-only",
            },
        }
    )
    assert isinstance(flat.cost, CostModel) and isinstance(nested.cost, CostModel)
    # slippage_model is documentation-only and must not reach CostModel as a kwarg.
    assert flat.cost.commission == nested.cost.commission == 0.0025


def test_unknown_key_fails_loud_on_both_sides() -> None:
    """A key that is neither an EngineConfig field, a recombine key, nor a cost key must RAISE — the
    guarantee that a typo/removed field can't silently no-op in either the train or serving path."""
    with pytest.raises(TypeError):
        engine_config_from_dict(
            {"max_hold_bars": 20, "hard_stop_pct": None, "definitely_not_a_real_engine_key": 1}
        )


def test_input_dict_not_mutated() -> None:
    """The deserializer copies its input — a caller's cfg.engine must survive the pops intact."""
    block = {
        "max_hold_bars": 20,
        "hard_stop_pct": None,
        "entry_gate": 0.4,
        "portfolio": {"enabled": False},
    }
    before = dict(block)
    engine_config_from_dict(block)
    assert block == before, "engine_config_from_dict mutated the caller's engine dict"
