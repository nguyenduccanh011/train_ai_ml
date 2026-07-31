"""Config-compatibility guard — ENGINE_UPGRADE §11.5.5 ("bảo đảm ngược THẬT").

The engine must LOAD every live bundle config without error. This catches the class of bug where a
config field exists only on a newer engine (e.g. ``universe_policy`` added in 0.4.2 → the dyn configs
make ``ExperimentConfig(**config)`` raise ``TypeError`` on 0.4.1) while ``format_version`` still claims
the bundle is compatible (§11.2). A config the engine can no longer load is a HARD failure, not a silent
degrade — exactly the guarantee to run on every wheel build.

Fixtures = snapshots of the 6 deployed bundle ``config.json`` under ``tests/fixtures/bundle_configs/``.
Refresh them when a bundle is re-exported (Phase 4). This is Level-1 (config LOADS). The fuller §11.5.5
checks — ``engine_config_from_dict`` 3-way partition (R3 / Phase 3) + a short signal-generation cycle —
are added when R3 lands.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stock_ml.src.pipeline.experiment import ExperimentConfig

_FIXTURES = Path(__file__).parent / "fixtures" / "bundle_configs"
_CONFIGS = sorted(p for p in _FIXTURES.glob("*.json") if not p.name.startswith("_"))


def test_live_config_fixtures_present() -> None:
    """Guard the guard: an empty fixture dir would make the parametrized test vacuously pass."""
    assert len(_CONFIGS) >= 6, f"expected >=6 live bundle configs, found {len(_CONFIGS)}"


@pytest.mark.parametrize("cfg_path", _CONFIGS, ids=lambda p: p.stem)
def test_engine_loads_live_config(cfg_path: Path) -> None:
    """The current engine must construct ``ExperimentConfig`` from every live bundle config (§11.5.5)."""
    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        cfg = ExperimentConfig(**config)
    except Exception as e:  # noqa: BLE001 — any load failure is a compat break we want to surface loudly
        pytest.fail(f"engine cannot load live config {cfg_path.name!r}: {type(e).__name__}: {e}")
    # Minimal sanity: a load that drops the signal-defining essentials is not really "loaded".
    assert getattr(cfg, "feature_set", None), f"{cfg_path.name}: empty feature_set after load"
