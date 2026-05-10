from __future__ import annotations

from src.components.targets.early_wave import EarlyWaveTarget
from src.components.targets.early_wave_dual import EarlyWaveDualTarget
from src.components.targets.early_wave_v2 import EarlyWaveV2Target
from src.components.targets.trend_regime import TrendRegimeTarget

_REGISTRY: dict[str, type] = {
    "trend_regime": TrendRegimeTarget,
    "early_wave": EarlyWaveTarget,
    "early_wave_v2": EarlyWaveV2Target,
    "early_wave_dual": EarlyWaveDualTarget,
}


def get_target(name: str, **kwargs: object) -> object:
    """Instantiate a target generator by name."""
    if name not in _REGISTRY:
        available = list(_REGISTRY)
        raise KeyError(f"Unknown target {name!r}. Available: {available}")
    return _REGISTRY[name](**kwargs)


def list_targets() -> list[str]:
    return list(_REGISTRY)
