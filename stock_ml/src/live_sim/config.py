"""Live simulator configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.backtest.engine import EngineConfig
from src.targets.forward import ForwardReturnTarget


@dataclass
class LiveSimConfig:
    """Configuration for live simulator walk-forward."""

    data_root: str
    symbols: list[str]
    out_dir: str
    sim_start: str  # "2025-01-02" (first trading day of test window)
    sim_end: str  # "2025-12-31" (last trading day of test window)

    train_years: int = 4
    gap_days: int = 25
    seed: int = 42

    target: ForwardReturnTarget = field(default_factory=ForwardReturnTarget)
    engine: EngineConfig = field(default_factory=EngineConfig)

    min_volume_filter: float = 0.0

    def __post_init__(self) -> None:
        if self.train_years < 1:
            raise ValueError("train_years must be >= 1")
        if self.gap_days < 0:
            raise ValueError("gap_days must be >= 0")
        if not self.symbols:
            raise ValueError("symbols list cannot be empty")
        if self.min_volume_filter < 0:
            raise ValueError("min_volume_filter must be >= 0")

    @property
    def required_gap_days(self) -> int:
        return max(self.target.horizon * 2 + 5, 7)
