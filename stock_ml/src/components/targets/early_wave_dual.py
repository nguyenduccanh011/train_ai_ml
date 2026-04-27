from __future__ import annotations

import pandas as pd

from src.components.targets.early_wave import EarlyWaveTarget, _early_exit_signal


class EarlyWaveDualTarget:
    """Dual-head: early-wave entry label + exit (target_sell) label.

    Mirrors legacy TargetGenerator(target_type="early_wave_dual").
    generate_entry_labels() → entry target (same as EarlyWaveTarget).
    generate_exit_labels()  → binary sell signal (forward drawdown).
    """

    name = "early_wave_dual"

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 20,
        forward_window: int = 10,
        gain_threshold: float = 0.08,
        loss_threshold: float = 0.05,
        n_classes: int = 3,
    ) -> None:
        self._entry = EarlyWaveTarget(
            short_window=short_window,
            long_window=long_window,
            forward_window=forward_window,
            gain_threshold=gain_threshold,
            loss_threshold=loss_threshold,
            n_classes=n_classes,
        )
        self.loss_threshold = loss_threshold
        self.forward_window = forward_window

    @property
    def num_classes(self) -> int:
        return self._entry.num_classes

    @property
    def supports_exit_labels(self) -> bool:
        return True

    def generate_entry_labels(self, df: pd.DataFrame) -> pd.Series:
        return self._entry.generate_entry_labels(df)

    def generate_exit_labels(
        self,
        df: pd.DataFrame,
        forward_window: int,
        loss_threshold: float,
    ) -> pd.Series | None:
        sell = _early_exit_signal(df["close"].values, forward_window, loss_threshold)
        return pd.Series(sell, index=df.index)
