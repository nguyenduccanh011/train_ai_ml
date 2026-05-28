"""Signal generation and freezing — T-1 prediction locked before T execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.basic import FEATURE_COLS
from src.live_sim.config import LiveSimConfig
from src.models.baseline import BaselineModel


@dataclass(frozen=True)
class FrozenSignalSet:
    """Immutable signal set generated at T-1, frozen before T execution."""

    generated_at: pd.Timestamp  # T-1 date when signals were generated
    for_execution_date: pd.Timestamp  # T date when signals will be executed
    signals: dict[str, int]  # symbol → {-1, 0, 1}
    n_buy: int
    n_sell: int
    n_neutral: int
    filters_applied: list[str]  # ["min_volume", ...]
    integrity_hash: str  # sha1 of signals to prevent tampering

    def buys(self) -> frozenset[str]:
        return frozenset(s for s, sig in self.signals.items() if sig == 1)

    def sells(self) -> frozenset[str]:
        return frozenset(s for s, sig in self.signals.items() if sig == -1)

    def neutrals(self) -> frozenset[str]:
        return frozenset(s for s, sig in self.signals.items() if sig == 0)


class SignalGenerator:
    """Generate FrozenSignalSet from model predictions on T-1 history."""

    def __init__(self, model: BaselineModel, config: LiveSimConfig):
        self.model = model
        self.config = config

    def generate(
        self,
        yesterday: pd.Timestamp,
        today: pd.Timestamp,
        history_feat: pd.DataFrame,
    ) -> FrozenSignalSet:
        """Generate and freeze signals for today's execution.

        Args:
            yesterday: T-1 date (signal generation date)
            today: T date (execution date)
            history_feat: features with columns [symbol, date, *FEATURE_COLS]
                         must have max(date) <= yesterday, no lookahead

        Returns:
            FrozenSignalSet — immutable signal dict locked at T-1

        Raises:
            ValueError: if history_feat contains future data (date > yesterday)
            ValueError: if features have NaN values
        """
        if history_feat.empty:
            raise ValueError(f"history_feat is empty at {yesterday}")

        max_date = pd.to_datetime(history_feat["date"]).max()
        if max_date > yesterday:
            raise ValueError(
                f"lookahead detected: history_feat has date {max_date} > yesterday {yesterday}"
            )

        raw_signals = {}
        for sym in self.config.symbols:
            sym_feat = history_feat[history_feat["symbol"] == sym]
            if sym_feat.empty:
                raw_signals[sym] = 0
                continue

            last_row = sym_feat.iloc[-1]
            last_row_date = pd.Timestamp(last_row["date"]).normalize().date()
            if last_row_date != yesterday.normalize().date():
                raise ValueError(
                    f"symbol {sym} missing data at {yesterday}: last row is {last_row['date']}"
                )

            feat_cols_present = [c for c in FEATURE_COLS if c in sym_feat.columns]
            if not feat_cols_present:
                raise ValueError(f"symbol {sym} has no feature columns")

            if sym_feat[feat_cols_present].iloc[-1].isna().any():
                raise ValueError(
                    f"symbol {sym} has NaN features at {yesterday}: {feat_cols_present}"
                )

            X = sym_feat[feat_cols_present].iloc[-1:].to_numpy(dtype=np.float32)
            sig = int(self.model.predict(X)[0])
            raw_signals[sym] = sig

        filtered = self._apply_filters(raw_signals, history_feat, yesterday)

        n_buy = sum(1 for s in filtered.values() if s == 1)
        n_sell = sum(1 for s in filtered.values() if s == -1)
        n_neutral = sum(1 for s in filtered.values() if s == 0)

        integrity_hash = self._compute_hash(yesterday, filtered)

        return FrozenSignalSet(
            generated_at=yesterday,
            for_execution_date=today,
            signals=filtered,
            n_buy=n_buy,
            n_sell=n_sell,
            n_neutral=n_neutral,
            filters_applied=["min_volume"] if self.config.min_volume_filter > 0 else [],
            integrity_hash=integrity_hash,
        )

    def _apply_filters(
        self,
        raw_signals: dict[str, int],
        history_feat: pd.DataFrame,
        yesterday: pd.Timestamp,
    ) -> dict[str, int]:
        """Apply all filters at T-1 time (no lookahead allowed)."""
        filtered = dict(raw_signals)

        if self.config.min_volume_filter > 0:
            for sym in filtered:
                sym_feat = history_feat[history_feat["symbol"] == sym]
                if not sym_feat.empty and "volume" in sym_feat.columns:
                    avg_vol_20 = sym_feat["volume"].tail(20).mean()
                    if avg_vol_20 < self.config.min_volume_filter:
                        filtered[sym] = 0  # neutralize

        return filtered

    @staticmethod
    def _compute_hash(date: pd.Timestamp, signals: dict[str, int]) -> str:
        """Compute SHA1 hash of signals for integrity check."""
        payload = json.dumps(
            {
                "date": date.isoformat(),
                "signals": signals,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
