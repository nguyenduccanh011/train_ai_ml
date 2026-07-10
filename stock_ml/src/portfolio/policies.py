"""Pluggable sizing policies: AlphaFrame -> TargetWeightFrame.

Every policy subclasses `_BasePolicy`, which owns the shared per-date pipeline:

    raw weights (policy-specific)
      -> direction filter (long / short / long_short / market_neutral)
      -> regime gate
      -> size scale
      -> normalize gross to max_gross
      -> clamp per-name |weight| <= max_per_name

so individual policies only implement `_raw_weights(group, ctx)` returning a signed
Series indexed by symbol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml.src.contracts import AlphaSpec, TARGET_WEIGHT_COLUMNS
from stock_ml.src.portfolio.base import PortfolioContext
from stock_ml.src.portfolio.gating import apply_regime_gate, apply_size_scale


class _BasePolicy:
    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        raise NotImplementedError

    def build(self, alpha: pd.DataFrame, ctx: PortfolioContext) -> pd.DataFrame:
        AlphaSpec.validate(alpha)
        out_rows = []
        for date, group in alpha.groupby("date", sort=True):
            group = group.drop_duplicates(subset="symbol").set_index("symbol")
            raw = self._raw_weights(group, ctx).astype(float)
            weights, gated = self._finalize(raw, group, ctx, date)
            scores = group["score"].reindex(weights.index)
            rank = scores.rank(ascending=False, method="first")
            for sym in weights.index:
                w = float(weights[sym])
                out_rows.append(
                    {
                        "date": date,
                        "symbol": sym,
                        "target_weight": w,
                        "side": int(np.sign(w)),
                        "rank": int(rank[sym]),
                        "gated": bool(gated.get(sym, False)),
                        "score": float(scores[sym]),
                    }
                )
        if not out_rows:
            return pd.DataFrame(columns=list(TARGET_WEIGHT_COLUMNS))
        return pd.DataFrame(out_rows, columns=list(TARGET_WEIGHT_COLUMNS))

    @staticmethod
    def _direction_filter(raw: pd.Series, direction: str) -> pd.Series:
        if direction == "long":
            return raw.clip(lower=0.0)
        if direction == "short":
            return raw.clip(upper=0.0)
        # long_short / market_neutral: keep both signs
        return raw

    def _finalize(
        self, raw: pd.Series, group: pd.DataFrame, ctx: PortfolioContext, date
    ) -> tuple[pd.Series, pd.Series]:
        w = self._direction_filter(raw, ctx.direction)
        w, gated = apply_regime_gate(w, ctx.regime_signal, date)
        w = apply_size_scale(w, ctx.size_signal, date)
        # Normalize gross to max_gross (full deployment), then cap per-name.
        gross = w.abs().sum()
        if gross > 0:
            w = w / gross * ctx.max_gross
        w = w.clip(lower=-ctx.max_per_name, upper=ctx.max_per_name)
        return w, gated


class ThresholdBinaryPolicy(_BasePolicy):
    """Equal-weight book from a hysteresis threshold band (legacy-equivalent sides).

    score > entry_threshold -> +1, score < exit_threshold -> -1, else 0.
    """

    def __init__(self, entry_threshold: float = 0.0, exit_threshold: float = 0.0):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        s = group["score"]
        side = np.where(s > self.entry_threshold, 1.0, np.where(s < self.exit_threshold, -1.0, 0.0))
        return pd.Series(side, index=group.index)


class ScoreProportionalPolicy(_BasePolicy):
    """Weight proportional to (signed) score within each date."""

    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        return group["score"].astype(float)


class TopKPolicy(_BasePolicy):
    """Select the k best names by score (plus k worst when both sides are allowed).

    weighting: "equal" (unit per selected name) or "score" (proportional to |score|).
    """

    def __init__(self, k: int = 10, weighting: str = "equal"):
        if k <= 0:
            raise ValueError("TopKPolicy requires k > 0")
        self.k = k
        self.weighting = weighting

    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        s = group["score"].astype(float)
        ordered = s.sort_values(ascending=False)
        both_sides = ctx.direction in ("long_short", "market_neutral")
        longs = ordered.head(self.k).index
        shorts = ordered.tail(self.k).index if both_sides else pd.Index([])

        w = pd.Series(0.0, index=group.index)
        if self.weighting == "score":
            w.loc[longs] = s.loc[longs].abs()
            w.loc[shorts] = -s.loc[shorts].abs()
        else:
            w.loc[longs] = 1.0
            w.loc[shorts] = -1.0
        return w


class MarketNeutralPolicy(_BasePolicy):
    """Dollar-neutral top-k long / bottom-k short, equal weight (net ≈ 0)."""

    def __init__(self, k: int = 10):
        if k <= 0:
            raise ValueError("MarketNeutralPolicy requires k > 0")
        self.k = k

    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        s = group["score"].astype(float)
        ordered = s.sort_values(ascending=False)
        n = min(self.k, len(ordered) // 2)
        w = pd.Series(0.0, index=group.index)
        if n == 0:
            return w
        w.loc[ordered.head(n).index] = 1.0
        w.loc[ordered.tail(n).index] = -1.0
        return w

    def _finalize(self, raw, group, ctx, date):
        # Force two-sided book regardless of configured direction.
        ctx_two = PortfolioContext(
            direction="long_short",
            max_gross=ctx.max_gross,
            max_per_name=ctx.max_per_name,
            regime_signal=ctx.regime_signal,
            size_signal=ctx.size_signal,
            ohlcv=ctx.ohlcv,
        )
        return super()._finalize(raw, group, ctx_two, date)


class VolTargetPolicy(_BasePolicy):
    """Score-proportional weights scaled by inverse realized volatility.

    Requires ctx.ohlcv ([symbol, date, close]); names without enough history fall back
    to plain score-proportional weighting.
    """

    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback

    def _realized_vol(self, ctx: PortfolioContext, date, symbols) -> pd.Series:
        vols = pd.Series(np.nan, index=symbols)
        if ctx.ohlcv is None or ctx.ohlcv.empty:
            return vols
        hist = ctx.ohlcv[ctx.ohlcv["date"] <= date]
        for sym in symbols:
            closes = hist[hist["symbol"] == sym]["close"].tail(self.lookback + 1)
            if len(closes) >= self.lookback:
                vols[sym] = closes.pct_change().dropna().std(ddof=0)
        return vols

    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        s = group["score"].astype(float)
        # date is the group key; recover it from the alpha (all rows share it upstream)
        date = group["date"].iloc[0] if "date" in group.columns else None
        vols = self._realized_vol(ctx, date, group.index)
        scale = (self.target_vol / vols).where(vols > 0, 1.0).fillna(1.0)
        return s * scale


class KellyFractionalPolicy(_BasePolicy):
    """Fractional-Kelly style: weight ∝ score (expected edge) × fraction."""

    def __init__(self, fraction: float = 0.5):
        self.fraction = fraction

    def _raw_weights(self, group: pd.DataFrame, ctx: PortfolioContext) -> pd.Series:
        return group["score"].astype(float) * self.fraction
