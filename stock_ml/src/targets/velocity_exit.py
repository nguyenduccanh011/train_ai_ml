"""Velocity-exit regression target (sell-side).

A turnover-aware generalisation of ``RiskExitRegressionTarget``. ``risk_exit`` holds
as long as *any* upside remains anywhere in the forward window, so on a slow grind up
it never sells — the position becomes a buy-and-hold (the live champion's ~158-bar
holds). This target keeps risk_exit's "don't sell into a real run" behaviour but stops
rewarding upside that only arrives *far* in the future: downside is measured over the
full ``horizon`` H, upside only over a short ``upside_horizon`` U <= H.

For each row `t`:
  fwd_downside(t)   = 1 - min_{k=1..H}( close[t+k] / close[t] )     # all coming risk, >= 0
  fwd_upside_near(t)= max_{k=1..U}( close[t+k] / close[t] ) - 1     # only NEAR upside, >= 0
  target(t)         = fwd_downside(t) - fwd_upside_near(t)

High value = sell. Behaviour vs risk_exit:
  - fast pop ahead (within U bars)  -> near-upside high -> target low  -> HOLD the move
  - slow grind (gain only past U)   -> near-upside ~0   -> target ~risk -> SELL (free capital)
  - imminent drop                   -> downside high    -> target high  -> SELL

So a position that only pays off slowly is exited early and the freed capital can chase a
faster setup, lifting trade count and pnl-per-hold velocity. With U == H this reduces
*exactly* to RiskExitRegressionTarget (continuity / backward-compat sanity).

Tail-NaN at the longest horizon H (last H rows per symbol unobservable), keeping the
fail-loud ``require_no_nan`` guard intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VelocityExitRegressionTarget:
    """Forward downside (horizon H) minus near-forward upside (horizon U<=H), as regression."""

    horizon: int = 20
    upside_horizon: int = 5
    # VOL-NORMALIZE (project deep-finding: "only VOL predicts MAGNITUDE"): divide the raw
    # downside-minus-near-upside label by the symbol's own trailing return-vol (std of daily
    # returns over vol_window, causal/backward-looking). The head then predicts ABNORMAL coming
    # downside relative to this stock's normal noise — so it HOLDS through ordinary high-vol
    # wiggles (a 5% drop on an 8%-vol name is normal -> low target) and SELLS only on a genuine
    # regime break (a 5% drop on a 2%-vol name is 2.5 sigma -> high target). Targets the
    # clip-winners wall (plain ratio over-sells high-vol runners). The normalizer uses only PAST
    # returns (no leak); the forward span is unchanged (still horizon H). False = plain ratio.
    vol_normalize: bool = False
    vol_window: int = 20

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.upside_horizon < 1:
            raise ValueError(f"upside_horizon must be >= 1, got {self.upside_horizon}")
        if self.upside_horizon > self.horizon:
            raise ValueError(
                f"upside_horizon ({self.upside_horizon}) must be <= horizon ({self.horizon})"
            )
        if self.vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {self.vol_window}")

    def _label(self, close: pd.Series) -> pd.Series:
        down_shifts = [close.shift(-k) for k in range(1, self.horizon + 1)]
        up_shifts = [close.shift(-k) for k in range(1, self.upside_horizon + 1)]
        # NaN once the full (longest) window runs past the series end.
        observable = close.shift(-self.horizon).notna()
        fwd_min = pd.concat(down_shifts, axis=1).min(axis=1).where(observable)
        fwd_max_near = pd.concat(up_shifts, axis=1).max(axis=1).where(observable)
        fwd_downside = 1.0 - fwd_min / close
        fwd_upside_near = fwd_max_near / close - 1.0
        label = fwd_downside - fwd_upside_near
        if self.vol_normalize:
            # Causal trailing return-vol of THIS symbol (past returns only -> no leak).
            vol = close.pct_change().rolling(self.vol_window, min_periods=2).std()
            label = label / (vol + 1e-4)
        return label

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        out["target"] = out.groupby("symbol")[close_col].transform(self._label)
        return out


@dataclass(frozen=True)
class CrossSectionalExitTarget:
    """Cross-sectional (market-neutral) forward-underperformance exit target (2026-07-11 root work).

    velocity_exit (absolute forward downside) is REGIME-FRAGILE: its exit-score IC flips sign by
    regime (2022 bear −0.14 correct, 2024 chop +0.11 INVERTED — it sells names that then rise),
    because "coming downside" means a real top in a bear but a recoverable dip in chop. A
    CROSS-SECTIONAL target removes the market direction: sell the names that will UNDERPERFORM
    THEIR PEERS over the next `horizon`, regardless of whether the whole market rises or falls.
    Same principle that made entry-RS / exit-RS regime-robust (per-symbol relative survives).

      fwd_ret(t)  = close[t+H]/close[t] - 1
      target(t)   = -( fwd_ret(t) - mean_over_symbols_on_date(fwd_ret) )   # high = underperforms = SELL

    Regime-robust because the cross-sectional demean cancels the common market move. Tail-NaN at H.
    """

    horizon: int = 10

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns or "date" not in df.columns:
            raise ValueError("df must contain 'symbol' and 'date'")
        out = df.copy()
        out["_fwd"] = out.groupby("symbol")[close_col].transform(
            lambda c: c.shift(-self.horizon) / c - 1.0
        )
        mkt = out.groupby("date")["_fwd"].transform("mean")
        out["target"] = -(out["_fwd"] - mkt)
        return out.drop(columns=["_fwd"])


@dataclass(frozen=True)
class ExitAmplitudeExhaustionTarget:
    """Exit target = negative forward UPSIDE amplitude (2026-07-11, user intuition: entry knows
    amplitude robustly, so amplitude-exhaustion IS the exit signal).

    The current exit head (velocity_exit = forward downside MINUS near upside) predicts a signed
    RETURN, which is regime-confounded (a drop = 'top' in bear, 'dip' in bull → exit IC FLIPS by
    regime). But the ENTRY's mfe head predicts forward UPSIDE amplitude robustly (IC +0.21 every
    year). Amplitude/magnitude is regime-INVARIANT. So frame the exit as amplitude-exhaustion:
    predict how LITTLE upside remains; SELL when no upside is coming (free the capital), regardless
    of whether the tape rises or falls.

      fwd_mfe(t) = max_{k=1..H}( high[t+k] / close[t] ) - 1     # forward upside amplitude, >= 0
      target(t)  = -fwd_mfe(t)                                   # HIGH (near 0) = no upside = SELL

    Sells when the coming upside is exhausted (regime-robust because it tracks magnitude, not
    signed direction). Optional vol_normalize divides by trailing return-vol so 'exhaustion' is
    relative to the symbol's own swing size. Tail-NaN at H.
    """

    horizon: int = 10
    vol_normalize: bool = False
    vol_window: int = 40

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        high_col = "high" if "high" in out.columns else close_col

        def _label(g: pd.DataFrame) -> pd.Series:
            c = g[close_col]
            observable = c.shift(-self.horizon).notna()
            fmax = pd.concat(
                [g[high_col].shift(-k) for k in range(1, self.horizon + 1)], axis=1
            ).max(axis=1)
            mfe = (fmax / c - 1.0).where(observable)
            lab = -mfe
            if self.vol_normalize:
                vol = c.pct_change().rolling(self.vol_window, min_periods=2).std()
                lab = lab / (vol + 1e-4)
            return lab

        out["target"] = out.groupby("symbol", group_keys=False).apply(_label)
        return out


@dataclass(frozen=True)
class CrossSectionalEntryTarget:
    """Cross-sectional (market-neutral) forward-OUTPERFORMANCE entry target (2026-07-11 root work).

    Mirror of CrossSectionalExitTarget for the BUY side. Diagnostic (hb_87) showed the entry head
    ranks AMPLITUDE robustly (mfe20 IC +0.21, positive every year) but its DIRECTION signal
    (fwd_ret IC) FLIPS in dead years (2024 −0.10, 2026 −0.05) and is even worse cross-sectionally
    (2024 DIR-CS −0.125) — i.e. it captures direction via market BETA (common factor), exactly what
    inverts in chop. This target removes the market move: buy names that will OUTPERFORM THEIR PEERS
    over the next `horizon`, regardless of whether the whole market rises or falls, so the head is
    forced to learn cross-sectional (regime-robust) direction instead of beta.

      fwd_ret(t)  = close[t+H]/close[t] - 1
      target(t)   = +( fwd_ret(t) - mean_over_symbols_on_date(fwd_ret) )   # high = outperforms = BUY

    Tail-NaN at H. `amp_weight` optionally re-injects amplitude: target *= (1 + amp_weight * mfe_rank)
    is NOT applied here (kept pure cross-sectional direction); amplitude stays with the ensemble's
    other heads / the entry_recov_rs feature set.
    """

    horizon: int = 10

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns or "date" not in df.columns:
            raise ValueError("df must contain 'symbol' and 'date'")
        out = df.copy()
        out["_fwd"] = out.groupby("symbol")[close_col].transform(
            lambda c: c.shift(-self.horizon) / c - 1.0
        )
        mkt = out.groupby("date")["_fwd"].transform("mean")
        out["target"] = out["_fwd"] - mkt
        return out.drop(columns=["_fwd"])


@dataclass(frozen=True)
class AmplitudeDirectionEntryTarget:
    """Amplitude-signed cross-sectional direction entry target (2026-07-11 root synthesis).

    Diagnostic (hb_87/88): the entry head's real strength is AMPLITUDE (mfe20 IC +0.21, positive
    every year — it knows which names will SWING big) but its DIRECTION flips in dead years
    (2024 fwd IC −0.10). A pure cross_sectional_entry target FIXES direction (2024 → +0.02) but
    DESTROYS amplitude (IC 0.21 → 0.06) and NAV drops (x31.8 → x30). This target keeps BOTH:
    reward high forward AMPLITUDE, but SIGNED by whether the name out/under-performs its peers.

      mfe(t)   = max_{k=1..H}( high[t+k] / close[t] ) - 1            # realized up-amplitude, >= 0
      cs(t)    = fwd_ret(t) - mean_over_symbols_on_date(fwd_ret)     # cross-sectional direction
      dir(t)   = tanh( cs(t) / scale )                               # soft sign in (-1, 1)
      target(t)= mfe(t) * dir(t)

    So a big-amplitude name that OUTPERFORMS peers → large positive (BUY); a big-amplitude name
    that UNDERPERFORMS peers (the 2024 trap: swings big but DOWN) → large negative (AVOID). The
    magnitude (mfe) is preserved so the amplitude ranking survives, while the sign is market-neutral
    (regime-robust) so the dead-year direction inversion is removed. Tail-NaN at H.
    """

    horizon: int = 20
    scale: float = (
        0.05  # cs excess-return scale for the tanh soft-sign (~5% relative move → ~tanh(1))
    )

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale}")

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns or "date" not in df.columns:
            raise ValueError("df must contain 'symbol' and 'date'")
        out = df.copy()
        high_col = "high" if "high" in out.columns else close_col

        def _mfe(g: pd.DataFrame) -> pd.Series:
            c = g[close_col]
            observable = c.shift(-self.horizon).notna()
            fmax = pd.concat(
                [g[high_col].shift(-k) for k in range(1, self.horizon + 1)], axis=1
            ).max(axis=1)
            return (fmax / c - 1.0).where(observable)

        out["_mfe"] = out.groupby("symbol", group_keys=False).apply(_mfe)
        out["_fwd"] = out.groupby("symbol")[close_col].transform(
            lambda c: c.shift(-self.horizon) / c - 1.0
        )
        cs = out["_fwd"] - out.groupby("date")["_fwd"].transform("mean")
        out["target"] = out["_mfe"] * np.tanh(cs / self.scale)
        return out.drop(columns=["_mfe", "_fwd"])


@dataclass(frozen=True)
class VelocityExitRegimeTarget:
    """Regime-conditional velocity-exit (shakeout-vs-top, 2026-07-11 forensic).

    velocity_exit sells whenever coming downside has no upside within a FIXED near window U.
    Forensic (hb_60/61): 49% of 'signal' exits are shakeouts (a dip that recovers within
    ~20 bars) and the ONE feature that separates a shakeout from a real top is MARKET REGIME
    (equal-weight index > MA(regime_ma) -> AUC 0.78): a dip in a bull tape bounces, a dip in a
    bear tape is a top. Feature-only failed (the head fits this target, not the feature), so we
    bake the regime into the LABEL: use a LONG upside window in a bull tape (so recoverable dips
    score low -> HOLD the shakeout) and a SHORT one in a bear tape (near downside -> SELL the top).

    Self-contained: builds the equal-weight index from df's own closes (causal MA), no external
    column needed. With upside_horizon_bull == upside_horizon_bear this reduces to VelocityExit.
    """

    horizon: int = 20
    upside_horizon_bull: int = 20  # bull tape: long near-window -> hold recoverable dips
    upside_horizon_bear: int = 5  # bear tape: short near-window -> sell into coming downside
    regime_ma: int = 50  # index > MA(regime_ma) = bull (forensic: MA50, AUC 0.78)
    vol_normalize: bool = True
    vol_window: int = 40

    def __post_init__(self) -> None:
        for u in (self.upside_horizon_bull, self.upside_horizon_bear):
            if not (1 <= u <= self.horizon):
                raise ValueError(f"upside_horizon {u} must be in [1, horizon={self.horizon}]")
        if self.regime_ma < 2:
            raise ValueError("regime_ma must be >= 2")

    def _downside(self, close: pd.Series) -> pd.Series:
        observable = close.shift(-self.horizon).notna()
        fwd_min = (
            pd.concat([close.shift(-k) for k in range(1, self.horizon + 1)], axis=1)
            .min(axis=1)
            .where(observable)
        )
        return 1.0 - fwd_min / close

    def _upside_near(self, close: pd.Series, u: int) -> pd.Series:
        observable = close.shift(-self.horizon).notna()
        fwd_max = (
            pd.concat([close.shift(-k) for k in range(1, u + 1)], axis=1)
            .max(axis=1)
            .where(observable)
        )
        return fwd_max / close - 1.0

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        out = df.copy()
        # Causal equal-weight market regime (bull = index > its own MA(regime_ma)).
        piv = out.pivot_table(index="date", columns="symbol", values=close_col).sort_index()
        idx = (1.0 + piv.pct_change().mean(axis=1)).cumprod()
        bull = idx > idx.rolling(self.regime_ma, min_periods=self.regime_ma // 2).mean()
        bull_map = bull.reindex(out["date"]).to_numpy()  # per-row regime aligned by date

        g = out.groupby("symbol")[close_col]
        down = g.transform(self._downside)
        up_bull = g.transform(lambda c: self._upside_near(c, self.upside_horizon_bull))
        up_bear = g.transform(lambda c: self._upside_near(c, self.upside_horizon_bear))
        up = up_bull.where(pd.Series(bull_map, index=out.index).fillna(False), up_bear)
        label = down - up
        if self.vol_normalize:
            vol = g.transform(
                lambda c: c.pct_change().rolling(self.vol_window, min_periods=2).std()
            )
            label = label / (vol + 1e-4)
        out["target"] = label
        return out
