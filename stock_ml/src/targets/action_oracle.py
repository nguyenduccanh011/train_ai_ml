"""Action-oracle target — hard {OUT, ENTER, HOLD, EXIT} labels for a SINGLE model.

Unlike the decoupled dual-ML system (a separate entry head and exit head, whose
realized PnL couples them so neither can be improved in isolation — the "masking"
problem), this target lets ONE multiclass classifier learn the whole trade decision:
when to ENTER, HOLD, EXIT, or stand OUT. The labels come from an oracle zigzag swing
segmentation:

  - ENTER (1): at a confirmed swing BOTTOM whose forward up-leg (to the next peak) is
               worth buying (>= ``min_fwd_leg``).
  - HOLD  (2): every bar strictly inside that bottom -> peak holding interval.
  - EXIT  (3): at the peak that closes the holding interval.
  - OUT   (0): every other confirmed bar (down-legs / weak legs you should sit out).

The classifier imitates this oracle from PAST-ONLY features; at inference it never sees
the future. Like ``zigzag.py``, the LABEL is intentionally future-derived (a pivot is
confirmed only after price reverses) — that is normal supervised learning. This module
writes ONLY the label column and never touches feature columns.

Leakage note: a label's forward dependence is the distance to its confirming pivot,
which is unbounded in principle. The train/test gap (``gap_days``) blocks boundary
contamination; reuse the zigzag-grade gap (>= 85) and audit. See ``target_forward_span``
in the registry (registered to ``ZIGZAG_FORWARD_SPAN``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_ml.src.targets.zigzag import _profitable_pivots, _zigzag_pivots

# Class codes — kept in sync with the SMAC dispatch branch in experiment.py.
# CUT (4) is emitted only when cut_drawdown > 0 (5-class oracle); the classifier reuses
# the same multiclass head and the SMAC branch treats CUT as a SELL, same as EXIT.
OUT, ENTER, HOLD, EXIT, CUT = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class ActionOracleTarget:
    """Hard 4-class action labels from an oracle zigzag swing segmentation.

    Args:
        pct: reversal threshold defining a swing (e.g. 0.10 = 10%).
        min_leg_bars: drop swings shorter than this many bars from the previous
            confirmed pivot (noise filter); 0 disables it.
        min_fwd_leg: only ENTER at bottoms whose up-leg to the next peak moves
            >= this fraction (a bounce worth buying); 0 = enter at every bottom.
        entry_confirm_pct: if > 0, shift ENTER from the exact zigzag bottom to the first
            bar in the up-leg that closes >= this fraction ABOVE the bottom — a confirmed
            reversal (a TA waits for the turn instead of catching the exact falling-knife
            low). Bars between the bottom and the confirmation stay OUT. 0 = enter at the
            bottom (legacy).
        entry_min_ret_120: if set, only ENTER at a bottom whose trailing 120-bar return
            (close[b]/close[b-120]-1, causal/past-only) is STRICTLY ABOVE this value — a
            regime gate that drops bottoms sitting in a strong downtrend (the recurring
            knife-catch cohort: fallen >10% over 120d). None = no regime gate. Bottoms
            failing the gate are left OUT (not entered, not held).
        cut_drawdown: if > 0, within a holding interval [bottom->peak] label CUT (4) at
            the FIRST bar whose close is >= this fraction BELOW the entry (bottom) close —
            a calibrated loss-cut signal, distinct from the noisy OUT class. After the cut
            the position is closed, so the rest of the interval stays OUT. 0 = 4-class
            (no CUT; HOLD all the way to the peak then EXIT).
        target_col: output column name (default "target").
    """

    pct: float = 0.10
    min_leg_bars: int = 0
    min_fwd_leg: float = 0.0
    cut_drawdown: float = 0.0
    cut_struct_ma: int | None = None
    cut_struct_supp: int = 20
    entry_min_ret_120: float | None = None
    entry_confirm_pct: float = 0.0
    entry_weekly_ma: int | None = None
    entry_weekly_lb: int = 10
    inner_swing_pct: float = 0.0
    exit_overext_ma: int | None = None
    exit_overext_pct: float = 0.15
    target_col: str = "target"

    def __post_init__(self) -> None:
        if not (0.0 < self.pct < 1.0):
            raise ValueError(f"pct must be in (0,1), got {self.pct}")
        if self.min_leg_bars < 0:
            raise ValueError(f"min_leg_bars must be >= 0, got {self.min_leg_bars}")
        if self.min_fwd_leg < 0:
            raise ValueError(f"min_fwd_leg must be >= 0, got {self.min_fwd_leg}")
        if not (0.0 <= self.cut_drawdown < 1.0):
            raise ValueError(f"cut_drawdown must be in [0,1), got {self.cut_drawdown}")

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        if close_col not in df.columns:
            raise ValueError(f"df must contain '{close_col}'")
        out = df.copy()

        def _labels_per_symbol(g: pd.DataFrame) -> pd.DataFrame:
            close = g[close_col].to_numpy(dtype=np.float64)
            n = len(close)
            g = g.copy()
            g[self.target_col] = _action_labels(
                close, self.pct, self.min_leg_bars, self.min_fwd_leg,
                self.cut_drawdown, self.entry_min_ret_120, self.entry_confirm_pct,
                self.cut_struct_ma, self.cut_struct_supp,
                self.exit_overext_ma, self.exit_overext_pct,
                self.entry_weekly_ma, self.entry_weekly_lb,
                self.inner_swing_pct,
            )
            return g

        out = out.groupby("symbol", group_keys=False).apply(_labels_per_symbol)
        return out


def _inner_subtrade_pos(close_sub: np.ndarray, inner_pct: float) -> np.ndarray:
    """In-position (1) / flat (0) per bar across a rising leg, EXITING at each sub-peak and
    RE-ENTERING at the following sub-trough whenever an internal counter-swing reverses by
    >= inner_pct (trade the sub-swings instead of holding through them). Entered at index 0."""
    n = len(close_sub)
    pos = np.ones(n, dtype=np.int8)
    if n < 3 or inner_pct <= 0:
        return pos
    sb, sp = _zigzag_pivots(close_sub, inner_pct)
    piv = sorted([(i, "b") for i in sb] + [(i, "p") for i in sp])
    state, prev = 1, 0
    for pi, typ in piv:
        if typ == "p":  # hold through the sub-peak, exit after
            pos[prev : pi + 1] = state
            state, prev = 0, pi + 1
        else:  # counter-swing OUT up to the sub-trough, re-enter AT it
            pos[prev:pi] = state
            pos[pi] = 1
            state, prev = 1, pi + 1
    pos[prev:] = state
    return pos


def _action_labels(
    close: np.ndarray,
    pct: float,
    min_leg_bars: int,
    min_fwd_leg: float,
    cut_drawdown: float = 0.0,
    entry_min_ret_120: float | None = None,
    entry_confirm_pct: float = 0.0,
    cut_struct_ma: int | None = None,
    cut_struct_supp: int = 20,
    exit_overext_ma: int | None = None,
    exit_overext_pct: float = 0.15,
    entry_weekly_ma: int | None = None,
    entry_weekly_lb: int = 10,
    inner_swing_pct: float = 0.0,
) -> np.ndarray:
    """Per-bar {OUT, ENTER, HOLD, EXIT, CUT} (float, NaN tail past last confirmed pivot)."""
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return labels

    bottoms, peaks = _zigzag_pivots(close, pct, min_leg_bars)
    piv = sorted([(i, "b") for i in bottoms] + [(i, "p") for i in peaks])
    if not piv:
        return labels  # no confirmed swing — leave all NaN (dropped from train)

    # Everything up to the last CONFIRMED pivot defaults to OUT (flat). Bars after it
    # belong to an unconfirmed leg → stay NaN so the splitter drops them from training.
    last_confirmed = piv[-1][0]
    labels[: last_confirmed + 1] = OUT

    # Only buy bottoms whose forward up-leg clears min_fwd_leg (a real bounce).
    if min_fwd_leg > 0:
        good_bottoms = set(_profitable_pivots(close, bottoms, peaks, "bottom", min_fwd_leg))
    else:
        good_bottoms = set(bottoms)

    # OVEREXTENSION EXIT (sell into strength to re-buy lower, not at the exact peak): precompute
    # the trailing MA so the exit can fire at the first bar in an up-leg that is stretched
    # >= exit_overext_pct above it — a more learnable/mean-revertible signal than the precise top.
    oe_ma = None
    if exit_overext_ma is not None:
        oe_ma = pd.Series(close).rolling(exit_overext_ma, min_periods=exit_overext_ma).mean().to_numpy()

    # MULTI-TIMEFRAME entry gate: a long MA (proxy for the WEEKLY trend, e.g. 50d ~= 10 weeks)
    # that is RISING = the higher timeframe has turned up. Only ENTER a daily bottom when the
    # weekly trend is also turning up (denoise — skip daily bounces that are counter-trend on
    # the weekly). Causal (trailing MA + past slope).
    wk_ma = None
    if entry_weekly_ma is not None:
        wk_ma = pd.Series(close).rolling(entry_weekly_ma, min_periods=entry_weekly_ma).mean().to_numpy()

    # Carve each profitable bottom -> next-peak interval as a holding period.
    for k, (idx, typ) in enumerate(piv):
        if typ != "b" or idx not in good_bottoms or k + 1 >= len(piv):
            continue
        # REGIME gate (causal, past-only): skip ENTER on a bottom sitting in a strong 120-bar
        # downtrend — the recurring knife-catch cohort (forensic: bottoms with ret_120d <= -10%
        # carry a 7% catastrophic-loss rate). A skipped bottom's bars stay OUT (not entered).
        if entry_min_ret_120 is not None:
            if idx < 120 or close[idx - 120] <= 0:
                continue
            if close[idx] / close[idx - 120] - 1.0 <= entry_min_ret_120:
                continue
        # MULTI-TIMEFRAME gate: weekly-trend MA must be RISING at the bottom (higher TF turned up).
        if wk_ma is not None:
            if idx < entry_weekly_ma + entry_weekly_lb:
                continue
            if np.isnan(wk_ma[idx]) or np.isnan(wk_ma[idx - entry_weekly_lb]) or wk_ma[idx] <= wk_ma[idx - entry_weekly_lb]:
                continue
        peak_idx = piv[k + 1][0]  # pivots alternate, so the next pivot is the peak
        # ENTRY CONFIRMATION: optionally wait for price to reclaim entry_confirm_pct above
        # the bottom (a confirmed reversal) instead of entering the exact falling-knife low.
        enter_idx = idx
        if entry_confirm_pct > 0 and close[idx] > 0:
            need = close[idx] * (1.0 + entry_confirm_pct)
            conf = None
            for j in range(idx + 1, peak_idx):
                if close[j] >= need:
                    conf = j
                    break
            if conf is None:
                continue  # up-leg never confirmed before the peak -> skip this bottom
            enter_idx = conf  # bars idx..conf-1 stay OUT (waiting for confirmation)
        # Exit at the first overextended bar (sell strength) if enabled, else at the peak.
        exit_idx = peak_idx
        if oe_ma is not None:
            for j in range(enter_idx + 1, peak_idx + 1):
                if (not np.isnan(oe_ma[j]) and oe_ma[j] > 0
                        and close[j] / oe_ma[j] - 1.0 >= exit_overext_pct):
                    exit_idx = j
                    break
        # INNER SUB-SWING decomposition: instead of holding the whole leg, trade the internal
        # counter-swings >= inner_swing_pct (sell the sub-peak, re-buy the sub-trough) — kills the
        # buy-and-hold bias that misses the sub-waves. Only with the plain peak-exit (no overext/cut).
        if (inner_swing_pct > 0 and oe_ma is None and exit_idx == peak_idx):
            spos = _inner_subtrade_pos(close[enter_idx : exit_idx + 1], inner_swing_pct)
            for k in range(len(spos)):
                gk = enter_idx + k
                prevp = spos[k - 1] if k > 0 else 0
                if spos[k] == 1:
                    labels[gk] = ENTER if prevp == 0 else HOLD
                else:
                    labels[gk] = EXIT if prevp == 1 else OUT
            labels[exit_idx] = EXIT  # force a clean exit at the leg's final peak
        else:
            labels[enter_idx] = ENTER
            if exit_idx > enter_idx + 1:
                labels[enter_idx + 1 : exit_idx] = HOLD
            labels[exit_idx] = EXIT
        # bars (exit_idx, peak_idx] stay OUT — exited early into strength, re-buy the dip later

    # CALIBRATED LOSS-CUT on the DOWN-LEGS: a hindsight zigzag never ENTERs a bad trade,
    # so a cut signal can't come from its own holding intervals (they are clean up-legs —
    # an intra-interval drop >= pct would have split the pivot; verified ~0 CUT labels).
    # The realistic place a mistakenly-held position bleeds is the DECLINE after a peak.
    # Label CUT on each peak->next-bottom down-leg once price has fallen >= cut_drawdown
    # below that peak. At inference a knife the model wrongly bought looks like exactly
    # this falling-off-a-high pattern -> CUT -> exit. Unlike a blanket exit-on-OUT (v2),
    # this fires only on real declines, not on minor pullbacks inside a healthy up-leg.
    if cut_drawdown > 0 or cut_struct_ma is not None:
        ma_arr = supp_arr = None
        if cut_struct_ma is not None:
            cser = pd.Series(close)
            ma_arr = cser.rolling(cut_struct_ma, min_periods=cut_struct_ma).mean().to_numpy()
            # trailing support = min close over the prior cut_struct_supp bars (shifted, causal)
            supp_arr = cser.rolling(cut_struct_supp, min_periods=cut_struct_supp).min().shift(1).to_numpy()
        for k, (idx, typ) in enumerate(piv):
            if typ != "p" or k + 1 >= len(piv):
                continue
            next_bottom = piv[k + 1][0]
            peak_px = close[idx]
            if peak_px <= 0:
                continue
            floor = peak_px * (1.0 - cut_drawdown) if cut_drawdown > 0 else None
            for j in range(idx + 1, next_bottom):  # exclusive of the next bottom (=ENTER)
                if cut_struct_ma is not None:
                    # CONFIRMED structure break: close pierces BOTH the trailing support AND
                    # the MA (trend) — a real breakdown, not a shallow pullback.
                    if (not np.isnan(supp_arr[j]) and not np.isnan(ma_arr[j])
                            and close[j] < supp_arr[j] and close[j] < ma_arr[j]):
                        labels[j] = CUT
                elif floor is not None and close[j] <= floor:
                    labels[j] = CUT

    return labels
