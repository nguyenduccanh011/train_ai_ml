"""Max-amplitude action oracle — the perfect-foresight, cost-aware, long-only optimal policy.

Instead of labeling actions by zigzag-pivot geometry (action_oracle), this target solves
for the long-only position sequence that DIRECTLY MAXIMIZES realized profit (the sum of
captured log-returns net of a round-trip cost), via the classic unlimited-transactions-
with-fee dynamic program. Backtracking the DP yields, per bar, the optimal position
(long / flat); transitions map to the 4-class action {OUT, ENTER, HOLD, EXIT}.

Why this targets "two-way profit" (good entry WITH good exit): the DP jointly picks the
entry (a low) and the exit (the following high) of every swing whose amplitude beats the
cost — it is the policy that captures the maximum realizable amplitude. The `fee` knob is
the minimum swing worth trading (replaces zigzag pct / min_fwd_leg). The DP also AVOIDS
downtrends natively: it only goes long when a profitable exit lies ahead, so a sustained
decline (no such exit) is left FLAT — subsuming the v5 regime lesson without an explicit
gate.

Leakage note: like zigzag.py / action_oracle.py the LABEL is intentionally future-derived
(normal supervised learning); only the label column is written, features stay past-only.
The optimal policy's forward dependence is the next swing; the train/test gap (>= 85)
blocks boundary contamination (registered to ZIGZAG_FORWARD_SPAN in the registry).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Class codes — identical mapping to action_oracle (the SMAC dispatch branch consumes these).
OUT, ENTER, HOLD, EXIT = 0, 1, 2, 3


@dataclass(frozen=True)
class MaxProfitActionTarget:
    """Perfect-foresight cost-aware long-only optimal action labels {OUT, ENTER, HOLD, EXIT}.

    Args:
        fee: round-trip cost charged once per completed trade, in return units (e.g. 0.03
            = a swing must clear ~3% to be worth taking). Higher = fewer, larger swings.
        entry_min_ret_120: if set, the DP may only OPEN a position at a bar whose trailing
            120-bar return (causal) is STRICTLY ABOVE this value — a regime constraint that
            stops the optimal policy from entering counter-trend bounces inside a downtrend
            (the unpredictable knife entries that hurt the unconstrained amplitude oracle).
            None = unconstrained.
        target_col: output column name (default "target").
    """

    fee: float = 0.03
    entry_min_ret_120: float | None = None
    target_col: str = "target"

    def __post_init__(self) -> None:
        if not (0.0 <= self.fee < 1.0):
            raise ValueError(f"fee must be in [0,1), got {self.fee}")

    def apply(self, df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        if "symbol" not in df.columns:
            raise ValueError("df must contain 'symbol'")
        if close_col not in df.columns:
            raise ValueError(f"df must contain '{close_col}'")
        out = df.copy()

        def _labels_per_symbol(g: pd.DataFrame) -> pd.DataFrame:
            close = g[close_col].to_numpy(dtype=np.float64)
            g = g.copy()
            g[self.target_col] = _maxprofit_labels(close, self.fee, self.entry_min_ret_120)
            return g

        out = out.groupby("symbol", group_keys=False).apply(_labels_per_symbol)
        return out


def _maxprofit_positions(
    logp: np.ndarray, fee_log: float, can_buy: np.ndarray | None = None
) -> np.ndarray:
    """Optimal long-only position per bar (1=hold at close, 0=flat) maximizing net log-profit.

    Unlimited transactions with a per-trade fee (LeetCode 714 in log-space), with backtrack.
    ``can_buy`` (optional bool per bar) gates where a position may be OPENED (regime
    constraint); selling/holding is always allowed so an open trade can still be closed.
    """
    n = len(logp)
    pos = np.zeros(n, dtype=np.int8)
    if n == 0:
        return pos
    if can_buy is None:
        can_buy = np.ones(n, dtype=bool)

    NEG = -1e18
    hold = np.empty(n)  # best net profit ending bar t HOLDING a position
    cash = np.empty(n)  # best net profit ending bar t FLAT
    buy_at = np.zeros(n, dtype=bool)  # hold[t] achieved by buying at t (from cash[t-1])
    sell_at = np.zeros(n, dtype=bool)  # cash[t] achieved by selling at t (from hold[t-1])

    hold[0] = -logp[0] if can_buy[0] else NEG  # can only hold at bar 0 by buying at bar 0
    cash[0] = 0.0
    buy_at[0] = can_buy[0]
    for t in range(1, n):
        buy = (cash[t - 1] - logp[t]) if can_buy[t] else NEG
        if buy >= hold[t - 1]:
            hold[t] = buy
            buy_at[t] = True
        else:
            hold[t] = hold[t - 1]
        sell = hold[t - 1] + logp[t] - fee_log
        if sell > cash[t - 1]:
            cash[t] = sell
            sell_at[t] = True
        else:
            cash[t] = cash[t - 1]

    # Backtrack from the end. The optimal terminal state is FLAT (a held position at the
    # very end would only be opened if it could be sold profitably, which the DP already
    # accounts for via cash).
    state_hold = cash[n - 1] < hold[n - 1]
    for t in range(n - 1, -1, -1):
        if state_hold:
            pos[t] = 1
            if buy_at[t]:
                state_hold = False  # bought at t -> flat at end of t-1
        else:
            pos[t] = 0
            if sell_at[t]:
                state_hold = True  # sold at t -> held through t-1
    return pos


def _maxprofit_labels(
    close: np.ndarray, fee: float, entry_min_ret_120: float | None = None
) -> np.ndarray:
    """Per-bar {OUT, ENTER, HOLD, EXIT} from the optimal long-only policy (float array)."""
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return labels
    valid = close > 0
    if not valid.any():
        return labels
    logp = np.log(np.where(valid, close, 1.0))
    fee_log = float(np.log(1.0 + fee))

    can_buy = None
    if entry_min_ret_120 is not None:
        # Causal regime gate: only allow OPENING a position where the trailing-120-bar
        # return exceeds the threshold (no entering counter-trend bounces in a downtrend).
        can_buy = np.zeros(n, dtype=bool)
        for t in range(120, n):
            if close[t - 120] > 0 and close[t] / close[t - 120] - 1.0 > entry_min_ret_120:
                can_buy[t] = True

    pos = _maxprofit_positions(logp, fee_log, can_buy)

    prev = np.empty(n, dtype=np.int8)
    prev[0] = 0
    prev[1:] = pos[:-1]
    labels[(pos == 1) & (prev == 0)] = ENTER
    labels[(pos == 1) & (prev == 1)] = HOLD
    labels[(pos == 0) & (prev == 1)] = EXIT
    labels[(pos == 0) & (prev == 0)] = OUT
    # Bars on invalid (non-positive) close get no actionable label.
    labels[~valid] = np.nan
    return labels
