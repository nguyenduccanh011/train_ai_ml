"""Operator registry for the feature DSL.

Each operator declares an ``axis`` that drives evaluation grouping:

* ``elementwise`` — row-aligned, no grouping.
* ``symbol``      — time-series, grouped by symbol (per-symbol history).
* ``date``        — cross-sectional, grouped by date (whole universe per day).

Complex indicators (RSI/ADX/ATR/MACD/MFI/Bollinger) are *built-in* ops, ported
verbatim from ``leading_v2`` so the parity gate (Phase 5) can match the old
builder to float tolerance. They are NOT decomposed into pure DSL because Wilder
smoothing + nested ADX are error-prone to express.

An op implementation has signature ``fn(ctx, args, kwargs) -> Series | dict``
where ``args`` are already-evaluated (a ``pd.Series`` for sub-expressions, a
Python ``float`` for numeric literals) and multi-output ops return a
``dict[str, pd.Series]`` selected via an ``.attr`` accessor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpSpec:
    name: str
    axis: str  # 'elementwise' | 'symbol' | 'date'
    fn: Callable
    outputs: tuple[str, ...] | None = None  # None = single Series; else multi-output


OPS: dict[str, OpSpec] = {}


def _register(name: str, axis: str, outputs: tuple[str, ...] | None = None):
    def deco(fn: Callable) -> Callable:
        OPS[name] = OpSpec(name=name, axis=axis, fn=fn, outputs=outputs)
        return fn

    return deco


def get_op(name: str) -> OpSpec | None:
    return OPS.get(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_scalar(v) -> bool:
    return not isinstance(v, pd.Series)


def _as_series(ctx, v) -> pd.Series:
    """Coerce a scalar to a constant Series aligned to ctx.index."""
    if isinstance(v, pd.Series):
        return v
    return pd.Series(v, index=ctx.index, dtype="float64")


def _roll(ctx, s: pd.Series, n: int, agg: str) -> pd.Series:
    return s.groupby(ctx.symbol, sort=False).transform(
        lambda x: getattr(x.rolling(n, min_periods=n), agg)()
    )


def _by_symbol(ctx, frames: dict[str, pd.Series], fn: Callable):
    """Apply ``fn`` to each per-symbol sub-DataFrame; return aligned result.

    Iterates groups explicitly (instead of ``groupby.apply``) so a single-symbol
    universe doesn't trip pandas' Series→DataFrame coercion and get mistaken for a
    multi-output op.
    """
    base = pd.DataFrame(frames)
    single_parts: list[pd.Series] = []
    multi_parts: list[pd.DataFrame] = []
    for _key, positions in base.groupby(ctx.symbol, sort=False).indices.items():
        sub = base.iloc[positions]
        out = fn(sub)
        if isinstance(out, pd.DataFrame):
            multi_parts.append(out)
        else:
            single_parts.append(out)
    if multi_parts:
        combined = pd.concat(multi_parts).reindex(ctx.index)
        return {col: combined[col] for col in combined.columns}
    return pd.concat(single_parts).reindex(ctx.index)


# ---------------------------------------------------------------------------
# Elementwise
# ---------------------------------------------------------------------------


@_register("Abs", "elementwise")
def _op_abs(ctx, args, kwargs):
    return _as_series(ctx, args[0]).abs()


@_register("Sign", "elementwise")
def _op_sign(ctx, args, kwargs):
    return np.sign(_as_series(ctx, args[0]))


@_register("Log", "elementwise")
def _op_log(ctx, args, kwargs):
    return np.log(_as_series(ctx, args[0]))


@_register("Fillna", "elementwise")
def _op_fillna(ctx, args, kwargs):
    # Replace NaN with a constant — the neutral fill for ops that are undefined on
    # degenerate windows (e.g. Corr over a constant/dead stretch, or a NaN-guarded
    # input like $volume/($volume>0)). The pipeline's fail-loud no-interior-NaN
    # policy requires total functions; 0 (or the given constant) = "no information".
    return _as_series(ctx, args[0]).fillna(float(args[1]))


@_register("Clip", "elementwise")
def _op_clip(ctx, args, kwargs):
    s = _as_series(ctx, args[0])
    lo = args[1] if len(args) > 1 else None
    hi = args[2] if len(args) > 2 else None
    return s.clip(lower=lo, upper=hi)


@_register("Max", "elementwise")
def _op_max(ctx, args, kwargs):
    # Rolling Max(series, window) when 2nd arg is a scalar window; else elementwise.
    if len(args) == 2 and _is_scalar(args[1]):
        return _roll(ctx, _as_series(ctx, args[0]), int(args[1]), "max")
    cols = pd.concat([_as_series(ctx, a) for a in args], axis=1)
    return cols.max(axis=1)


@_register("Min", "elementwise")
def _op_min(ctx, args, kwargs):
    if len(args) == 2 and _is_scalar(args[1]):
        return _roll(ctx, _as_series(ctx, args[0]), int(args[1]), "min")
    cols = pd.concat([_as_series(ctx, a) for a in args], axis=1)
    return cols.min(axis=1)


# ---------------------------------------------------------------------------
# Time-series (per-symbol)
# ---------------------------------------------------------------------------


@_register("Ref", "symbol")
def _op_ref(ctx, args, kwargs):
    return args[0].groupby(ctx.symbol, sort=False).shift(int(args[1]))


@_register("Mean", "symbol")
def _op_mean(ctx, args, kwargs):
    return _roll(ctx, args[0], int(args[1]), "mean")


@_register("Std", "symbol")
def _op_std(ctx, args, kwargs):
    return _roll(ctx, args[0], int(args[1]), "std")


@_register("Sum", "symbol")
def _op_sum(ctx, args, kwargs):
    return _roll(ctx, args[0], int(args[1]), "sum")


@_register("Delta", "symbol")
def _op_delta(ctx, args, kwargs):
    return args[0].groupby(ctx.symbol, sort=False).diff(int(args[1]))


@_register("Pct", "symbol")
def _op_pct(ctx, args, kwargs):
    return args[0].groupby(ctx.symbol, sort=False).pct_change(int(args[1]))


@_register("EMA", "symbol")
def _op_ema(ctx, args, kwargs):
    n = int(args[1])
    return (
        args[0]
        .groupby(ctx.symbol, sort=False)
        .transform(lambda x: x.ewm(span=n, adjust=False).mean())
    )


@_register("Quantile", "symbol")
def _op_quantile(ctx, args, kwargs):
    n = int(args[1])
    q = float(args[2])
    return (
        args[0]
        .groupby(ctx.symbol, sort=False)
        .transform(lambda x: x.rolling(n, min_periods=n).quantile(q))
    )


@_register("TsRank", "symbol")
def _op_tsrank(ctx, args, kwargs):
    n = int(args[1])
    return (
        args[0]
        .groupby(ctx.symbol, sort=False)
        .transform(lambda x: x.rolling(n, min_periods=n).rank())
    )


@_register("HeikenAshi", "symbol", outputs=("color", "body", "trend"))
def _op_heiken_ashi(ctx, args, kwargs):
    """Heiken-Ashi candle state (causal). HA_close=(O+H+L+C)/4; HA_open is the
    recursive average of the prior HA_open/HA_close (smoothed reversal candles).

    Outputs:
      color: +1 green / -1 red / 0 doji  (sign of HA_close - HA_open)
      body : signed body size (HA_close - HA_open) / HA_open
      trend: signed run-length of consecutive same-color HA candles (momentum of
             the HA recoloring — a flip toward 0 flags a reversal).
    Only past bars feed each value (HA_open[i] depends on i-1), so it is leakage-safe.
    """

    def _fn(g: pd.DataFrame) -> pd.DataFrame:
        o = g["open"].to_numpy(dtype=float)
        h = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        c = g["close"].to_numpy(dtype=float)
        nbar = len(c)
        ha_close = (o + h + low + c) / 4.0
        ha_open = np.empty(nbar, dtype=float)
        if nbar > 0:
            ha_open[0] = (o[0] + c[0]) / 2.0
            for i in range(1, nbar):
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        color = np.sign(ha_close - ha_open)
        body = np.where(ha_open != 0.0, (ha_close - ha_open) / ha_open, 0.0)
        trend = np.zeros(nbar, dtype=float)
        for i in range(nbar):
            if i > 0 and color[i] != 0.0 and color[i] == color[i - 1]:
                trend[i] = trend[i - 1] + color[i]
            else:
                trend[i] = color[i]
        return pd.DataFrame({"color": color, "body": body, "trend": trend}, index=g.index)

    return _by_symbol(
        ctx, {"open": args[0], "high": args[1], "low": args[2], "close": args[3]}, _fn
    )


@_register("Corr", "symbol")
def _op_corr(ctx, args, kwargs):
    n = int(args[2])

    def _fn(g: pd.DataFrame) -> pd.Series:
        return g["a"].rolling(n, min_periods=n).corr(g["b"])

    return _by_symbol(ctx, {"a": _as_series(ctx, args[0]), "b": _as_series(ctx, args[1])}, _fn)


# ---------------------------------------------------------------------------
# Indicators (per-symbol, ported from leading_v2)
# ---------------------------------------------------------------------------


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return _rsi_boundary_fill(rsi, avg_gain, avg_loss)


def _rsi_boundary_fill(rsi: pd.Series, avg_gain: pd.Series, avg_loss: pd.Series) -> pd.Series:
    """Fill the two RSI edge cases by their TRUE value, not a blanket 50.

    ``avg_loss == 0`` (past warmup, only up-moves) is maximal strength -> RSI 100, and the
    symmetric ``avg_gain == 0`` (only down-moves) -> RSI 0. The old ``fillna(50.0)`` collapsed
    both extremes to neutral, telling the model an all-gains run is 'balanced' — the opposite
    of the truth. Genuine warmup NaN (before ``min_periods``) is left as NaN so feature-warmup
    trimming removes it rather than injecting a fabricated 50.
    """
    warm = avg_loss.isna() | avg_gain.isna()  # true warmup -> stays NaN
    rsi = rsi.where(~((avg_loss == 0) & ~warm), 100.0)  # only up-moves -> overbought 100
    rsi = rsi.where(~((avg_gain == 0) & ~warm), 0.0)  # only down-moves -> oversold 0
    return rsi


@_register("RSI", "symbol")
def _op_rsi(ctx, args, kwargs):
    period = int(args[1])
    return _by_symbol(ctx, {"close": args[0]}, lambda g: _rsi(g["close"], period))


def _rsi_sma(close: pd.Series, period: int) -> pd.Series:
    """Cutler's RSI — same formula as RSI but with a *simple* moving average of
    gains/losses instead of Wilder's smoothing. Fully causal (trailing rolling mean,
    no future data). More reactive than Wilder's, so it reaches oversold/overbought
    extremes more often. Provided as a distinct, leakage-safe variant.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return _rsi_boundary_fill(rsi, avg_gain, avg_loss)


@_register("RSISMA", "symbol")
def _op_rsi_sma(ctx, args, kwargs):
    period = int(args[1])
    return _by_symbol(ctx, {"close": args[0]}, lambda g: _rsi_sma(g["close"], period))


def _atr(high, low, close, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


@_register("ATR", "symbol")
def _op_atr(ctx, args, kwargs):
    period = int(args[3])
    return _by_symbol(
        ctx,
        {"high": args[0], "low": args[1], "close": args[2]},
        lambda g: _atr(g["high"], g["low"], g["close"], period),
    )


@_register("ADX", "symbol", outputs=("adx", "plus_di", "minus_di"))
def _op_adx(ctx, args, kwargs):
    period = int(args[3])

    def _fn(g: pd.DataFrame) -> pd.DataFrame:
        high, low, close = g["high"], g["low"], g["close"]
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
        )
        tr = pd.concat(
            [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        plus_di = (
            100
            * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
            / (atr + 1e-8)
        )
        minus_di = (
            100
            * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
            / (atr + 1e-8)
        )
        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di
        di_ratio = di_diff / (di_sum.replace(0.0, np.nan) + 1e-8)
        adx = di_ratio.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() * 100
        return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})

    return _by_symbol(ctx, {"high": args[0], "low": args[1], "close": args[2]}, _fn)


@_register("MACD", "symbol", outputs=("line", "signal", "hist"))
def _op_macd(ctx, args, kwargs):
    fast = int(args[1]) if len(args) > 1 else 12
    slow = int(args[2]) if len(args) > 2 else 26
    signal = int(args[3]) if len(args) > 3 else 9

    def _fn(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"]
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        line = ema_fast - ema_slow
        sig = line.ewm(span=signal, adjust=False).mean()
        hist = line - sig
        return pd.DataFrame({"line": line, "signal": sig, "hist": hist})

    return _by_symbol(ctx, {"close": args[0]}, _fn)


@_register("OBV", "symbol")
def _op_obv(ctx, args, kwargs):
    def _fn(g: pd.DataFrame) -> pd.Series:
        return (np.sign(g["close"].diff()) * g["volume"]).fillna(0).cumsum()

    return _by_symbol(ctx, {"close": args[0], "volume": args[1]}, _fn)


@_register("MFI", "symbol")
def _op_mfi(ctx, args, kwargs):
    period = int(args[4])

    def _fn(g: pd.DataFrame) -> pd.Series:
        high, low, close, volume = g["high"], g["low"], g["close"], g["volume"]
        tp = (high + low + close) / 3.0
        mf = tp * volume
        pos_mf = mf.where(tp.diff() > 0, 0.0)
        neg_mf = mf.where(tp.diff() < 0, 0.0)
        pos_sum = pos_mf.rolling(period, min_periods=period).sum()
        neg_sum = neg_mf.rolling(period, min_periods=period).sum()
        mfi = 100.0 - (100.0 / (1.0 + pos_sum / (neg_sum.replace(0.0, np.nan) + 1e-8)))
        # neg_sum == 0 past warmup (all money-flow inbound) is overbought -> MFI 100, and the
        # symmetric pos_sum == 0 -> MFI 0. Blanket fillna(50) would wrongly call an all-inflow
        # window 'neutral'. Genuine warmup NaN stays NaN for warmup-trim to remove.
        warm = pos_sum.isna() | neg_sum.isna()
        mfi = mfi.where(~((neg_sum == 0) & ~warm), 100.0)
        mfi = mfi.where(~((pos_sum == 0) & ~warm), 0.0)
        return mfi

    return _by_symbol(
        ctx,
        {"high": args[0], "low": args[1], "close": args[2], "volume": args[3]},
        _fn,
    )


@_register("Bollinger", "symbol", outputs=("mid", "upper", "lower", "width", "pct"))
def _op_bollinger(ctx, args, kwargs):
    n = int(args[1])
    num_std = float(args[2]) if len(args) > 2 else 2.0

    def _fn(g: pd.DataFrame) -> pd.DataFrame:
        close = g["close"]
        mid = close.rolling(n, min_periods=n).mean()
        std = close.rolling(n, min_periods=n).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        rng = upper - lower
        width = rng / mid.replace(0.0, np.nan)
        # Flat window (std==0 -> band width 0): %B is undefined by the raw formula.
        # By TA convention a zero-width band means price sits ON the mid line -> %B = 0.5
        # (neutral). Emitting NaN here instead breaks the fail-loud no-NaN guard for
        # thinly-traded names with >=20 identical closes (penny/price-locked), while
        # 0.5 is the correct neutral reading. Only fires when rng==0 (band collapsed).
        rng_nz = rng.where(rng != 0.0)
        pct = ((close - lower) / rng_nz).where(rng != 0.0, 0.5)
        return pd.DataFrame(
            {"mid": mid, "upper": upper, "lower": lower, "width": width, "pct": pct}
        )

    return _by_symbol(ctx, {"close": args[0]}, _fn)


@_register("ROC", "symbol")
def _op_roc(ctx, args, kwargs):
    n = int(args[1])
    return (
        args[0]
        .groupby(ctx.symbol, sort=False)
        .transform(lambda x: x / x.shift(n).replace(0.0, np.nan) - 1.0)
    )


@_register("Aroon", "symbol", outputs=("up", "down"))
def _op_aroon(ctx, args, kwargs):
    """Aroon(high, low, period). Causal, per-symbol.

    Over a trailing window of ``period`` bars, ``up`` measures how recently the
    highest high occurred and ``down`` how recently the lowest low occurred:
    ``up = argmax_pos / (period - 1) * 100`` (oldest=0 .. newest=period-1), so a
    fresh high → 100, a stale high → →0. Only past/current bars feed the window
    (rolling, min_periods=period), so it is leakage-safe.
    """
    period = int(args[2])
    denom = float(period - 1) if period > 1 else 1.0

    def _fn(g: pd.DataFrame) -> pd.DataFrame:
        high, low = g["high"], g["low"]
        up = high.rolling(period, min_periods=period).apply(np.argmax, raw=True) / denom * 100.0
        down = low.rolling(period, min_periods=period).apply(np.argmin, raw=True) / denom * 100.0
        return pd.DataFrame({"up": up, "down": down}, index=g.index)

    return _by_symbol(ctx, {"high": args[0], "low": args[1]}, _fn)


# ---------------------------------------------------------------------------
# Cross-sectional (per-date). Leakage-safe: NO bfill of warmup NaNs.
# ---------------------------------------------------------------------------


@_register("CSRank", "date")
def _op_csrank(ctx, args, kwargs):
    return args[0].groupby(ctx.date, sort=False).rank(pct=True)


@_register("CSZScore", "date")
def _op_cszscore(ctx, args, kwargs):
    g = args[0].groupby(ctx.date, sort=False)
    mean = g.transform("mean")
    std = g.transform("std")
    return (args[0] - mean) / std.replace(0.0, np.nan)


@_register("CSMean", "date")
def _op_csmean(ctx, args, kwargs):
    return args[0].groupby(ctx.date, sort=False).transform("mean")


@_register("CSStd", "date")
def _op_csstd(ctx, args, kwargs):
    return args[0].groupby(ctx.date, sort=False).transform("std")


@_register("CSGroupMedian", "date")
def _op_csgroupmedian(ctx, args, kwargs):
    by = kwargs.get("by")
    if by is None:
        raise ValueError("CSGroupMedian requires keyword arg by=<field>")
    grouper = [ctx.date, _as_series(ctx, by)]
    return args[0].groupby(grouper, sort=False).transform("median")


@_register("CSGroupRank", "date")
def _op_csgrouprank(ctx, args, kwargs):
    by = kwargs.get("by")
    if by is None:
        raise ValueError("CSGroupRank requires keyword arg by=<field>")
    grouper = [ctx.date, _as_series(ctx, by)]
    return args[0].groupby(grouper, sort=False).rank(pct=True)


@_register("CSGroupZScore", "date")
def _op_csgroupzscore(ctx, args, kwargs):
    by = kwargs.get("by")
    if by is None:
        raise ValueError("CSGroupZScore requires keyword arg by=<field>")
    grouper = [ctx.date, _as_series(ctx, by)]
    g = args[0].groupby(grouper, sort=False)
    mean = g.transform("mean")
    std = g.transform("std")
    return (args[0] - mean) / std.replace(0.0, np.nan)


@_register(
    "ZigZag",
    "symbol",
    outputs=(
        "last_dir",
        "last_leg_return",
        "last_leg_dur",
        "prev_leg_return",
        "prev_leg_dur",
        "bars_since_pivot",
        "return_since_pivot",
        "progress_to_deviation",
        "dist_to_confirm",
        "price_pos_in_swing",
        "max_adverse_since_pivot",
        "n_pivots",
    ),
)
def _op_zigzag(ctx, args, kwargs):
    """Causal zigzag swing-state features (NO lookahead).

    ``ZigZag($close, pct, min_leg_bars)``. Unlike ``ZigzagPivotTarget`` (which
    labels bars by proximity to a pivot known only with hindsight), every output
    here uses only pivots **confirmed as of bar t** — a pivot at its extreme bar
    becomes visible only on the later bar where price reversed by ``pct``. So at t
    these describe the last confirmed swing and the in-progress leg, exactly what a
    live system would know. Leakage-safe as model inputs (see prefix-invariance test).

    ``min_leg_bars`` drops swings shorter than that many bars (noise filter), 0 off.

    Confirmed-structure outputs (NaN until enough pivots exist):
      last_dir            direction of the last confirmed leg (+1 up→peak, -1 down→bottom)
      last_leg_return     signed return of the last confirmed leg
      last_leg_dur        bars spanned by the last confirmed leg
      prev_leg_return     signed return of the leg before that
      prev_leg_dur        bars spanned by the leg before that
      bars_since_pivot    bars since the last confirmed pivot's extreme
      return_since_pivot  close/last_pivot_price - 1 (move since last pivot)
      price_pos_in_swing  position of close within the last confirmed swing range [0,1]
      n_pivots            confirmed-pivot count so far, capped at 3 (maturity flag; never NaN)
    Developing-leg outputs (defined once a leg direction is set):
      progress_to_deviation  deviation from the running extreme / pct; →1 = a pivot
                             is about to confirm (the leg is rolling over)
      dist_to_confirm        1 - progress_to_deviation
      max_adverse_since_pivot deepest pullback from the running extreme seen this leg
    """
    pct = float(args[1])
    min_leg = int(args[2]) if len(args) > 2 else 0

    cols = (
        "last_dir",
        "last_leg_return",
        "last_leg_dur",
        "prev_leg_return",
        "prev_leg_dur",
        "bars_since_pivot",
        "return_since_pivot",
        "progress_to_deviation",
        "dist_to_confirm",
        "price_pos_in_swing",
        "max_adverse_since_pivot",
        "n_pivots",
    )

    def _fn(g: pd.DataFrame) -> pd.DataFrame:
        c = g["close"].to_numpy(dtype=np.float64)
        n = len(c)
        out = {k: np.full(n, np.nan, dtype=np.float64) for k in cols}
        out["n_pivots"] = np.zeros(n, dtype=np.float64)  # maturity flag, never NaN
        if n == 0:
            return pd.DataFrame(out, index=g.index)

        direction = 0  # +1 up-leg (tracking high), -1 down-leg (tracking low)
        ext_idx, ext_price = 0, c[0]
        leg_max_dev = 0.0  # deepest deviation against the leg since the last pivot
        last_confirmed_idx = 0
        pivots: list[tuple[int, float, int]] = []  # (idx, price, type:+1 peak/-1 bottom)

        for i in range(1, n):
            price = c[i]
            if direction >= 0 and price > ext_price:
                ext_price, ext_idx, direction = price, i, 1
            elif direction <= 0 and price < ext_price:
                ext_price, ext_idx, direction = price, i, -1
            elif direction == 1 and price <= ext_price * (1.0 - pct):
                if ext_idx - last_confirmed_idx >= min_leg:
                    pivots.append((ext_idx, ext_price, +1))  # confirm peak
                    last_confirmed_idx = ext_idx
                direction, ext_price, ext_idx, leg_max_dev = -1, price, i, 0.0
            elif direction == -1 and price >= ext_price * (1.0 + pct):
                if ext_idx - last_confirmed_idx >= min_leg:
                    pivots.append((ext_idx, ext_price, -1))  # confirm bottom
                    last_confirmed_idx = ext_idx
                direction, ext_price, ext_idx, leg_max_dev = 1, price, i, 0.0

            # developing-leg deviation from the running extreme
            if direction == 1:
                dev = (ext_price - price) / ext_price
            elif direction == -1:
                dev = (price - ext_price) / ext_price
            else:
                dev = np.nan
            if dev == dev:  # not NaN
                leg_max_dev = max(leg_max_dev, dev)
                out["progress_to_deviation"][i] = dev / pct
                out["dist_to_confirm"][i] = 1.0 - dev / pct
                out["max_adverse_since_pivot"][i] = leg_max_dev

            out["n_pivots"][i] = min(len(pivots), 3)

            if pivots:
                lp_idx, lp_price, lp_type = pivots[-1]
                out["last_dir"][i] = lp_type
                out["bars_since_pivot"][i] = i - lp_idx
                out["return_since_pivot"][i] = price / lp_price - 1.0
                if len(pivots) >= 2:
                    pp_idx, pp_price, _ = pivots[-2]
                    out["last_leg_return"][i] = lp_price / pp_price - 1.0
                    out["last_leg_dur"][i] = lp_idx - pp_idx
                    lo_, hi_ = sorted((pp_price, lp_price))
                    # Degenerate swing (two pivots at the same price) → neutral 0.5
                    # instead of 0/0 NaN, so a flat stretch doesn't trip fail-loud.
                    out["price_pos_in_swing"][i] = (price - lo_) / (hi_ - lo_) if hi_ > lo_ else 0.5
                    if len(pivots) >= 3:
                        ppp_idx, ppp_price, _ = pivots[-3]
                        out["prev_leg_return"][i] = pp_price / ppp_price - 1.0
                        out["prev_leg_dur"][i] = pp_idx - ppp_idx

        return pd.DataFrame(out, index=g.index)

    return _by_symbol(ctx, {"close": args[0]}, _fn)
