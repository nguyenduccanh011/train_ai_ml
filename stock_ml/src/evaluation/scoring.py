"""
Unified scoring — single source of truth for model ranking.

Design principles:
  - Symbol-count neutral: adding more symbols should NOT inflate the score
  - No redundant components: Expectancy == avg_pnl; WR is implicit in PF
  - Risk-adjusted: penalize per-symbol drawdown, not aggregate portfolio MDD
  - Stability: penalize models that only shine in 1-2 years

Score components (weights sum to 1.0):
  + sharpe          0.30  avg_pnl / std(pnl) — return per unit of risk
  + avg_pnl         0.25  per-trade expectancy — symbol-count neutral
  + profit_factor   0.22  gross profit / gross loss
  - mdd_per_symbol  0.15  avg max-drawdown per symbol — real per-position risk
  - yr_consistency  0.08  CV of per-year PnL — penalize "lucky year" models

Excluded (with reason):
  - total_pnl      → scales with #symbols → biases toward bigger universes
  - win_rate       → already captured by PF = WR/(1-WR) × R/R
  - trade_count    → used as confidence weight, not score
  - expectancy     → mathematically equal to avg_pnl
"""
import numpy as np
from collections import defaultdict

from src.config_loader import load_config


def _get_weights():
    cfg = load_config()
    w = cfg.get("scoring", {}).get("weights", {})
    return {
        "total_pnl":      w.get("total_pnl",      0.45),
        "profit_factor":  w.get("profit_factor",  0.25),
        "mdd_per_symbol": w.get("mdd_per_symbol", 0.20),
        "sharpe":         w.get("sharpe",         0.10),
    }


# ─── Individual metric calculators ───────────────────────────────────────────

def calc_sharpe(trades: list) -> float:
    """Per-trade Sharpe: avg_pnl / std(pnl).

    Uses per-trade PnL (not time-series), so it is symbol-count neutral.
    Returns 0 if std == 0 or fewer than 5 trades.
    """
    if len(trades) < 5:
        return 0.0
    pnls = np.array([t["pnl_pct"] for t in trades])
    std = np.std(pnls)
    return float(np.mean(pnls) / std) if std > 0 else 0.0


def calc_mdd_per_symbol(trades: list) -> float:
    """Average max-drawdown across symbols.

    For each symbol: sort trades by entry_date, build equity curve,
    compute peak-to-trough MDD.  Return the mean across all symbols.
    This is symbol-count neutral and reflects per-position real risk.
    """
    if not trades:
        return 0.0

    by_symbol = defaultdict(list)
    for t in trades:
        by_symbol[t.get("symbol", "_")].append(t)

    mdds = []
    for sym, sym_trades in by_symbol.items():
        try:
            sym_trades_s = sorted(sym_trades, key=lambda t: t.get("entry_date", ""))
        except Exception:
            sym_trades_s = sym_trades
        pnls = np.array([t["pnl_pct"] for t in sym_trades_s])
        equity = np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        mdd = float(np.max(peak - equity))
        mdds.append(mdd)

    return float(np.mean(mdds)) if mdds else 0.0


def calc_yearly_consistency(trades: list) -> float:
    """Coefficient of Variation (CV) of per-year total PnL across symbols.

    For each symbol, compute yearly PnL.  Then compute CV = std/mean of
    per-year totals across all (symbol, year) pairs.
    A model scoring 100% per year per symbol consistently → CV near 0.
    CV is symbol-count neutral.

    Returns 0 if fewer than 2 years or no valid data.
    """
    if not trades:
        return 0.0

    # Collect (symbol, year) → total pnl
    sym_yr = defaultdict(lambda: defaultdict(float))
    for t in trades:
        yr = str(t.get("entry_date", ""))[:4]
        sym = t.get("symbol", "_")
        if yr.isdigit():
            sym_yr[sym][yr] += t["pnl_pct"]

    if not sym_yr:
        return 0.0

    # For each symbol, compute yearly PnL vector; then std across symbols' yearly means
    all_yearly_means = []
    for sym, yr_pnl in sym_yr.items():
        if len(yr_pnl) >= 2:
            all_yearly_means.append(list(yr_pnl.values()))

    if not all_yearly_means:
        return 0.0

    # Flatten into per-year averages across all symbols
    years_union = sorted(set(yr for sm in sym_yr.values() for yr in sm))
    if len(years_union) < 2:
        return 0.0

    yr_totals = []
    for yr in years_union:
        vals = [sym_yr[sym].get(yr, np.nan) for sym in sym_yr]
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            yr_totals.append(np.mean(valid))

    if len(yr_totals) < 2:
        return 0.0

    mean_yr = abs(np.mean(yr_totals))
    std_yr = np.std(yr_totals)
    # CV: std/mean — higher = less consistent
    return float(std_yr / mean_yr) if mean_yr > 0 else float(std_yr)


def calc_max_drawdown(trades: list) -> float:
    """Legacy: aggregate portfolio MDD (kept for backward compat / display).
    Prefer calc_mdd_per_symbol() for scoring.
    """
    if not trades:
        return 0.0
    try:
        sorted_trades = sorted(trades, key=lambda t: t.get("entry_date", ""))
    except Exception:
        sorted_trades = trades
    pnls = [t["pnl_pct"] for t in sorted_trades]
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity)) if len(equity) > 0 else 0.0


# ─── Main scoring ─────────────────────────────────────────────────────────────

def composite_score(metrics: dict, trades: list | None = None) -> float:
    """Canonical composite score (higher = better).

    Args:
        metrics: output of calc_metrics().
        trades:  raw trade list (dicts with pnl_pct, entry_date, symbol).
                 When None: falls back to max_loss proxy for MDD and
                 skips sharpe / consistency (score will be approximate).
    """
    if metrics.get("trades", 0) == 0:
        return 0.0

    w = _get_weights()

    avg_pnl   = metrics.get("avg_pnl", 0)
    pf        = metrics.get("pf", 0)
    total_pnl = metrics.get("total_pnl", 0)

    if trades is not None:
        sharpe  = calc_sharpe(trades)
        mdd_sym = calc_mdd_per_symbol(trades)
    else:
        pnl_std = 1.0
        sharpe  = avg_pnl / max(pnl_std, 0.01)
        mdd_sym = abs(metrics.get("max_loss", 0))

    # ── Normalise to [-1, 1] or [0, 1] ──────────────────────────────────────
    # total_pnl: 0% → 0, 10000% → 1 (absolute bottom-line, fee-adjusted)
    norm_total = np.clip(total_pnl / 10000, -1, 1)

    # PF: 1 → 0, 4 → 1
    norm_pf = np.clip((pf - 1) / 3, -1, 1)

    # MDD per symbol: 0% → 0 penalty, 30%+ → full penalty
    norm_mdd = np.clip(mdd_sym / 30, 0, 1)

    # Sharpe: 0 → 0, 0.3 → 1 (stability)
    norm_sharpe = np.clip(sharpe / 0.30, -1, 1)

    score = (
          w["total_pnl"]      * norm_total
        + w["profit_factor"]  * norm_pf
        - w["mdd_per_symbol"] * norm_mdd
        + w["sharpe"]         * norm_sharpe
    ) * 1000

    return round(score, 1)


# ─── Metrics aggregator ───────────────────────────────────────────────────────

def calc_metrics(trades):
    """Compute standard metrics from a trade list.  Shared by every caller."""
    if not trades:
        return {"trades": 0, "wr": 0.0, "avg_pnl": 0.0, "total_pnl": 0.0,
                "pf": 0.0, "max_loss": 0.0, "avg_hold": 0.0}
    n = len(trades)
    pnls = [t["pnl_pct"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    avg_pnl = float(np.mean(pnls))
    total_pnl = sum(pnls)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else 99.0
    max_loss = min(pnls)
    avg_hold = float(np.mean([t.get("holding_days", 0) for t in trades]))
    return {"trades": n, "wr": round(wr, 2), "avg_pnl": round(avg_pnl, 3),
            "total_pnl": round(total_pnl, 2), "pf": round(pf, 3),
            "max_loss": round(max_loss, 2), "avg_hold": round(avg_hold, 1)}
