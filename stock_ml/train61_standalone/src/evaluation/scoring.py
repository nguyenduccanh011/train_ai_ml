"""
Unified scoring - single source of truth for model ranking.

Design principles:
  - Quality-first: rank by edge + risk + stability, not raw turnover alone
  - Live-aware: include confidence shrinkage and compressed total-PnL scale
  - Risk-adjusted: penalize per-symbol drawdown, not aggregate portfolio MDD
  - Stability: penalize models that only shine in 1-2 years

Score components (weights sum to 1.0):
  + sharpe          0.30  avg_pnl / std(pnl) - return per unit of risk
  + avg_pnl         0.25  per-trade expectancy (symbol-count neutral)
  + profit_factor   0.22  gross profit / gross loss
  - mdd_per_symbol  0.15  avg max-drawdown per symbol - per-position risk
  - yr_consistency  0.08  CV of per-year PnL - penalize lucky-year models
  + total_pnl_scale 0.10  compressed bonus from log1p(total_pnl) in live mode

Excluded (with reason):
  - raw total_pnl   - can bias to larger universes; only compressed scale is used
  - win_rate        - already captured by PF = WR/(1-WR) * R/R
  - trade_count     - used as confidence weight, not standalone additive score
  - expectancy      - mathematically equal to avg_pnl
"""

from collections import defaultdict

import numpy as np

from src.config_loader import load_config


def _get_weights():
    cfg = load_config()
    w = cfg.get("scoring", {}).get("weights", {})
    return {
        "sharpe": w.get("sharpe", 0.30),
        "avg_pnl": w.get("avg_pnl", 0.25),
        "profit_factor": w.get("profit_factor", 0.22),
        "mdd_per_symbol": w.get("mdd_per_symbol", 0.15),
        "yr_consistency": w.get("yr_consistency", 0.08),
        # Live-oriented extension: compressed scale term to reward durable throughput
        # without letting raw total_pnl dominate cross-universe comparison.
        "total_pnl_scale": w.get("total_pnl_scale", 0.10),
    }


def _get_scoring_params():
    cfg = load_config()
    s = cfg.get("scoring", {})
    return {
        "mode": s.get("mode", "live"),  # legacy | live
        "confidence_k": float(s.get("confidence_k", 120.0)),
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

    if not any(len(yr_pnl) >= 2 for yr_pnl in sym_yr.values()):
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


def calc_symbol_coverage(trades: list) -> dict:
    if not trades:
        return {"symbol_count": 0, "top_symbol_pnl_ratio": 0.0}

    pnl_by_symbol = defaultdict(float)
    for trade in trades:
        pnl_by_symbol[trade.get("symbol", "_")] += trade.get("pnl_pct", 0.0)

    total_abs_pnl = sum(abs(pnl) for pnl in pnl_by_symbol.values())
    top_abs_pnl = max((abs(pnl) for pnl in pnl_by_symbol.values()), default=0.0)
    ratio = top_abs_pnl / total_abs_pnl if total_abs_pnl > 0 else 0.0
    return {
        "symbol_count": len(pnl_by_symbol),
        "top_symbol_pnl_ratio": round(float(ratio), 4),
    }


# ─── Main scoring ─────────────────────────────────────────────────────────────


def composite_score(metrics: dict, trades: list | None = None) -> float:
    if metrics.get("trades", 0) == 0:
        return 0.0

    avg_pnl = metrics.get("avg_pnl", 0)
    pf = metrics.get("pf", 0)
    avg_hold = float(metrics.get("avg_hold", 0.0) or 0.0)

    if trades is not None:
        sharpe = calc_sharpe(trades)
        mdd_sym = calc_mdd_per_symbol(trades)
        yr_cv = calc_yearly_consistency(trades)
    else:
        sharpe = avg_pnl
        mdd_sym = abs(metrics.get("max_loss", 0))
        yr_cv = 0.0

    norm_sharpe = float(np.tanh(sharpe / 0.55))
    norm_avg = float(np.tanh(avg_pnl / 12.0))
    norm_pf = float(1.0 - np.exp(-max(pf - 1.0, 0.0) / 9.0))
    norm_mdd = float(1.0 - np.exp(-max(mdd_sym, 0.0) / 35.0))
    norm_yr = float(max(yr_cv - 0.35, 0.0) / 2.0)
    norm_hold = float(max(avg_hold - 50.0, 0.0) / 25.0)

    quality_score = (
        0.18 * norm_sharpe
        + 0.28 * norm_avg
        + 0.26 * norm_pf
        - 0.12 * np.clip(norm_mdd, 0, 1)
        - 0.05 * np.clip(norm_yr, 0, 1)
        - 0.01 * np.clip(norm_hold, 0, 1)
    ) * 1000

    n_trades = max(int(metrics.get("trades", 0)), 0)
    confidence = min(1.0, float(np.sqrt(n_trades / 1000.0))) if n_trades > 0 else 0.0

    total_pnl = float(metrics.get("total_pnl", 0.0))
    pnl_scale = float(np.tanh(max(total_pnl, 0.0) / 18000.0))
    return round(quality_score * confidence + 0.18 * pnl_scale * 1000, 1)


# ─── Metrics aggregator ───────────────────────────────────────────────────────


def calc_metrics(trades):
    """Compute standard metrics from a trade list.  Shared by every caller."""
    if not trades:
        return {
            "trades": 0,
            "wr": 0.0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "pf": 0.0,
            "max_loss": 0.0,
            "avg_hold": 0.0,
        }
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
    return {
        "trades": n,
        "wr": round(wr, 2),
        "avg_pnl": round(avg_pnl, 3),
        "total_pnl": round(total_pnl, 2),
        "pf": round(pf, 3),
        "max_loss": round(max_loss, 2),
        "avg_hold": round(avg_hold, 1),
    }
