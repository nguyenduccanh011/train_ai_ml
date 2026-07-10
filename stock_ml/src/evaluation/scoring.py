"""
Unified scoring - single source of truth for model ranking.

Design principles:
  - Capital-efficiency first: reward return PER BAR HELD (per-bar is the largest weight), so a
    fast move scores high and buy&hold / long-hold riders are crushed (long hold dilutes per-bar).
  - Total-PnL completeness: per-bar alone would settle for a few sharp trades; a saturating
    total_pnl term rewards EXPANDING into more (even slower) profitable trades once the fast
    phases run out. Bounded by tanh + fixed 61-sym universe → no buy&hold / volume runaway.
  - Risk-adjusted: penalize per-symbol drawdown, not aggregate portfolio MDD.
  - Stability: penalize models that only shine in 1-2 years.
  - Confidence shrinkage: scale quality by min(1, sqrt(trades/target)) so thin samples can't top
    the board; capped at 1.0 so extra trades never buy score on volume alone.

Actual formula (composite_score), weights are HARDCODED in the function (the cfg `scoring.weights`
path is currently NOT wired in):

  avg_per_bar  = avg_pnl / max(avg_hold, 1)
  norm_sharpe  = tanh(sharpe / 0.55)
  norm_avg_bar = tanh(avg_per_bar / 0.0015)
  norm_total   = tanh((total_pnl / n_symbols) / 1.475)   # PER SYMBOL → symbol-count neutral
  norm_pf      = 1 - exp(-max(pf-1, 0) / 9)
  norm_mdd     = 1 - exp(-max(mdd_per_symbol, 0) / 0.35)   # mdd_sym is fraction-scale (~0.2-0.4)
  norm_yr      = max(yr_cv - 0.35, 0) / 2
  quality = ( 0.15*norm_sharpe + 0.27*norm_avg_bar + 0.20*norm_total + 0.18*norm_pf
            - 0.15*clip(norm_mdd,0,1) - 0.07*clip(norm_yr,0,1) ) * 1000
  composite = quality * min(1, sqrt(trades/target))   # confidence = thin-sample shrinkage only

Case-1/case-2 behavior (more trades): adding EMPTY trades doesn't raise total_pnl and lowers
per-bar → score drops (prefer fewer); adding PRODUCTIVE trades raises total_pnl → score rises
(prefer more). Trade count itself is never an additive reward, only a confidence weight.

Excluded (with reason):
  - raw avg_pnl     - replaced by avg_per_bar; raw value rewards long holds (buy&hold premium)
  - win_rate        - already captured by PF = WR/(1-WR) * R/R
  - trade_count     - used as confidence weight, not standalone additive score
  - avg_hold penalty- subsumed by avg_per_bar (long holds dilute per-bar directly)
"""

from collections import defaultdict

import numpy as np

from stock_ml.src.utils.config_loader import load_config

# SCORING-V2 objective knobs (2026-06-18 user redesign — see composite_score). Tunable / git-reversible.
# MDD penalty SHAPE (2026-06-18b): replaced the free-zone (no penalty below 0.22) with a GENTLE
# CONVEX curve-to-0 — norm_mdd = clip((mdd/SCORE_MDD_DIV)**SCORE_MDD_POW, 0, 1). The free-zone made
# the score BLIND to risk improvements below 0.22 (forensic: 2409+vol-gate cut mdd 0.213->0.203 for
# zero composite credit). The old exp curve over-bit at LOW mdd (the problem the free-zone removed);
# pow1.5 is the middle path — small gradient near the operating range (won't reject profitable
# runners) yet always >0 so risk reductions are visible/rewarded. Empirically keeps the conv-pullback
# win over the no-conv line (+0.7) while making the safer vol-gate the clear champion (+4.8).
SCORE_MDD_DIV = 0.40    # mdd scale: norm_mdd reaches 1.0 (full penalty) at this per-symbol MDD
SCORE_MDD_POW = 1.5     # convexity: >1 = gentle near 0, steeper toward SCORE_MDD_DIV
SCORE_PNL_W = 0.45      # total-PnL weight (0.34->0.40->0.45) — user 2026-06-19: shift weight from the
#                         per-bar + trade-count penalties to PnL so "let winners run" (fewer/longer but
#                         equal-PnL, equal-MDD trades, e.g. oxt08) is no longer docked on throughput.
SCORE_PNL_CAP = 2.6     # linear PnL credit up to this per-symbol level (≈ non-saturating in the live range)


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


def calc_sortino(trades: list, mar: float = 0.0) -> float:
    """Per-trade Sortino: avg_pnl / DOWNSIDE deviation (only sub-MAR returns count as risk).

    Sharpe uses the symmetric std, so it penalises UPSIDE dispersion — a strategy whose edge is
    letting winners run (bigger, lumpier WINNERS, same losers) gets a worse Sharpe even though its
    real risk (drawdown, loss size) is unchanged. Forensic proof (2026-06-15): a full-wave-capture
    variant had identical downside-deviation (0.0407 vs 0.0408) yet lower Sharpe purely from bigger
    winners. Sortino measures only DOWNSIDE variability, so catching full waves is not punished.
    Returns 0 if <5 trades or no downside.
    """
    if len(trades) < 5:
        return 0.0
    pnls = np.array([t["pnl_pct"] for t in trades])
    downside = np.minimum(pnls - mar, 0.0)
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return float(np.mean(pnls - mar) / dd) if dd > 0 else 0.0


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
        # Risk-adjusted return = SORTINO (downside deviation), not Sharpe (symmetric std). Sharpe
        # penalised bigger WINNERS as "risk"; Sortino only counts downside, so letting winners run
        # (catch the full wave) is rewarded, not docked. See calc_sortino.
        sortino = calc_sortino(trades)
        norm_riskadj = float(np.tanh(sortino / 1.55))  # scale 1.55: champion Sortino~1.36 maps to
        #                                                 the same ~0.70 the old Sharpe/0.55 gave,
        #                                                 so absolute scores stay comparable.
        mdd_sym = calc_mdd_per_symbol(trades)
        yr_cv = calc_yearly_consistency(trades)
    else:
        # Legacy no-trades path (CSV/live summary): no trade list to compute downside deviation, so
        # fall back to the stored Sharpe. MUST be in metrics — no silent default.
        sharpe = metrics.get("sharpe")
        if sharpe is None:
            raise ValueError(
                "composite_score requires either trades list or 'sharpe' in metrics dict. "
                "Neither provided. This indicates incomplete backtest data."
            )
        norm_riskadj = float(np.tanh(float(sharpe) / 0.55))
        mdd_sym = abs(metrics.get("max_loss", 0))
        yr_cv = 0.0

    # Per-bar expectancy: reward capital efficiency (return per unit of time held), NOT raw
    # per-trade gain. A 50%-in-40-bars trade beats 50%-in-200-bars. This removes the buy&hold
    # premium that raw avg_pnl carried (long holds inflated it) and crushes long-hold imposters.
    avg_per_bar = avg_pnl / max(avg_hold, 1.0)
    total_pnl = float(metrics.get("total_pnl", 0.0) or 0.0)

    # n_symbols = universe size (stored on the leaderboard row); fall back to distinct traded names
    # for legacy CSV/live paths. Used to normalise BOTH the total-PnL term and the confidence target
    # so the whole score stays symbol-count neutral (like sharpe/mdd/yr) and comparable across
    # universes of different sizes.
    n_symbols = int(metrics.get("n_symbols", 0) or 0)
    if n_symbols <= 0 and trades:
        n_symbols = len({t.get("symbol") for t in trades})

    norm_avg_bar = float(np.tanh(avg_per_bar / 0.0015))
    # Total-PnL completeness, PER SYMBOL: per-bar stays the priority (fast moves score high), but
    # once the fast phases are exhausted a model that EXPANDS into slower-but-still-profitable trades
    # to grow total PnL is rewarded. PER-SYMBOL (total_pnl / n_symbols) keeps it symbol-count neutral
    # — it rewards extracting MORE profit from each name (temporal completeness), NOT trading more
    # names (breadth), so it can't resurrect the universe-expansion lever. Saturating tanh bounds it
    # (no buy&hold runaway). Empty churn doesn't accumulate total_pnl → gains nothing here while it
    # loses on per-bar (case-1/case-2: more trades earn reward only when they actually add PnL).
    # Scale 1.475 = prior aggregate scale 90 / 61-sym baseline → 61-sym scores are unchanged.
    total_per_sym = total_pnl / max(n_symbols, 1)
    # SCORING-V2 (2026-06-18, user objective redesign): the saturating tanh PnL term gave a big total-PnL
    # gap only ~half-credit and the exp MDD penalty bit HARDEST at low MDD — together they rejected configs
    # that catch more profitable runners at the cost of a small MDD rise (conviction-pullback: +422u but
    # mdd 0.19->0.21). V2 fixes both: (3) LINEAR PnL (full marginal credit, capped only to bound a freak
    # outlier), (1) MDD FREE-ZONE — zero penalty below SCORE_MDD_FREE (stop squeezing already-low MDD),
    # linear above (still self-caps how much MDD is accepted). (2) PnL weight 0.34->SCORE_PNL_W.
    norm_total = float(np.clip(total_per_sym / 1.70, 0.0, SCORE_PNL_CAP))
    norm_pf = float(1.0 - np.exp(-max(pf - 1.0, 0.0) / 9.0))
    # MDD gentle-convex curve-to-0 (2026-06-18b, replaces the free-zone): always >0 so sub-0.22 risk
    # reductions are rewarded, but convex (pow>1) keeps the gradient small in the operating range so
    # profitable higher-mdd runners are not over-penalised the way the old exp curve did.
    norm_mdd = float(np.clip((max(mdd_sym, 0.0) / SCORE_MDD_DIV) ** SCORE_MDD_POW, 0.0, 1.0))
    norm_yr = float(max(yr_cv - 0.35, 0.0) / 2.0)

    # Weights (2026-06-11): per-bar lowered 0.27->0.20 and total-PnL raised 0.20->0.27 so that
    # ONCE a model trades enough (its per-bar anti-buy&hold job already done by the confidence
    # shrinkage + tanh-bounded total), TOTAL PROFIT is prioritised over raw capital-velocity. This
    # stops a higher-PnL active model losing to a marginally-faster one (forensics: t1129 had +12%
    # PnL / higher PF+Sharpe yet ranked below the champion purely on turnover).
    # Weights (2026-06-11b): per-bar 0.20->0.13 and total 0.27->0.34. Per-bar is already tanh-
    # saturated (~0.90-0.96 for every real model) so it barely discriminates yet was docking
    # longer-hold models that capture MORE total profit; shifting that weight to total rewards the
    # genuine-profit axis the user prioritises in the mature phase (ride the whole wave). Per-bar
    # stays a small anti-buy&hold floor.
    quality_score = (
        0.15 * norm_riskadj
        + 0.08 * norm_avg_bar  # 0.13->0.08 (user 2026-06-19): cut the per-bar/hold-time penalty so
        + SCORE_PNL_W * norm_total
        + 0.18 * norm_pf
        - 0.15 * np.clip(norm_mdd, 0, 1)
        - 0.07 * np.clip(norm_yr, 0, 1)
    ) * 1000

    n_trades = max(int(metrics.get("trades", 0)), 0)
    # Confidence = thin-sample shrinkage with SMOOTH diminishing returns, NO hard cap (2026-06-11):
    #   confidence = 1 - exp(-n_trades / k),   k = 15 round-trips/symbol * n_symbols  (≈915 @ 61 sym)
    # Concave & strictly increasing, so a higher-volume model is never out-ranked by a thinner one
    # at equal quality (more trades always help a little — no flat ceiling where 1300 == 2100), but
    # the MARGINAL value of trades tapers: once a model trades enough (~1300+ on the 61-sym set) extra
    # round-trips barely move the score and quality/total-PnL decides. Genuine thin samples (a few
    # hundred trades -> 0.2-0.4) are still shrunk so buy&hold imposters can't win on quality alone.
    # Replaces the old sqrt(n / full-coverage) + hard 1.0 cap, which acted as a throughput PENALTY:
    # it discounted a higher-PnL model 9% for trading 2162 vs 2643 times (root-caused 2026-06-11).
    KPS = 600.0 / 60.0  # =10 round-trips/symbol e-folding constant; flatter in the 2000+ zone so
    #                     once trades are ample, QUALITY/total-PnL decides over raw trade count.
    k = KPS * n_symbols if n_symbols > 0 else 900.0
    confidence = float(1.0 - np.exp(-n_trades / k)) if n_trades > 0 else 0.0

    # Confidence INFLUENCE blend (2026-06-11b): keep the k=10 thin-sample SHAPE but reduce how hard
    # the trade-count haircut bites, so a higher-quality model trading somewhat fewer times is no
    # longer out-ranked purely on turnover. mult = (1-a) + a*confidence, a=0.80. Diluting the
    # multiplier (not lowering k) preserves the relative penalty on genuinely thin samples (<~1000
    # trades) so buy&hold imposters stay buried, while a≥~0.6 is the floor below which they erupt.
    CONF_ALPHA = 0.68  # 0.80->0.68 (user 2026-06-19): soften the trade-count haircut (raise the floor
    #                    0.20->0.32) so a higher-quality model trading somewhat fewer times isn't docked
    conf_mult = (1.0 - CONF_ALPHA) + CONF_ALPHA * confidence

    return round(quality_score * conf_mult, 1)


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
