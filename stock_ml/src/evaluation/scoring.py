"""
Unified scoring — single source of truth for model ranking.

Provides a canonical composite_score() used by model_manager, run_pipeline,
and all comparison scripts.  Weights are configurable via config/models.yaml
under the ``scoring`` key; sensible defaults are provided when that section
is absent.
"""
import numpy as np

from src.config_loader import load_config


def _get_weights():
    cfg = load_config()
    scoring_cfg = cfg.get("scoring", {})
    w = scoring_cfg.get("weights", {})
    return {
        "total_pnl": w.get("total_pnl", 0.30),
        "profit_factor": w.get("profit_factor", 0.25),
        "win_rate": w.get("win_rate", 0.20),
        "avg_pnl": w.get("avg_pnl", 0.15),
        "max_loss_penalty": w.get("max_loss_penalty", 0.10),
    }


def composite_score(metrics: dict) -> float:
    """Canonical composite score (higher = better).

    *metrics* must contain at least: total_pnl, pf, wr, avg_pnl, max_loss.
    All values in the same units as calc_metrics() output (percentages for
    pnl/wr, ratio for pf).
    """
    if metrics.get("trades", 0) == 0:
        return 0.0

    w = _get_weights()

    total_pnl = metrics.get("total_pnl", 0)
    pf = metrics.get("pf", 0)
    wr = metrics.get("wr", 0)
    avg_pnl = metrics.get("avg_pnl", 0)
    max_loss = abs(metrics.get("max_loss", 0))

    norm_pnl = np.clip(total_pnl / 500, -1, 1)
    norm_pf = np.clip((pf - 1) / 3, -1, 1)
    norm_wr = np.clip((wr - 50) / 30, -1, 1)
    norm_avg = np.clip(avg_pnl / 3, -1, 1)
    norm_ml = np.clip(max_loss / 30, 0, 1)

    score = (
        w["total_pnl"] * norm_pnl
        + w["profit_factor"] * norm_pf
        + w["win_rate"] * norm_wr
        + w["avg_pnl"] * norm_avg
        - w["max_loss_penalty"] * norm_ml
    ) * 1000

    return round(score, 1)


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
