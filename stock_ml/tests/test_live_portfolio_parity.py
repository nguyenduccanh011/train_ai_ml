"""Phase 5 parity: live sizing == batch sizing for the same day's scores.

Proves the live simulator and the batch backtest turn an identical set of alpha
scores into an identical book, so enabling portfolio mode keeps them consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.live_sim.portfolio_bridge import frozen_to_targets  # noqa: E402
from src.live_sim.signals import FrozenSignalSet  # noqa: E402
from src.portfolio.base import PortfolioContext  # noqa: E402
from src.portfolio.factory import build_portfolio_constructor  # noqa: E402


def _frozen(scores: dict[str, float], date="2021-03-01") -> FrozenSignalSet:
    return FrozenSignalSet(
        generated_at=pd.Timestamp(date) - pd.Timedelta(days=1),
        for_execution_date=pd.Timestamp(date),
        signals={s: (1 if v > 0 else (-1 if v < 0 else 0)) for s, v in scores.items()},
        n_buy=0,
        n_sell=0,
        n_neutral=0,
        filters_applied=[],
        integrity_hash="x",
        scores=scores,
    )


def test_live_and_batch_produce_same_weights():
    scores = {"A": 0.6, "B": 0.3, "C": -0.1, "D": -0.5}
    date = "2021-03-01"
    cfg = {"policy": "top_k", "params": {"k": 2}, "max_gross": 1.0, "max_per_name": 1.0}

    # Live path: frozen scores -> bridge.
    live = frozen_to_targets(_frozen(scores, date), cfg, direction="long")

    # Batch path: same scores as a one-day AlphaFrame -> same constructor.
    alpha = pd.DataFrame(
        {"symbol": list(scores), "date": pd.Timestamp(date), "score": list(scores.values())}
    )
    batch = build_portfolio_constructor(cfg).build(
        alpha, PortfolioContext(direction="long", max_gross=1.0, max_per_name=1.0)
    )

    live_w = live.set_index("symbol")["target_weight"].sort_index()
    batch_w = batch.set_index("symbol")["target_weight"].sort_index()
    pd.testing.assert_series_equal(live_w, batch_w)


def test_frozen_carries_scores():
    fs = _frozen({"A": 0.2, "B": -0.2})
    assert fs.scores == {"A": 0.2, "B": -0.2}
    # signals derived consistently with score sign
    assert fs.buys() == frozenset({"A"})
    assert fs.sells() == frozenset({"B"})
