"""Portfolio construction tests (Phase 2): AlphaFrame -> TargetWeightFrame.

Pure-math checks on synthetic score frames — no models, no execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from src.portfolio import build_portfolio_constructor  # noqa: E402
from src.portfolio.base import PortfolioContext  # noqa: E402
from src.portfolio.factory import build_portfolio_context  # noqa: E402


def _alpha(scores: dict[str, float], date: str = "2020-01-02") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": list(scores.keys()),
            "date": pd.Timestamp(date),
            "score": list(scores.values()),
        }
    )


def test_topk_selects_highest_and_equal_weights():
    alpha = _alpha({"A": 0.5, "B": 0.3, "C": 0.1, "D": -0.2, "E": -0.4})
    pc = build_portfolio_constructor({"policy": "top_k", "params": {"k": 2}})
    out = pc.build(alpha, PortfolioContext(direction="long", max_gross=1.0, max_per_name=1.0))

    active = out[out["target_weight"] != 0].set_index("symbol")
    assert set(active.index) == {"A", "B"}  # two highest scores
    assert abs(active["target_weight"].sum() - 1.0) < 1e-9  # full deployment
    assert abs(active.loc["A", "target_weight"] - active.loc["B", "target_weight"]) < 1e-9


def test_market_neutral_is_dollar_neutral():
    alpha = _alpha({"A": 0.9, "B": 0.5, "C": 0.0, "D": -0.5, "E": -0.9})
    pc = build_portfolio_constructor({"policy": "market_neutral", "params": {"k": 2}})
    out = pc.build(alpha, PortfolioContext(direction="market_neutral", max_gross=1.0, max_per_name=1.0))

    net = out["target_weight"].sum()
    gross = out["target_weight"].abs().sum()
    assert abs(net) < 1e-9  # net ≈ 0
    assert abs(gross - 1.0) < 1e-9  # gross deployed
    longs = set(out[out["target_weight"] > 0]["symbol"])
    shorts = set(out[out["target_weight"] < 0]["symbol"])
    assert longs == {"A", "B"}
    assert shorts == {"D", "E"}


def test_long_only_drops_negative_scores():
    alpha = _alpha({"A": 0.4, "B": -0.4})
    pc = build_portfolio_constructor({"policy": "score_proportional"})
    out = pc.build(alpha, PortfolioContext(direction="long")).set_index("symbol")
    assert out.loc["A", "target_weight"] > 0
    assert out.loc["B", "target_weight"] == 0


def test_max_per_name_caps_concentration():
    alpha = _alpha({"A": 0.9, "B": 0.05, "C": 0.05})
    pc = build_portfolio_constructor({"policy": "score_proportional"})
    out = pc.build(alpha, PortfolioContext(direction="long", max_gross=1.0, max_per_name=0.4))
    assert out["target_weight"].max() <= 0.4 + 1e-9


def test_threshold_binary_band():
    alpha = _alpha({"A": 0.03, "B": 0.0, "C": -0.03})
    pc = build_portfolio_constructor(
        {"policy": "threshold_binary", "params": {"entry_threshold": 0.02, "exit_threshold": -0.02}}
    )
    # long-only: only A (score above entry) is held
    out = pc.build(alpha, PortfolioContext(direction="long")).set_index("symbol")
    assert out.loc["A", "target_weight"] > 0
    assert out.loc["B", "target_weight"] == 0
    assert out.loc["C", "target_weight"] == 0


def test_regime_gate_forces_flat():
    alpha = _alpha({"A": 0.5, "B": 0.5})
    regime = pd.DataFrame(
        {"date": [pd.Timestamp("2020-01-02")] * 2, "symbol": ["A", "B"], "gate": [1, 0]}
    )
    pc = build_portfolio_constructor({"policy": "top_k", "params": {"k": 5}})
    ctx = build_portfolio_context({"max_gross": 1.0}, direction="long", regime_signal=regime)
    out = pc.build(alpha, ctx).set_index("symbol")
    assert out.loc["B", "gated"]
    assert out.loc["B", "target_weight"] == 0
    assert abs(out.loc["A", "target_weight"] - 1.0) < 1e-9  # A absorbs full gross
