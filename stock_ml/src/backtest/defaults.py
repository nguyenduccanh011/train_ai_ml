"""Trading cost defaults — single source of truth.

VN market realistic baseline:
  - commission: 0.15% per side (broker fee)
  - tax: 0.10% on sell only (Vietnamese stock transfer tax)
  - slippage: 0.10% per side (typical bid-ask + execution drift on liquid VN30 names)

Override at call site if a specific symbol/regime warrants different costs.
"""

from __future__ import annotations

DEFAULT_TRADING_COST: dict[str, float] = {
    "commission": 0.0015,
    "tax": 0.0010,
    "slippage": 0.0010,
}

DEFAULT_INITIAL_CAPITAL: float = 100_000_000.0
