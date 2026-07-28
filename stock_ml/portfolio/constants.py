"""Portfolio-layer (Stage-2) constants — single source of truth.

Ported verbatim from stock-serving serving/portfolio/core.py (itself a validated
port of hb_deploy_gtos.py). Golden-guarded: stock_ml/tests/test_portfolio_golden.py.
"""
from __future__ import annotations

from dataclasses import dataclass

# cost / execution constants (identical to nh_nav2 + hb_deploy_gtos)
S0 = 0.001          # slippage already baked into entry/exit prices
FEE = 0.004         # 2*commission + tax
ROUNDTRIP = 0.006   # realistic round-trip; s_new = (ROUNDTRIP-FEE)/2 per side
ADVANCE_FEE = 0.0008
CS4 = ["dist20low", "dist_ma20", "rsi14", "ret20"]
FCOLS = ['score', 'exit_score', 'ret5', 'ret20', 'ret60', 'vol20', 'dist_h20', 'dist_h63',
         'dist_l20', 'dist_l63', 'ma20r', 'ma50r', 'atr_pct', 'updays10', 'volr', 'rs_mom20', 'rs_mom60']


@dataclass
class PortfolioConstants:
    k: int = 10
    margin: float = 0.005          # preemption priority margin
    kconv: float = 2.0             # conviction z sizing gain
    tplus: int = 2                 # T+2 min-hold + settlement
    r5thr: float = 0.02            # ret7 gate
    skip: float = 0.40             # conviction floor gate (base, causal-adjusted per year)
    skip_mode: str = "causal"      # causal (per-year, no look-ahead) | fixed | off
    skip_gain: float = 2.0         # panel-adaptive gate: skip += max(0, mu_past-mu_ref)*gain
    skip_mu_ref: float = 0.644     # reference conviction mean (~200-symbol panel); calibrated VN
    gt: float = 0.08               # green-trail giveback
    os_pct: float = 90.0           # overshoot filter percentile (in-sample; KNOWN non-causal,
                                   # parity-preserved — causal-ize post-refactor, see design doc §6)
    date_lo: str = "2020-01-01"
    market_start: str = "2018-06-01"
    # research-lever knobs (defaults = champion gtos behaviour, golden-guarded):
    rewrite_on: bool = True        # False = no exit rewrite at all (ret5g/ret7g variants)
    ec_check_bar: int = 2          # early-cut checks close[entry+N] red, cuts at N+1 (gt1 uses 1)
    ret_win: int = 7               # ret-gate momentum window (ret5g variant uses 5)
