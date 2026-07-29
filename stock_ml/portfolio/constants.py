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
    # sizing/overshoot statistics mode:
    #   causal (DEFAULT, promoted 2026-07-29) = expanding per-year like the SKIP gate (year
    #            with <30 prior trades -> neutral w=1.0 / no overshoot filter); adding a fold
    #            never changes the past. At deploy time full-history stats ARE expanding, so
    #            causal is the honest backtest of live behaviour: 3-seed T+2 138.5% / -13.0%.
    #   full   = mu/sd/off + osthr from the WHOLE period — kept for parity with the
    #            _champ_prod_replay reference + the original champion golden (140.3% / -10.8%).
    stat_mode: str = "causal"
    # DD-signature valves (loss forensic 2026-07-29: the 3 deep-DD episodes had 3 distinct
    # book signatures — vol-concentration / sequential knife-fills in a crash / full-beta).
    # Both OFF by default -> code path byte-identical to the golden.
    vol_cap_q: float | None = None   # cap concurrent HIGH-VOL holdings; a fill is high-vol when
                                     # its vol20 >= same-day cross-sectional quantile q of the
                                     # market panel (causal: same-day cross-section only)
    vol_cap_max: int = 2             # max simultaneous high-vol positions when vol_cap_q is set
    crash_pause_ret5: float | None = None  # pause NEW fills while EW-market 5d return <= this
    # exit-side valve (the entry-side valves above were REFUTED: max-DD comes from HELD
    # positions riding the slide, not from new fills). Force-exit held positions on
    # market-stress days; new fills on LATER days stay allowed (the V-bottom entries
    # are profitable — crash_pause showed blocking them costs 9pp CAGR for zero DD).
    riskoff_ret5: float | None = None  # evict held legs while EW-market 5d return <= this
    riskoff_scope: str = "all"         # "all" | "losers" (only legs below their entry)
    # slow state-based valve (event-triggered pause/sell-down both REFUTED — too late,
    # sells the V-bottom): scale NEW fills' weight while market breadth is weak instead
    # of blocking anything. Targets the 2022 signature (full 0.99 exposure into a bear).
    regime_w_scale: float | None = None  # multiply new-fill weight by this when breadth low
    regime_breadth_thr: float = 0.35     # breadth = pct of market panel above own MA50
    # liquidity-COLLAPSE veto (2026-07-29 forensic): a formerly-liquid name whose current
    # traded value has collapsed is in distress (no-bid spiral, 3x limit-chain risk,
    # worst pnl cohort); chronically-thin names are FINE (best cohort) and must NOT be
    # cut. Veto entries where trailing-10d ADV < liqcol_adv10_ty AND trailing-1y ADV
    # >= liqcol_adv252_ty (tỷ VND = traded_value/1e6). None = OFF (golden-identical).
    liqcol_adv10_ty: float | None = None
    liqcol_adv252_ty: float = 10.0
    # capital-allocation experiments (2026-07-29; default None/off = golden-identical):
    w_invvol: float | None = None    # inverse-vol tilt: w *= clip(0.025/vol20, 1/x, x)
    w_liq_full_ty: float | None = None  # liquidity-proportional size: w *= clip(adv10/X_ty, 0.3, 1)
    max_expo: float | None = None    # skip new fills while invested value >= this frac of NAV
