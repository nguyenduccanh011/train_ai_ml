"""Leakage-safe backtest engine.

Rules (configurable via EngineConfig.entry_bar_fill_type):

**Mode: open_next (default, leakage-safe)**
  - Entry: OPEN of bar t+1 (next-bar fill after signal at bar t)
  - Exit: OPEN of bar t+1
  - No lookahead bias. Recommended for production.

**Mode: close_next (leakage-safe, +3.62pp improvement)**
  - Entry: CLOSE of bar t+1 (next bar after signal at bar t)
  - Exit: CLOSE of bar t+1
  - No lookahead bias. Better entry/exit timing than open_next.

**Mode: close_same (lookahead bias - for testing only)**
  - Entry: CLOSE of bar t (same bar as signal)
  - Exit: CLOSE of bar t+1 (next bar)
  - WARNING: Has lookahead bias. Do not use for production.

Common rules:
  - Costs (per side, multiplicative): slippage, commission, tax
  - One open trade per symbol at a time (no pyramiding)
  - Signals must come from out-of-sample data (when using leakage-safe modes)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .defaults import DEFAULT_TRADING_COST

_VNI_CACHE: dict = {}


def _load_vnindex(path="portable_data/vn_stock_ai_dataset_cleaned/context_features/symbol=VNINDEX/timeframe=1D/data.csv"):
    """VNINDEX daily close (date->close), for RS-vs-market (close/VNINDEX). Cached. None if absent."""
    if "vni" in _VNI_CACHE:
        return _VNI_CACHE["vni"]
    import os
    if not os.path.exists(path):
        _VNI_CACHE["vni"] = None
        return None
    v = pd.read_csv(path)
    v["d"] = pd.to_datetime(v["timestamp"]).dt.tz_localize(None).dt.normalize()
    v = v.drop_duplicates("d", keep="last").set_index("d")["close"].astype(float)
    _VNI_CACHE["vni"] = v
    return v


_RUNSCORE_CACHE: dict = {}


def _load_runscore(path="results/_research_2429/runscore.parquet"):
    """Causal OOF 'remaining-run' head predictions (walk-forward, leg+momentum+volume features),
    per-symbol Series of runscore_z indexed by date. Cached. None if absent. See train_runscore.py."""
    if "rs" in _RUNSCORE_CACHE:
        return _RUNSCORE_CACHE["rs"]
    import os
    if not os.path.exists(path):
        _RUNSCORE_CACHE["rs"] = None
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    by = {s: g.set_index("date")["runscore_z"] for s, g in df.groupby("symbol")}
    _RUNSCORE_CACHE["rs"] = by
    return by


def _causal_leg_age(closes: np.ndarray, pct: float):
    """Per-bar CAUSAL zigzag current-leg age + amplitude, normalized vs the stock's history.

    Probe (wave_head_probe): the AGE of the current (unconfirmed) impulse leg predicts remaining
    forward run with incremental IC -0.27 AFTER removing momentum (young leg = more run ahead);
    the AMPLITUDE-so-far vs typical up-leg adds incremental IC +0.14 (a strong impulse keeps
    running). Returns (legage_norm, legamp_norm):
      legage_norm = cur_leg_age / typical_up_leg_len  - 1   (>0 older than typical)
      legamp_norm = |cur_leg_move| / typical_up_leg_amp - 1 (>0 bigger move than typical)
    Only pivots CONFIRMED on/before bar i are used (the reversal must have already happened),
    so it is strictly causal.
    """
    n = len(closes)
    if n < 2:
        return np.zeros(n), np.zeros(n)
    direction = 0
    ext_idx = 0
    ext_price = closes[0]
    prev_pidx = 0
    confirm: dict[int, tuple] = {}  # confirm_bar -> (pivot_idx, kind, up_leg_len, up_leg_amp)
    for i in range(1, n):
        p = closes[i]
        if direction >= 0 and p > ext_price:
            ext_price = p; ext_idx = i; direction = 1
        elif direction <= 0 and p < ext_price:
            ext_price = p; ext_idx = i; direction = -1
        elif direction == 1 and p <= ext_price * (1.0 - pct):
            _amp = abs(closes[ext_idx] / closes[prev_pidx] - 1.0) if closes[prev_pidx] else None
            confirm[i] = (ext_idx, +1, ext_idx - prev_pidx, _amp)  # up-leg just ended at a peak
            prev_pidx = ext_idx; direction = -1; ext_price = p; ext_idx = i
        elif direction == -1 and p >= ext_price * (1.0 + pct):
            confirm[i] = (ext_idx, -1, None, None)
            prev_pidx = ext_idx; direction = 1; ext_price = p; ext_idx = i
    active = np.zeros(n, dtype=np.int64)
    typ_len = np.full(n, np.nan)
    typ_amp = np.full(n, np.nan)
    last_pidx = 0
    ulens: list[int] = []
    uamps: list[float] = []
    cur_tl = np.nan
    cur_ta = np.nan
    for i in range(n):
        if i in confirm:
            pidx, kind, ulen, uamp = confirm[i]
            last_pidx = pidx
            if kind == +1 and ulen:
                ulens.append(ulen); cur_tl = float(np.median(ulens))
                if uamp:
                    uamps.append(uamp); cur_ta = float(np.median(uamps))
        active[i] = last_pidx
        typ_len[i] = cur_tl
        typ_amp[i] = cur_ta
    idx = np.arange(n)
    cur_leg_age = idx - active
    cur_leg_amp = np.abs(closes / np.where(closes[active] == 0, np.nan, closes[active]) - 1.0)
    typ_len = np.where(np.isnan(typ_len) | (typ_len <= 0), np.nan, typ_len)
    typ_amp = np.where(np.isnan(typ_amp) | (typ_amp <= 0), np.nan, typ_amp)
    legage_norm = np.nan_to_num(cur_leg_age / typ_len - 1.0, nan=0.0)
    legamp_norm = np.nan_to_num(cur_leg_amp / typ_amp - 1.0, nan=0.0)
    return legage_norm, legamp_norm


@dataclass
class CostModel:
    commission: float = DEFAULT_TRADING_COST["commission"]
    tax: float = DEFAULT_TRADING_COST["tax"]
    slippage: float = DEFAULT_TRADING_COST["slippage"]

    def fill_buy(self, raw_price: float) -> float:
        return raw_price * (1.0 + self.slippage)

    def fill_sell(self, raw_price: float) -> float:
        return raw_price * (1.0 - self.slippage)

    def round_trip_cost(self) -> float:
        """Sum of fee/tax fractions applied to gross pnl (per-side commission ×2 + tax)."""
        return 2.0 * self.commission + self.tax


@dataclass
class EngineConfig:
    # Exit controls are REQUIRED — no silent defaults. Every run must declare its
    # max-hold and hard-stop explicitly so an omitted key fails loudly instead of
    # silently inheriting a hidden -0.08 / 20-bar policy. To disable: set
    # hard_stop_pct=None, and/or drop "max_hold" from exit_priority.
    max_hold_bars: int
    hard_stop_pct: float | None  # close trade if mtm drops below this; None disables the stop
    min_hold_bars: int = 1
    signal_exit_enabled: bool = True  # whether to use model's sell signal (-1) as exit trigger
    # AGE INCUBATION (regime-conditional exit, TIME axis): forensics on champion t1187 —
    # the residual signal exits are premature and YOUNG (deferral benefit concentrates in
    # hold<9 bars: +67.9u of +88.6; signal exits at hold>=16 are already net-positive +10u).
    # A young trade signal-cut at hold 1-8 has not had time to reach the +15% trail arm, so
    # it dies before any profit protection and bounces after. When >0, the "signal" exit rule
    # is SUPPRESSED until the position is at least this many bars old (defers to trail/hard
    # stop), incubating young trades so they can develop into trail-managed winners. The entry
    # survival score does NOT separate which young trades to keep (corr~0) — age alone is the
    # lever. 0 = disabled. Orthogonal to the market-regime gate (washout axis vs age axis).
    signal_exit_min_age: int = 0
    # Incubation underwater guard: blanket young-incubation lifts pnl & sharpe but also lets
    # the entry-driven young LOSERS (straight-down, never reach the trail arm) ride deeper →
    # mdd jumps (2.91→4.06). When set, incubation is SKIPPED (the signal exit is honored even
    # while young) once the trade's current close-to-entry mtm falls below this floor — i.e.
    # cut the genuine young losers, incubate only the flat/green young trades. None = no guard
    # (incubate all young trades). Only consulted when signal_exit_min_age > 0.
    signal_exit_incubate_floor: float | None = None
    # EXP3 SIGNAL-EXIT MFE PROTECT BAND (forensic t1831): the 'signal' exit captures a NEGATIVE
    # fraction of MFE (-1.34) and dumps winners that had already reached double-digit gains — 442
    # trades peaking 10-27% MFE gave back 46.9u. When both bounds are set, the 'signal' exit is
    # SUPPRESSED while peak-gain since entry is in [lo, hi) (e.g. [0.10, 0.27)), deferring to the
    # trailing/overext/downleg rules so the trade can ride toward the trail arm. If
    # signal_exit_protect_require_trend, the suppression applies ONLY while price is above a rising
    # trend MA (reuses trailing_skip_above_ma's trend_up). None = off (legacy). Causal.
    signal_exit_protect_lo: float | None = None
    signal_exit_protect_hi: float | None = None
    signal_exit_protect_require_trend: bool = False
    # EXP3 COHERENCE VETO (sell-then-rebuy-higher fix): 66% of loss-then-reenter trades rebuy the
    # same name HIGHER within 40 bars — the decoupled heads contradict (exit dumps a dip, entry
    # rebuys the bounce). When set, the 'signal' exit is SKIPPED if the entry head's causal z
    # (252/60 z of the 'score' column, same window as recombine) is still >= this threshold — i.e.
    # don't sell while the entry thesis is still a BUY. None = off. Causal (score up to bar i).
    signal_exit_skip_if_entry_z: float | None = None
    # SCORE3 CONTINUATION VETO (2026-06-17, forensic t1930 decomp): the EXP3 veto above uses the
    # base momentum 'score' (realized-pnl IC ~0) and tested negative (batch D, 437-476) — momentum
    # is high at toppy names you SHOULD sell. score3 = the continuation head, the ONLY entry head
    # with IC vs REALIZED pnl > 0 (+0.114) and IC vs bars_to_peak +0.155: high-score3 trades are
    # genuine runners (peak +16.8% vs +10.0%, run 23 vs 11 bars) that the velocity exit dumps too
    # early. When set, the 'signal' exit is SKIPPED while the causal 252/60 z of the 'score3' column
    # is >= this threshold — hold the predicted continuation, handing it to the trailing/overext arm.
    # Distinct lever from signal_exit_skip_if_entry_z (different head, evaluated independently; both
    # may be set). None = off. Causal. Non-masking: suppresses an exit, drops no trade.
    signal_exit_skip_if_score3_z: float | None = None
    # MARKET-REGIME signal-exit skip (2026-07-11, shakeout_vs_top forensic hb_60/61): 49% of 'signal'
    # exits are shakeouts (dip that recovers +13.7%) and the strongest discriminator is broad-market
    # regime — VNINDEX > MA(N) => AUC 0.78 (overall) / 0.59-0.94 (within-year): a signal-exit fired in
    # a bull tape is usually a shakeout, in a bear tape a real top. The exit-recombine z-scores the ML
    # signal so this slow common-factor is washed out at the head; act on it at the ENGINE instead —
    # SKIP the 'signal' exit (defer to trail/overext/max_hold) while VNINDEX/MA(N) - 1 >= margin. Only
    # holds in a BULL tape (safe: never holds into a bear-flip, unlike a fixed longer-hold). Optionally
    # gate on the trade being a winner (skip_if_mkt_winner_only) so losers still cut. None = off. Causal
    # (VNINDEX up to the bar). Needs the VNINDEX csv.
    signal_exit_skip_if_mkt_above_ma: int | None = None
    signal_exit_skip_if_mkt_margin: float = 0.0
    signal_exit_skip_if_mkt_winner_only: bool = False
    # SLOW-TREND protect (regime-masking 2026-06-17): the signal_exit_protect trend check defaults to
    # the SHORT MA10; a deeper pullback breaking MA10 releases the protect and the signal exit cuts a
    # bull winner whose LONGER trend is intact. Set this to a SLOW MA window (e.g. 50/100) so the
    # protect holds winners through deep pullbacks while close > rising MA(window) — a per-stock regime
    # gate (above slow MA = bull = hold; below = cut). None = default short trend_up.
    signal_exit_protect_ma: int | None = None
    # PROTECT-RELEASE early-escape (2026-06-17, pm100_dip035 forensic): the protect band above holds a
    # winner through deep pullbacks while close>rising MA(protect_ma); but a SLOW MA releases ~20-30 bars
    # AFTER the swing top, so the signal exit fires deep on the down-leg (forensic: median 18-19 bars past
    # the top, giveback 141.5u > realized 120.8u). When set, the protect is RELEASED early (the signal exit
    # is allowed to fire) once the close has dropped >= release_drop_k * vol below the running peak, where
    # vol = ATR(release_vol_window)/close (per-stock, causal). Vol-scaled so a high-vol name needs a deeper
    # give-back to release (distinct from a fixed pop-lock %). Releases the SUPPRESSION of an already-firing
    # exit — does NOT predict the top, never arms a tight trail on the way up. None = off (protect behaves
    # exactly as before). Non-masking: suppresses one fewer exit, drops no trade.
    signal_exit_protect_release_drop_k: float | None = None
    signal_exit_protect_release_vol_window: int = 20
    # VOL-ADAPTIVE EXTENSION HOLD (2026-06-20, vol_adaptive_forensic on 2482): the signal-exit fires at
    # close ~= MA20 (ext -0.4%) yet winners then run a further +2.7 ATR (low-vol +3.7 ATR, 93% run >0.5
    # ATR) — it sells INTO THE MEAN, pre-empting the overext "sell-into-strength" rule which therefore
    # never fires. The fixed-% protect bands can't express this because "how extended" is RELATIVE to
    # the stock's own ATR. When set (K), SUPPRESS the signal exit while the trade is in profit, its trend
    # is intact (close > rising MA(signal_exit_hold_ma)), AND it is NOT yet extended K*ATR above that MA
    # (ext_atr = (close/MA - 1)/(ATR14/close) < K) — let the un-stretched winner ride toward extension and
    # exit via overext/trail/the signal once stretched. A per-stock vol-normalised "don't sell the mean,
    # sell the stretch". Protective gates (downleg/hard_stop/force) still fire, so a roll-over without
    # extension still exits. None = off. Causal (ATR/MA up to the bar). Non-masking: suppresses one exit,
    # drops no trade.
    signal_exit_hold_ext_atr: float | None = None
    signal_exit_hold_ma: int = 20
    # Optional SELECTIVITY for the extension hold: only hold an un-stretched winner when the CONTINUATION
    # head (score3 z) still predicts a run >= this z (capture the winners that will keep going; let the
    # roll-overs exit on the signal) — turns the risk-superior-but-throughput-masked uniform hold into a
    # selective one (fewer holds, bigger captured runs -> recover the per-bar/conf throughput cost). And
    # only hold when the stock's ATR/close <= this cap (target the lower-vol names whose post-exit run is
    # largest in ATR). None = no extra gate. Causal.
    signal_exit_hold_min_score3_z: float | None = None
    signal_exit_hold_max_atrpct: float | None = None
    # VOL-DEPENDENT hold depth (forensic: low-vol winners run +3.7 ATR post-exit vs high-vol +2.1 ATR —
    # they keep running further even in ATR units). When True, the K threshold is SCALED per stock by
    # clip(volscale_ref / (ATR/close), 0.5, 2.0) so LOW-vol names are held to a DEEPER ATR-extension
    # (capture their bigger run) and HIGH-vol names exit sooner. None/False = flat K. Causal.
    signal_exit_hold_volscale: bool = False
    signal_exit_hold_volscale_ref: float = 0.035
    # RS-vs-MARKET adaptive hold depth (2026-06-21): RS-vs-market is the strongest separator (leaders'
    # dips win 72%% vs laggards 52%%) but SELECTION-walled (laggards still net-positive). A NON-selection
    # use: scale the hold K by RS — hold LEADERS (rs_vsma>0) to a DEEPER ATR-extension (they run more),
    # exit LAGGARDS sooner. K_eff *= clip(1 + rs_scale*rs_vsma, 0.5, 2.0). Tunes the EXIT per-stock, does
    # NOT gate entries — so it can realize RS under the total-PnL objective where the gate could not.
    signal_exit_hold_rs_scale: float = 0.0
    signal_exit_hold_rs_feature: str = "rs_vsma"   # rs_vsma | rs_sl5 | rs_sl20 | rs_nh
    # RS-vs-MARKET adaptive RELEASE (2026-06-21, symmetric to the hold): scale the protect early-release
    # drop-threshold by RS — release a LAGGARD's protect SOONER (smaller drop triggers the exit; its
    # giveback is more likely terminal, it won't bounce) and a LEADER's LATER (let it ride). drop_k_eff
    # *= clip(1 + rs_scale*rs_vsma, 0.5, 2.0). Pairs with signal_exit_hold_rs_scale. 0 = off.
    signal_exit_release_rs_scale: float = 0.0
    # MARKET BIG-TREND adaptive hold depth (2026-06-21, wave-structure probe): loss-streaks cluster in
    # market-wide down-windows and long-hold WINNERS systematically enter in a healthier BIG-TREND context
    # (VNINDEX above its long MA / higher in its 120d wave) than short-hold losers — but the regime is
    # SELECTION-walled (every regime bucket is net-positive; dropping correction-entries halves PnL). The
    # NON-selection use, symmetric to rs_scale (stock-vs-market): scale the hold K by the MARKET's own
    # big-trend strength (same for all symbols on a date) — hold winners DEEPER when the broad trend
    # supports continuation, exit sooner when the market is weak/late in its wave. K_eff *= clip(1 +
    # mkt_scale*mkt_trend, 0.5, 2.0). Multiplies orthogonally with rs_scale/volscale. 0 = off. Causal
    # (VNINDEX up to the bar, same-day). Needs the VNINDEX csv.
    signal_exit_hold_mkt_scale: float = 0.0
    signal_exit_hold_mkt_feature: str = "ma100"   # ma100 | ma200 | pos120 | dd120
    # WAVE-STRUCTURE (current-leg AGE) adaptive hold depth (2026-06-21, wave_head_probe): the age of
    # the current unconfirmed zigzag leg predicts remaining forward run with incremental IC -0.27 AFTER
    # removing momentum (orthogonal — NOT subsumed like market-trend). A YOUNG impulse leg has more run
    # ahead -> hold deeper; an AGED leg is exhausted -> exit sooner. K_eff *= clip(1 - legage_scale *
    # legage_norm, 0.5, 2.0) where legage_norm = cur_leg_age/typical_up_leg_len - 1 (>0 = older). The
    # FIRST exit-lever fed by an explicit multi-wave-structure input. 0 = off. Causal (pivots confirmed
    # on/before the bar). pct = the zigzag reversal threshold defining a leg.
    signal_exit_hold_legage_scale: float = 0.0
    signal_exit_hold_legage_pct: float = 0.08
    # WAVE-STRUCTURE (current-leg AMPLITUDE) adaptive hold, orthogonal partner to legage (probe incr IC
    # +0.14 vs forward-run): a current impulse leg whose move-so-far is BIG vs the stock's typical up-leg
    # keeps running -> hold deeper; a small/weak leg -> exit sooner. K_eff *= clip(1 + legamp_scale *
    # legamp_norm, 0.5, 2.0), legamp_norm = |cur_leg_move|/typical_up_leg_amp - 1 (>0 bigger). Shares the
    # legage zigzag (signal_exit_hold_legage_pct). 0 = off. Causal.
    signal_exit_hold_legamp_scale: float = 0.0
    # 'REMAINING-RUN' ML HEAD adaptive hold (2026-06-21): a walk-forward GBM on leg-structure + momentum
    # + volume predicts forward 40-bar run (OOF, causal, z-scored per test-year cross-section; see
    # train_runscore.py -> runscore.parquet). Scale the hold K by its z so the model holds winners it
    # predicts will RUN deeper and exits the rest sooner — the nonlinear/volume version of the linear
    # legage/legamp levers. K_eff *= clip(1 + runscore_scale * runscore_z, 0.5, 2.0). 0 = off. Causal
    # (OOF predictions never see their own test year). Needs runscore.parquet.
    signal_exit_hold_runscore_scale: float = 0.0
    # PYRAMID-TO-REVEALED-WINNER (2026-06-21, micro-structure root cause: the winner/loser signal is
    # REVEALED post-entry via the early surge, NOT available pre-entry — so you cannot size-up at entry,
    # but you CAN add once a winner reveals). ADD `pyramid_add_units` of size at bar `pyramid_add_bars`
    # once the trade proves an early surge (return-since-fill >= pyramid_add_min_ret) — concentrate capital
    # on the revealed-winner cohort (probe: up-by-bar-3 => 93% wr / hold 40+). One position, growing. The
    # add fills at that bar's close (+slippage), exits with the base unit (size changes only pnl, NOT the
    # price-based exit timing). pnl_pct -> base_net + units*add_net, so per-symbol MDD (cumsum of pnl) sees
    # both the bigger pyramided winners AND the bigger losing-adds — a faithful PnL/MDD tradeoff. 0 = off.
    pyramid_add_units: float = 0.0
    pyramid_add_bars: int = 3
    pyramid_add_min_ret: float = 0.03
    # PROFIT-FLOOR for the extension hold (forensic G: signal-exit WINNERS never fade post-exit — 82-94%%
    # run >0.5 ATR in every regime — and even cut LOSERS bounce +6-14%%; a trade trending above its MA
    # tends to RUN). The hold defaults to in-profit-only (close>entry); set this slightly NEGATIVE (e.g.
    # -0.03) to also hold a near-breakeven trade that is still trending AND continuation-confirmed — catch
    # the about-to-bounce names the signal-exit cuts at a local low. 0.0 = in-profit only. Causal.
    signal_exit_hold_profit_floor: float = 0.0
    # SNR RUNNER-EXTENSION (2026-06-17, forensic t1930 decomp #3): in a clean-trend universe regime
    # (high signal-to-noise = broad market trending up with low cross-sectional dispersion) winners
    # run further, yet the velocity signal-exit cuts them on the first wobble. When set, the 'signal'
    # exit is SUPPRESSED (handed to the trail/overext arm) while the universe SNR at the current bar
    # >= exit_snr_extend_threshold AND the trade's peak gain >= exit_snr_min_gain (let only winners
    # ride; losers still exit). SNR = rolling(window) universe-mean return / cross-sectional dispersion
    # of per-symbol window returns. Causal (returns up to the bar). Decomp 2 log-only: +2.4u realized at
    # ~flat per-sym MDD. Non-masking: drops no trade. None = off. Orthogonal to the score3 veto (stackable).
    exit_snr_extend_threshold: float | None = None
    exit_snr_extend_window: int = 20
    exit_snr_min_gain: float = 0.0
    # GIVEBACK-AT-DEFER GUARD (2026-07-09, SNR_EXTEND_AUTOPSY §1c): only defer the signal-exit
    # when the trade has ALREADY given back >= this fraction (in ENTRY-price units, matching the
    # autopsy metric: (peak_high - close) / entry_fill) from its peak high at the defer bar.
    # Autopsy seed-42: defers fired within 0.10 of the peak (entry units) are active top-calls —
    # n=12, 83% end worse (-0.46u); defers after a deeper give-back are mid-wave pullbacks —
    # n=39, +2.97u incl. both mega-runners (VCI deferred at 0.294 giveback, VIC at 0.224).
    # 0.0 = off (legacy snr-extend behaviour, byte-identical). Causal (peak of highs up to bar i).
    exit_snr_defer_min_giveback: float = 0.0
    # Trailing give-back stop: close if price falls this fraction below the highest
    # high seen since entry (peak-to-current drawdown). None disables it. Locks in a
    # swing top instead of riding the move all the way back down (the 'give-back' leak).
    trailing_stop_pct: float | None = None
    # MFE-armed activation: the trailing stop stays DISARMED until peak gain since entry
    # reaches this fraction (e.g. 0.20 = +20% MFE). Below it the trail never fires, so normal
    # trades keep their full tail and only big winners get give-back protection — targeting the
    # forensic 'winners-turned-giveback' bucket (peak +37% -> exit +18%) without tail amputation.
    # None = trail armed from entry (legacy behaviour).
    trailing_activate_pct: float | None = None
    # STRUCTURE TRAIL (2026-06-19, TA-lens: the %-giveback trail is structure-BLIND and clips
    # runners on shallow pullbacks that don't break structure; ta_structure_analysis diagnostic:
    # a Donchian-20-low trail ~DOUBLES runner return 15.6%->28.3% by riding until structure breaks).
    # When set, the DEFAULT 'trailing_stop' tier exits on a STRUCTURAL break (close < the prior-N-bar
    # low) instead of the fixed %-drop-from-peak — ride the trend through wiggles. The protective arms
    # (overext/vol/dist) keep their tight %-band. None = off (legacy %-trail). Causal (prior N lows).
    trailing_struct_donch_win: int | None = None
    # When True the structure trail ALSO replaces the overext_trail tight %-band — let the
    # overextended runner ride the structure (the +12%-above-MA20 handoff is where most runner PnL
    # exits, so the %-clip there is the main giveback). Risk: overextended tops that round-trip give
    # back more (continuation-vs-top wall). Default False (overext keeps its tight lock).
    trailing_struct_apply_overext: bool = False
    # micro-diff gate: ride structure ONLY in a clean uptrend (close>SMA50 & SMA50 rising); when the
    # trend fades, revert to the tight %-trail so the fader exits sooner (cut the give-back cohort and
    # free the symbol earlier -> recover the blocked-entry velocity cost). False = ride always.
    trailing_struct_trend_only: bool = False
    # ENTRY-PEAK-CONDITIONED TRAILING ACTIVATION (EXP-B, forensic t1837): the entry mfe-head
    # (entry_ensemble3 = mfe_regression, column 'score4') predicts the forward 20-bar PEAK with
    # rank-IC +0.225, yet realized-pnl IC is only +0.033 — the fixed trailing_activate_pct (0.27)
    # throws that edge away: 417 trades peak in the 10-27% band, unprotected, and round-trip to a
    # signal-exit (giveback 45.3u). When mfe_act_k is set AND the signals carry 'score4', the trail
    # ARM threshold becomes PER-TRADE: act_pct = clip(mfe_act_k * score4[entry_signal_bar],
    # mfe_act_floor, mfe_act_cap). A trade the head predicts will peak LOW arms early (locks the
    # deadband); a predicted big runner keeps a high arm and rides the wide trail. This is the
    # ONLY lever that conditions the EXIT on the (otherwise unused) entry peak prediction.
    # None = off (fixed trailing_activate_pct). Causal (entry-bar prediction only).
    mfe_act_k: float | None = None
    mfe_act_floor: float = 0.08
    mfe_act_cap: float = 0.27
    # EXP2 TWO-TIER TRAILING (protect the 10-27% MFE give-back band): the single arm at
    # trailing_activate_pct (e.g. +27%) leaves trades peaking 10-27% MFE unprotected — 442 such
    # trades gave back 46.9u (forensic t1831). When BOTH are set, a SECOND looser tier arms at
    # trailing_tier2_activate_pct (lower MFE, e.g. +0.12) with a WIDER fixed band
    # trailing_tier2_stop_pct (e.g. 0.11) for peak_gain in [tier2_activate, trailing_activate);
    # above trailing_activate the tight default tier takes over so big runners keep the tight lock.
    # The tier-2 band is a FIXED width — it bypasses the ATR/score modulation (which sizes the
    # tight tier) — but still respects trend-intact suppression (only fires once the trend MA
    # breaks). None = off (single tier). Causal.
    trailing_tier2_activate_pct: float | None = None
    trailing_tier2_stop_pct: float | None = None
    # REGIME-ADAPTIVE TRAIL TIGHTNESS (2026-06-18, user "un-hardcode the fixed rule" thesis): the trail
    # band is a FIXED param (trailing_stop 0.08 / overext_trail 0.04); make it ADAPT to the consolidation
    # regime instead. When trailing_cons_tight_mult is set and the consolidation_score (count over
    # trailing_cons_window bars of daily-range < trailing_cons_range) >= trailing_cons_thr (a distribution/
    # sideways top), MULTIPLY the trail band by the mult (<1 = tighter) to lock profit at the real top
    # (cuts LATE_EXIT); in a clean trend (low consolidation) the loose default rides on (holds the
    # SOLD_THEN_RAN pullbacks). consolidation_score is the strongest top-vs-premature separator
    # (exit_struct_discriminate sep +0.82). None = off (fixed band). Causal (range over past bars).
    trailing_cons_tight_mult: float | None = None
    trailing_cons_window: int = 20
    trailing_cons_range: float = 0.02
    trailing_cons_thr: int = 2
    # WR-EXPRESSION unmask: gate the tier-2 trail to the PREDICTED-LOSER cohort only. Forensic
    # (champion 1830): high momentum-csrank entries split sharply by path — winners (83% reach
    # MFE>8%, exit via overext/trailing) vs losers (only 23% reach MFE>8%, peak ~5% then fade to
    # the signal-exit at -5.6%). A GLOBAL tier-2 clips the winners' tails (pnl 115->89). When
    # tier2_csr_threshold is set, the tier-2 band fires ONLY for trades whose entry-bar
    # cross-sectional momentum rank (entry_csr column from recombine) >= the threshold, so the
    # faster cut hits the predicted losers while high-MFE winners ride the wide default trail.
    # None = tier-2 (if configured) applies to all trades (legacy). Causal.
    tier2_csr_threshold: float | None = None
    # WR-EXPRESSION unmask (v2 — MAE discriminator): the predicted-loser cohort separates from
    # the winners far more cleanly on DRAWDOWN than on MFE: high-csrank winners barely dip
    # (avg MAE -2.4%) while high-csrank losers bleed to -8.6% MAE before the lagging signal
    # exit fires at -5.6%. A shallow hard stop fired ONLY on the high-csrank cohort
    # (entry_csr >= csr_hard_stop_threshold) cuts those losers earlier without touching the
    # winners (whose MAE never reaches the floor). Independent of exit_priority/hard_stop_pct.
    # Both None = off. Causal.
    csr_hard_stop_pct: float | None = None
    csr_hard_stop_threshold: float | None = None
    # ENTRY-CONDITIONED TRAILING (regime-conditional exit): forensics t1137 show 'cold'
    # entries (weak prior-N-bar momentum) rarely run (only 16.8% reach +15%) yet pop +5-10%
    # then give it all back to ~0 — and the global activate=+15% trail never arms to protect
    # them. When entry_momentum_window>0 AND cold_trailing_stop_pct is set, a position whose
    # prior-window runup at entry is <= entry_cold_threshold uses the (cold_activate, cold_trail)
    # pair (earlier/ tighter profit-lock) instead of the default pair; hot entries keep the loose
    # default trail so runners run. None/0 = disabled (identical to single-trail behaviour).
    entry_momentum_window: int = 0
    entry_cold_threshold: float = 0.0
    cold_trailing_activate_pct: float | None = None
    cold_trailing_stop_pct: float | None = None
    # STRENGTH-GATED POP LOCK (conditional exit): forensics t1137 — at the first bar a trade
    # reaches +pop_lock_arm_pct MFE, the trade's extension above its short MA there has rank-IC
    # +0.33 vs realized pnl. WEAK pops (close barely above the MA) fade to ~0 (39% fade); STRONG
    # pops (stretched, fast) continue (fade 21%). So lock a tight trail ONLY on the weak pops and
    # let strong pops keep the loose default trail. When pop_lock_arm_pct AND pop_lock_trail_pct
    # are set: at the first bar peak-gain crosses pop_lock_arm_pct, classify the position weak iff
    # close/SMA(pop_lock_ext_window)-1 < pop_lock_ext_threshold; weak positions then trail at
    # pop_lock_trail_pct (armed from that bar). None = disabled. Orthogonal to the entry-momentum
    # trail above; evaluate this first.
    pop_lock_arm_pct: float | None = None
    pop_lock_ext_window: int = 10
    pop_lock_ext_threshold: float = 0.0
    pop_lock_trail_pct: float | None = None
    # BREAKEVEN LOCK (2026-06-20, non-selection RIDE mechanic): once a trade's peak gain reaches
    # breakeven_lock_mfe (e.g. +0.12 = it PROVED itself a winner), install a HARD FLOOR at
    # entry*(1+breakeven_lock_offset) — exit if a later low breaks it. Unlike the %-trail (follows
    # the peak, clips runners on shallow pullbacks) this floors ONLY at breakeven, so a runner that
    # stays well above entry is never touched; it cuts ONLY the winner-round-trips-to-a-LOSS cohort
    # (cap their loss at ~0). Applies to EVERY trade that proved itself = NOT a selection gate (the
    # all-cohorts-net-positive wall is about gating ENTRIES, not flooring proven winners). Checked
    # ahead of exit_priority (like structural_stop). None = off. Causal.
    breakeven_lock_mfe: float | None = None
    breakeven_lock_offset: float = 0.0
    # VOL-ADAPTIVE HARD STOP (2026-06-20, vol_adaptive_forensic on 2482): the fixed-%% stop/downleg lets
    # LOW-vol losers bleed to -2.4 ATR before cutting while HI-vol cut at -1.6 ATR (a fixed %% = a
    # different RISK distance per stock). When set, the stop level = -hard_stop_atr_mult * (ATR14/close
    # at the ENTRY bar) — a per-stock, constant-RISK stop (cut every name at the same ATR distance).
    # Checked ahead of exit_priority (like csr/structural stops). None = off (fixed hard_stop_pct only).
    hard_stop_atr_mult: float | None = None
    exit_priority: list[str] = field(
        default_factory=lambda: ["hard_stop", "signal", "max_hold"]
    )  # order of exit conditions to check
    cost: CostModel = field(default_factory=CostModel)
    entry_bar_fill_type: str = "open_next"  # O t+1 | C t+1 | C t (entry only)
    exit_bar_fill_type: str | None = (
        None  # O t+1 | C t+1 (exit only); None = use entry_bar_fill_type
    )
    # PULLBACK LIMIT ENTRY (patient fill): when set, a buy signal at bar i places a limit at
    # close[i]*(1-entry_pullback_pct) and fills only if a later bar's LOW reaches it within
    # entry_pullback_window bars; otherwise the signal is SKIPPED (price ran away). Trade
    # forensics: the head buys the right names but mistimes the fill onto hot green spikes
    # (red-bar entries WR 70% vs thrust 54%) — this re-times the fill to weakness WITHOUT
    # changing which names it trades. None disables (canonical close/open fill). Causal:
    # a limit placed at signal time, executed only when price later trades down to it.
    entry_pullback_pct: float | None = None
    entry_pullback_window: int = 5
    # HYBRID FILL-IF-MISSED (2026-06-17, user's "chờ pullback; nếu giá chạy luôn thì vào luôn thay vì
    # bỏ"): when the patient limit never fills within the window the signal is normally DROPPED (the
    # runaway). When True, instead take the trade at-market at the window-end bar's close (a chased,
    # buffer-less fill) rather than skipping. Tests whether capturing the dropped runaways (+0.08 pnl
    # at-market but no price buffer) is composite-accretive on the full champion. None/False = drop
    # (legacy). Causal (window-end bar). No-op unless entry_pullback_pct is set.
    entry_pullback_fill_if_missed: bool = False
    # CANDLE-CONFIRMATION FILL (user 2026-06-19 'cụm màu/nến'): the pullback fills on weakness (a red
    # touch bar); candle_cluster.py shows the touch bar's candle splits the tail HARD — a GREEN
    # (dip-then-recover) touch bar = avg +12.9% / big-loss 2.7%, a still-falling red touch = +4.6% /
    # big-loss 9.2% (the knives). When True, require the limit-touch bar to close GREEN (close>=open =
    # the dip reversed intrabar); a red touch is SKIPPED and the scan continues for a later green touch
    # within the window (else the signal is skipped = knife avoided). Trades WR (red-dip 70%) for TAIL
    # (fewer knives) — composite-accretive where MDD is penalized (the crutch-free line). False = off.
    entry_pullback_confirm_reversal: bool = False
    # RUNAWAY-PHASE cap (2026-06-19, user "phân biệt lệnh runaway"): fill_if_missed buys at the
    # window-end close regardless of how far the runaway ran (+15% = a terrible extended fill = the
    # −400 catastrophe). When set, take the at-market fill ONLY if the window-end close is within
    # max_premium of the SIGNAL close — i.e. capture only the EARLY runaway phase (ran a little, small
    # buffer loss), SKIP the extreme runaways (ran too far). None = no cap (fill any runaway). Causal.
    fill_if_missed_max_premium: float | None = None
    # EXP1 CONDITIONAL PULLBACK (forensic t1831): 66% of fills land BELOW MA20 after a slow
    # (avg 24-bar) deep pullback wait — the model's weakest cohort (avg_pnl 0.057, big% 0.148) vs
    # quick shallow fills that stay above MA (avg 0.11, big% 0.22). When set (>0), the pending
    # pullback limit is CANCELLED (signal skipped) the first bar during the wait whose close falls
    # below its SMA(this window) — abort the fill once the trend has rolled over while waiting,
    # keeping only the healthy quick-dip fills. 0/None = off (legacy: fill on any low<=limit).
    entry_pullback_cancel_below_ma: int | None = None
    # EXP-D TREND-SCALED PULLBACK DEPTH (additive, forensic t1837 §E): the FIXED 4.5% pullback
    # lands 48% of fills BELOW MA20 (avg pnl .053, weak) vs the 6-12%-above-MA fills (avg .132,
    # best) — in a shallow-extension setup the deep limit drags the fill under the trend MA. When
    # entry_pullback_floor_ma>0, FLOOR the limit at SMA(window)*(1+buffer): the effective pullback
    # auto-shrinks when price is near its MA (keep the fill above the trend) and stays the full
    # depth when price is extended far above it. Same SIGNALS, better fill PRICE/timing (additive —
    # does not skip or add trades, only re-prices the limit). Clamped to <= close[i] (never a
    # limit above the signal close). 0/None = off (legacy fixed-depth pullback). Causal (MA at
    # the signal bar i only).
    entry_pullback_floor_ma: int | None = None
    entry_pullback_floor_buffer: float = 0.0
    # STRUCTURAL-LOW FILL ANCHOR (2026-06-18, decouple_oracle FILL-PRICE lever): the +32pp entry lever
    # the oracle measured is a FILL-PRICE gain (enter at the leg BOTTOM on the SAME legs), NOT a
    # which-to-buy gain (entry SELECTIVITY — features/raw-conviction — tested negative: cuts net-positive
    # trades). When entry_pullback_structural_lookback>0, the limit is DEEPENED toward the recent
    # structural swing-low: limit = min(fixed-depth limit, min(low over lookback)*(1+buffer)) — i.e. on a
    # leg whose base sits below the 4.5% dip, target the base instead of a fixed shallow dip, so the fill
    # lands nearer the leg bottom. NEVER shallower than the fixed pullback (min picks the lower price) and
    # clamped <= close[i]. Pair with entry_pullback_fill_if_missed=True to keep the legs that never
    # return to the base (fill at-market window-end) so this re-prices fills WITHOUT cutting trades.
    # 0/None = off (legacy fixed-depth). Causal (low window ending at the signal bar i).
    # TESTED NEGATIVE 2026-06-18 (entry_structfill_sweep.py): deepening toward the base is monotonically
    # worse (comp 486→116) — the deep limit rarely fills (the base is a knife or a round-trip when price
    # returns), so most signals fall through to fill_if_missed and buy the runaway HIGH (fillDisc goes
    # negative). The oracle's +32pp "enter at the bottom" is a foresight artifact; the 4.5% crutch +
    # skip-if-missed is the realizable optimum. Kept as an off-by-default lever (the harness reproduces it).
    entry_pullback_structural_lookback: int | None = None
    entry_pullback_structural_buffer: float = 0.0
    # WEEKLY-TREND DEEPEN (2026-06-19, diag_mtf_rs): big-loss KNIVES concentrate in weekly-down/flat
    # (wk_up separates knife 39% vs winner 51%, partial-IC +0.05 ORTHOGONAL to the entry score — a
    # genuinely-new multi-timeframe signal the daily heads miss). A weekly GATE fails (the weekly-down
    # cohort is net-positive = dip-buy edge), so MODULATE: when the bar is NOT in a weekly uptrend
    # (close<=SMA(wk_ma) or SMA(wk_ma) not rising over 5 bars), DEEPEN the pullback by (1+k) so the
    # knife fills lower / falls through, while weekly-up keeps the normal depth. None = off. Causal.
    entry_pullback_wk_deepen_k: float | None = None
    entry_pullback_wk_ma: int = 50
    entry_pullback_wk_deepen_cap: float = 2.0
    # RESUMPTION RE-ENTRY (2026-06-19, reentry_opportunity diag: 80% of trailing-exits RESUME — close
    # back above the exit bar's high within ~3 bars — and re-entering yields mean +3.10% / +1186u, REAL
    # continuation, NOT the failed chase (this re-enters an ESTABLISHED trend AFTER riding one leg, not
    # a fresh signal). When set, after a trailing exit, if within resume_reentry_win bars a close breaks
    # back above the exit bar's high (structure reasserts), re-enter AT-MARKET (bypass the pullback) to
    # recover the continuation the trail handed back. Premium-capped to avoid a gap chase. React, not
    # predict. Pair with trailing_struct_donch_win to ride it on structure. None = off. Causal.
    resume_reentry_win: int | None = None
    resume_reentry_max_premium: float = 0.15
    # CONVICTION-GRADED PULLBACK DEPTH (2026-06-18, user "un-hardcode" thesis; attacks the 48% fill-rate
    # = the fixed pullback REJECTS 52% of buy-intents). Instead of one fixed depth, SHALLOW the limit for
    # STRONG signals (high RSI14 + extended above MA20 = a clean-trend runner -> fill it at a reasonable
    # price instead of waiting for a deep dip that never comes) and keep the FULL depth for weak signals
    # (wait for the discount). depth = base * clip(1 - conv_k * strength, conv_floor, 1), strength∈[0,1]
    # from causal RSI/ext (past+current bar). Distinct from vol_scale (vol≠conviction) and from the binary
    # strength_skip (this is GRADED, still uses a limit). None/False = off (fixed depth). Causal.
    entry_pullback_conv_scale: bool = False
    entry_pullback_conv_k: float = 0.6
    entry_pullback_conv_floor: float = 0.4
    entry_pullback_conv_rsi_lo: float = 50.0
    entry_pullback_conv_ext_cap: float = 0.10
    # REGIME-ADAPTIVE conv gate (2026-06-18, top-1 2409 forensic): conv-graded pullback trades MDD
    # for PnL and the diagnostic (probe_conv_regime_sep) found its MDD damage is SEPARABLE by entry
    # vol — in the hi-vol regime conv adds ~0 PnL (-8u) but the MOST drawdown (-115u). When set,
    # DISABLE the conv discount (force full pullback depth) on bars whose causal 20-bar realized-vol
    # z-score exceeds this cap, so strong signals in choppy/high-vol regimes still wait for the dip.
    # None = off (conv applies in every regime, legacy). Causal (rolling z over conv_vol_lb).
    entry_pullback_conv_vol_z: float | None = None
    entry_pullback_conv_vol_lb: int = 60
    # Extension gate (loop iter-2): raw close/MA20-1 above which the conv shallow-fill is disabled
    # (extended setups must wait for the full pullback). None = off. See probe_conv_eff_sep.
    entry_pullback_conv_ext_gate: float | None = None
    # Combined-signal conviction (loop iter-5): blend efficiency + range-position into the conv
    # strength (not just rsi+ext), per the IC-0.184 OOF combo. False = legacy rsi/ext only.
    entry_pullback_conv_use_combo: bool = False
    entry_pullback_conv_combo_w: tuple = (0.35, 0.30, 0.20, 0.15)  # (rsi, ext, eff, rpos) weights
    # Head-z blend (loop iter-8): mix the ML entry-head prediction (escore_z, the masked forward-peak
    # signal IC_MFE 0.110) into the price combo strength — the most direct "ML drives the rule" path.
    # strength = (1-w)*price_combo + w*sigmoid(escore_z). 0 = price only. Causal (entry-bar z).
    entry_pullback_conv_head_w: float = 0.0
    # RAW head channel (2026-06-19 research D1): feed the RAW entry score (rank-IC 0.215 to fwd-peak)
    # into the head sigmoid instead of its causal-z (rank-IC 0.124 — z masks 42% of the signal at the
    # exact decision point). arg = a*(score - b); sigmoid centers the [0,0.8] raw range. Default off ->
    # uses escore_z (champion byte-identical). Only changes the head term's ARGUMENT, not its weight.
    entry_pullback_conv_head_raw: bool = False
    entry_pullback_conv_head_raw_a: float = 8.0
    entry_pullback_conv_head_raw_b: float = 0.20
    # MFE-head blend (loop iter-9): same path for score4 (mfe_bigmove head, IC_MFE 0.124 > main 0.110),
    # blended alongside the main head: strength = (1-hw-mw)*price + hw*sig(escore_z) + mw*sig(emfe_z).
    entry_pullback_conv_mfe_w: float = 0.0
    # CONTINUATION-head blend (2026-06-21): score3 is the ONLY entry head with IC vs REALIZED pnl > 0
    # (+0.114) — high-score3 = genuine runners (peak +16.8%% vs +10%%, run 23 vs 11 bars). The exit-hold
    # vein proved it's the best continuation predictor; fold it into the conv FILL too so a high-score3
    # (genuine-runner) setup fills SHALLOWER = enter earlier and capture more of its run. Same blend path:
    # strength += sw*sigmoid(s3z). 0 = off. Needs the 'score3' column (continuation ensemble head). Causal.
    entry_pullback_conv_score3_w: float = 0.0
    # RS-vs-MARKET conviction channel (2026-06-21, user: VNINDEX now available). RS line = close/VNINDEX;
    # rs_vsma = RS vs its own 50d trend (leader>0 / laggard<0). The STRONGEST per-trade separator found
    # (corr rs_sl5 +0.16, rs_vsma +0.11; strong-RS dips win 72%% vs weak 52%%) — a dip in a LEADER (strong
    # RS-vs-market) bounces, a dip in a LAGGARD is a knife. All-cohorts-net-positive (gate walled), so
    # fold it into the conv FILL (champion's law): a high-RS-vs-market setup fills SHALLOWER (catch the
    # leader's bounce). strength += rw*clip(0.5 + rs_vsma/(2*cap)). 0 = off. Needs the VNINDEX csv. Causal.
    entry_pullback_conv_rs_w: float = 0.0
    entry_pullback_conv_rs_cap: float = 0.08
    # WAVE-STRUCTURE conviction channel (2026-06-21): the same causal-zigzag leg structure that drives
    # the exit-hold also separates ENTRY quality (probe: cur_leg_amp/pos_in_leg_amp incr IC +0.14 vs
    # realized pnl). A dip in a YOUNG + STRONG-impulse leg is a runner the fixed 4.5% pullback often
    # misses -> blend a leg-strength [0,1] into the conv strength so it fills on a SHALLOWER dip; an
    # aged/weak leg keeps the full pullback. leg_strength = clip(0.5 + (legamp_norm - legage_norm)/
    # (2*cap), 0, 1). The first wave-structure input on the ENTRY side. 0 = off. Causal.
    entry_pullback_conv_leg_w: float = 0.0
    entry_pullback_conv_leg_pct: float = 0.06
    entry_pullback_conv_leg_cap: float = 1.0
    # VOLUME-CONFIRMATION conviction channel (2026-06-19, research-loop: volume = the only unwalled
    # real-supply signal). Blend a causal accumulation-minus-distribution balance into the conv
    # strength: over the trailing 20 bars count high-rel-volume strong-close (accum) minus high-rel-
    # volume weak-close (dist) bars; net accumulation -> higher conviction -> shallower pullback (catch
    # the demand-backed runner near the signal price). strength = (1-hw-mw-vw)*price + ... + vw*vol_acc.
    # vol_acc = clip(ad_balance/scale, 0, 1). 0 = off (champion). Pure price/volume, causal.
    entry_pullback_conv_vol_conf_w: float = 0.0
    entry_pullback_conv_vol_conf_scale: float = 8.0
    # VWAP / VOLUME-ZONE modulator (user proposal 2026-06-19: "giá tương đối của phiên predict so với
    # vùng giá có đột phá khối lượng"). dist_vwap = close / VWAP(window) - 1 (price vs where money
    # traded). As a HEAD-FEATURE it is masked (entry retrain -2.7..-7.7), but the champion's law is to
    # realize a weak signal as a MODULATOR of the deterministic fill: when price is BELOW the volume
    # zone (dist_vwap<0 = already a discount vs heavy-volume value) -> higher conviction -> shallower
    # pullback (don't wait, buy the value); when ABOVE -> deeper dip required. vwap_sig = clip(0.5 -
    # dist_vwap/(2*cap), 0, 1). Blended with weight vwap_w. 0 = off (champion). Pure price/volume, causal.
    entry_pullback_conv_vwap_w: float = 0.0
    entry_pullback_conv_vwap_win: int = 20
    entry_pullback_conv_vwap_cap: float = 0.10
    # EX-ANTE STRUCTURE conviction channels (2026-07-10, ENTRY_STRUCTURE_MAP lever #1): the two
    # largest ex-ante separators at the SIGNAL bar on gb_x08 are per-symbol SNR (mean/std of 20-bar
    # log returns; Q4 +15.8%/lệnh WR 68.8%, the only cohort positive in BOTH 2024 and 2026) and
    # dist-MA20 (Q4 +17.5% WR 77.2% vs Q0 +5.5%). Realize them the champion's way: as CONTINUOUS
    # conviction inputs (no gate — SELECTION WALL untouched). Each is z-normed per-symbol with the
    # system-standard causal 252/60 rolling z, squashed sigmoid(z) -> [0,1] and blended into the
    # conv strength with weight w (same pattern as the head/rs/leg channels; combo path only).
    # 0.0 = off (champion byte-identical). Causal (data up to the current bar only).
    entry_pullback_conv_snr_w: float = 0.0
    entry_pullback_conv_dma20_w: float = 0.0
    # COMBO trail modulator (loop iter-6): scale the trailing/overext give-back band by the entry-bar
    # combo (IC 0.248 vs realized MFE). trail_pct *= clip(1 + k*(combo-mid), floor, cap). k>0 widens
    # for high-combo runners. None = off. Pure price, causal. See probe_combo_mfe.
    trailing_combo_k: float | None = None
    trailing_combo_mid: float = 0.5
    trailing_combo_floor: float = 0.6
    trailing_combo_cap: float = 1.8
    # VOL-ADAPTIVE PULLBACK DEPTH (crutch-flex, 2026-06-16): the FIXED entry_pullback_pct (4.5%) is
    # inflexible — too deep for low-vol names (their signals rarely dip 4.5% -> dropped) and shallow
    # for high-vol names. When entry_pullback_vol_scale=True, the per-signal depth becomes
    # k * trailing_return_vol(window) (causal, this symbol's OWN vol), clamped to [lo, hi]. High-vol
    # names wait for a proportionally deeper dip, low-vol names a shallower one -> the dip-wait ADAPTS
    # per stock instead of one rigid global %. Mirrors the vol-normalized exit win. Off by default
    # (falls back to the fixed entry_pullback_pct). Causal: vol from PAST returns only (no leak).
    entry_pullback_vol_scale: bool = False
    entry_pullback_vol_k: float = 2.0
    entry_pullback_vol_window: int = 20
    entry_pullback_vol_lo: float = 0.02
    entry_pullback_vol_hi: float = 0.10
    # TOP-STRUCTURE ADAPTIVE PULLBACK DEPTH (C', 2026-06-18): the fixed dip can fill onto a TOPPING dip
    # (the right-shoulder/du-dinh trap). Diagnostics (top_struct_probe) found Dow lower-high + upper-wick
    # exhaustion are the structural TOP signals the entry set is blind to. When this is on, the per-bar
    # pullback depth is DEEPENED by k * max(0, causal-z(lower_high20)+causal-z(upwick_5)) (clamped to
    # _cap), so a buy in topping structure must wait for a DEEPER dip -> fills LOWER (lower in-trade MAE,
    # the user's "rebuy lower"), or is skipped if it never dips that far. Healthy-trend bars (score<=0)
    # keep the base depth. ADDS to the base/vol depth. Off by default. Causal (O/H/L/C only, z 252/60).
    entry_pullback_topstruct_scale: bool = False
    entry_pullback_topstruct_k: float = 0.012
    entry_pullback_topstruct_cap: float = 0.04
    entry_pullback_topstruct_thr: float = 0.0   # only deepen when the top-struct z-sum >= thr (leave
    #                                             healthy-trend entries at the base depth = preserve PnL)
    # REGIME-CONDITIONAL ENTRY/EXIT (downtrend_timing_review): the symbol's per-bar trend regime
    # is "downtrend" when close < SMA50 AND SMA50 is falling over the trailing 20 bars (the same
    # cut used in the forensic). Two asymmetric levers keyed off the regime at the SIGNAL bar:
    #   downtrend_skip_pullback: when True, a buy whose signal bar is in a downtrend SKIPS the
    #     patient pullback limit and fills immediately (close_next/open_next). Downtrend bounces are
    #     short — waiting for a 4.5% dip either fills onto the rollover or misses the bounce. Uptrend
    #     buys keep the pullback. No-op unless entry_pullback_pct is set.
    #   downtrend_hard_stop_pct: a TIGHTER hard stop (e.g. -0.08) applied ONLY to trades whose
    #     entry signal bar was in a downtrend — cut failed bounces fast instead of riding the slow
    #     downleg-12% force-gate to ~-3.3%. Checked ahead of exit_priority (like csr_hard_stop), so
    #     uptrend trades (which need room) are untouched. None = off. Both causal (SMA50 at past bars).
    downtrend_skip_pullback: bool = False
    downtrend_hard_stop_pct: float | None = None
    # STRUCTURAL STOP (EXP-3 pattern-anchored buffer, 2026-06-17): the 4.5% pullback's value is ~85%
    # an MDD price-BUFFER realised AT ENTRY (enter lower); EXP-1 proved no exit can recover it after
    # entering high. The only way to enter a strong NO-DIP runner with controlled drawdown is to take
    # the buffer from PRICE STRUCTURE: enter at-market and stop out if price breaks the pre-entry swing
    # low (the consolidation/base support). The per-trade loss is then capped at (entry - support) —
    # tight for a consolidation-breakout, supplying the same buffer a waited-for dip otherwise gives.
    # When set (>0), the stop level = min(low over the structural_stop_lookback bars ending at the
    # SIGNAL bar) * (1 - structural_stop_buffer); exit when a later LOW breaks it. Checked ahead of
    # exit_priority (like csr/downtrend stops). Causal (support from past bars only). None = off.
    structural_stop_lookback: int | None = None
    structural_stop_buffer: float = 0.0
    # MACD "LAST-LINE SHIELD" TOP/DECLINE EXIT (2026-06-18, user domain rule for the du-dinh round-top
    # the velocity exit head is blind to — VND forensic: head stayed z<0.2 at the +24% top, trailing
    # never armed <27%, so a +24% winner round-tripped to +3.5%). Force an exit once a 3-way decline
    # confirmation fires: MACD histogram < 0 (momentum turned down) AND close < SMA(ma_win) (price below
    # the trend) AND the current bar is RED (close < prior close). Checked ahead of exit_priority (like
    # structural_stop), so it bypasses the protect band / market-washout deferral that delayed the VND
    # fill to the bounce. Pure price/EMA, causal. None/False = off (legacy). MACD on close EMAs.
    macd_shield_enabled: bool = False
    macd_shield_fast: int = 12
    macd_shield_slow: int = 26
    macd_shield_signal: int = 9
    macd_shield_ma_win: int = 20
    # RSI-SLOPE last-line shield (research 2026-06-18: rsi_slope_5 topIC -0.155 = the STRONGEST top-signal,
    # beats MACD hist -0.124; the SHAPE/slope, NOT the point rsi -0.040 — the user's shape>point thesis).
    # Exit when RSI14 is falling sharply (rsi - rsi[slope_win] <= slope_thr) AND close < SMA(ma_win) =
    # momentum rollover below trend. Deep-gated by min_gain (protect the gain, don't clip early pullbacks).
    # Ahead of exit_priority (a backstop). Own reason 'rsi_shield'. Causal. None/False = off.
    rsi_shield_enabled: bool = False
    rsi_shield_slope_win: int = 5
    rsi_shield_slope_thr: float = -8.0
    rsi_shield_ma_win: int = 20
    rsi_shield_min_gain: float = 0.20
    # Only fire the shield once the trade's running PEAK gain >= this (protect a WINNER's top, don't
    # clip healthy early-trend pullbacks where MACD dips negative mid-run). 0.0 = fire on any pullback.
    macd_shield_min_gain: float = 0.0
    # STRENGTH-ROUTER SKIP-PULLBACK (2026-06-17, user accepts more per-symbol mdd for the +53u masked
    # pnl): the strength axis (RSI14 + ext-vs-MA20) is the one signal-bar feature with a real realized-
    # pnl gradient (RSI quintile win-rate 52%->79%, total-pnl 6.9->44.5 vs the FLAT primary score). The
    # strong early-wave signals run without the 4.5% dip so the pullback DROPS them. When
    # entry_strength_skip_pullback=True, a buy whose signal bar is STRONG (RSI14 >= strength_skip_rsi
    # AND ext-vs-MA20 >= strength_skip_ext) SKIPS the patient pullback and fills at-market (close_next)
    # instead of waiting/dropping — capturing the runner early; weak signals keep the deep buffer. This
    # trades pnl for per-symbol mdd along a Pareto (lower rsi thr = more skips = more pnl + more mdd).
    # Off by default. Causal (RSI/ext from past+current bar only). No-op unless entry_pullback_pct set.
    entry_strength_skip_pullback: bool = False
    strength_skip_rsi: float = 60.0
    strength_skip_ext: float = 0.0
    # SELL-INTO-STRENGTH limit exit (mirror of the entry pullback): when a sell triggers at bar i,
    # place a limit at close[i]*(1+exit_rally_pct) and fill at it if a later bar's HIGH reaches it
    # within exit_rally_window bars; else market-fill at the window-end close (must exit). Sells the
    # bounce for a better price instead of dumping at next close. None disables.
    exit_rally_pct: float | None = None
    exit_rally_window: int = 5
    # Restrict the sell-into-strength rally fill to ONLY the 'signal' exits (the ML/force sells that the
    # forensic shows dump into the bottom third of the window on a bounce); the mechanical overext/
    # trailing exits already sell high (pos 0.61) so a rally-wait there just bleeds on the miss-fallback.
    # False = apply to every exit (default). Off-effect only when exit_rally_pct is set.
    exit_rally_signal_only: bool = False
    # MARKET-REGIME EXIT GATE (regime-conditional exit): forensics t1151 — the ML/force
    # signal exit is value-destructive (-27.8 units, premature 56%). Nearly ALL of that
    # bleed is concentrated on days the WHOLE tape is in a sharp short-term drop: when the
    # equal-weight universe's `window`-bar return <= `threshold`, signal exits are premature
    # 65% and the names bounce +6% (beta dip, not idiosyncratic breakdown). When enabled,
    # the "signal" exit rule is SUPPRESSED on those market-drop dates (the position defers to
    # the trailing stop / hard stop / max_hold), so we stop dumping into market-wide washouts
    # and re-buying higher. Causal: the market return uses only data up to the current bar,
    # exactly like the signal itself. None/False = disabled (identical to legacy behaviour).
    exit_market_gate_enabled: bool = False
    exit_market_drop_window: int = 5
    exit_market_drop_threshold: float = -0.03
    # Gate mode: "cumret" (default) compares the window-bar cumulative market return to
    # threshold (a fixed % drop). "zscore" compares the window-return to its own trailing
    # distribution (z = (winret - mean) / std over exit_market_z_lookback bars) and fires
    # when z <= threshold — a vol-NORMALIZED washout that adapts across calm vs crisis
    # regimes (a -5% 5d drop means very different things in 2024 vs the 2020 crash). For
    # zscore mode set threshold to a NEGATIVE z (e.g. -1.0, -1.5).
    exit_market_drop_mode: str = "cumret"
    exit_market_z_lookback: int = 60
    # OVER-EXTENSION TOP EXIT (sell-the-top, the UP-slope mirror of the downleg gate):
    # behavioral audit of the champion's exit head (project_graded_risk_exit_gate) — the
    # trained head is PHASE-INVERTED: flat ~+0.12z at wave-tops (blind, never rings the bell
    # to sell) yet PEAKS +0.36z at price bottoms (panic-sells the low). Extension above the
    # SMA is the signal it ignores: close/SMA(20)-1 peaks AT the top (+1.15z) and a higher
    # extension leads a deeper forward drawdown (IC -0.16), while RSI/streak peak at tops but
    # don't predict the drop (overbought useless). When overext_ma_window>0, force a SELL once
    # close is >= overext_pct over its SMA(overext_ma_window) — optionally only on a reversal
    # bar (overext_reversal: current close < prior close) so we sell into the turn, not every
    # bar of a parabola. This is a DISTINCT exit rule (own "overext" reason) that runs OUTSIDE
    # the signal-exit age/market gates — those gates re-introduce the top-blindness on young
    # extended winners. 0 = disabled.
    overext_ma_window: int = 0
    overext_pct: float = 0.12
    # ATR-ADAPTIVE overext threshold (2026-06-19, TA-lens: "overextended" is RELATIVE to the stock's
    # own volatility, not a fixed 12%). The fixed 12% fires the tight overext_trail handoff TOO EARLY
    # on high-vol names (12% = normal noise -> clips the volatile runner) and too late on calm names.
    # When set, the effective overext extension = clip(overext_atr_mult * ATR14/close, lo, hi) so a
    # volatile stock must extend further (ride longer) and a calm one locks sooner. None = fixed pct.
    overext_atr_mult: float | None = None
    overext_atr_lo: float = 0.08
    overext_atr_hi: float = 0.24
    overext_reversal: bool = True
    # REGIME-CONDITIONAL overext (recover bull-year pnl): overext's only forensic cost is
    # clipping strong-uptrend runners (e.g. 2021) — in a fast trend the extension keeps
    # extending. When set, SKIP the overext sell while the SMA(overext_ma_window) is rising
    # at least this fraction over the trailing overext_skip_lookback bars (a strong uptrend ->
    # let it run); fire overext only in flat/choppy extension. None = always fire. Causal.
    overext_skip_ma_slope_pct: float | None = None
    overext_skip_lookback: int = 5
    # REVERSAL-CONFIRMED overext (sell AT a turn, not on the way up): the bool overext_reversal
    # only checks close<prev. overext_reversal_mode picks a stronger candlestick / EMA-cross
    # confirmation so the top-sell waits for an actual reversal. Overrides overext_reversal when
    # not None. Causal — uses bar i and earlier only. Modes:
    #   "none"        fire on extension alone (overext_reversal=False)
    #   "down1"       close[i] < close[i-1]              (== overext_reversal=True)
    #   "strong_down" red bar AND close drop >= overext_strong_down_pct vs prior close
    #   "engulf2"     bearish engulfing (prev up body, current down body engulfs it)
    #   "three_down"  three lower closes in a row (momentum rolled over)
    #   "ema_cross"   close crosses below EMA(overext_ema_span) this bar
    #   "bear_div"    bearish momentum divergence: TsRank(close,w) - TsRank(macd_hist,w) >=
    #                 overext_div_threshold (price ranking higher than momentum = a top tell).
    #                 DATA: divergence has ~0 IC unconditionally but IC −0.066..−0.079 vs fwd
    #                 return ONLY when extended (ext>10-15%) — so it must be PAIRED with a low
    #                 overext_pct (the conditioning); fires the faded mid-winners (peak +9-10%,
    #                 below the +14% overext) at their divergence-confirmed turn. Causal.
    overext_reversal_mode: str | None = None
    overext_ema_span: int = 10
    overext_strong_down_pct: float = 0.02
    overext_div_window: int = 20
    overext_div_threshold: float = 0.0
    overext_floor_pct: float | None = None
    # MARKET-REGIME overext skip (regime ensemble-lite): in a strong MARKET bull, SKIP the
    # overext top-sell so winners run with the tape (overext's only forensic cost is clipping
    # bull runners; champion pnl is ~59% from bull years). Distinct from overext_skip_ma_slope_pct
    # (per-SYMBOL uptrend). Reuses the EW-market proxy (mirror of the market gates). Off = legacy.
    overext_skip_bull_enabled: bool = False
    overext_bull_window: int = 5
    overext_bull_threshold: float = 1.0
    overext_bull_mode: str = "zscore"
    overext_bull_z_lookback: int = 60
    # DISTRIBUTION-CONDITIONED overext threshold (sequence-geometry refines the mechanical sell):
    # the model is BLIND to multi-bar distribution/accumulation structure. At the overext trigger
    # bar, ad_balance_20 (count of accumulation candles minus distribution candles over the last 20
    # bars, where dist = high-vol weak-close, accum = high-vol strong-close) has IC +0.148 vs the
    # forward 20-bar continuation: distribution-heavy extensions have LESS upside left (sell EARLIER
    # = lower overext_pct) while accumulation-heavy extensions keep running (hold LATER = higher
    # overext_pct). When enabled, the effective overext_pct becomes:
    #   overext_pct + overext_dist_slope * ad_balance_20   (clamped to [pct_lo, pct_hi])
    # so positive ad_balance (accumulation) RAISES the bar (hold) and negative (distribution) LOWERS
    # it (sell sooner). Pure price/volume, causal (bar i and the trailing 20 only). None = off.
    overext_dist_slope: float | None = None
    overext_dist_pct_lo: float = 0.08
    overext_dist_pct_hi: float = 0.16
    # ASYMMETRIC distribution floor: when ad_balance_20 <= overext_dist_neg_thresh (distribution
    # heavy / weak internals), use overext_dist_neg_pct as the (lower) extension threshold so the
    # over-extended-into-distribution winner is sold EARLIER; otherwise the normal overext_pct
    # applies (accum runners are NEVER deferred or clipped). One-sided => no runner-clipping cost.
    overext_dist_neg_thresh: float | None = None
    overext_dist_neg_pct: float = 0.09
    # RE-ENTRY COOLDOWN (whipsaw fix): after a LOSING exit on a symbol, block a new buy in that
    # symbol for this many bars. Forensic: downleg loss-exits rebuy the same name ~6% higher within
    # 15 bars 60.9% of the time (25.5u giveup). 0 = off. Causal (only past exits).
    reentry_cooldown_bars: int = 0
    # RE-ENTRY PREMIUM CAP (anti gap-up-chase, 2026-06-17): within this many bars after ANY exit on
    # a symbol, only re-enter at/below prior_exit_price*(1+pct). Unlike the cooldown (which blocks the
    # whole leg), this still re-enters but ONLY at a non-chase price — the existing pullback wait fills
    # at the capped limit; if price never dips that low in the window the signal is skipped (no chase).
    # Attacks the ~5.5% same-wave rebuy premium directly. None = off. Causal (only past exits).
    reentry_max_premium_pct: float | None = None
    reentry_max_premium_bars: int = 15
    # OVEREXT -> TIGHT-TRAIL (extend-longer): when set, an overext trigger does NOT hard-sell;
    # it arms a tight trailing give-back band (this pct) so a runner rides higher while a fader is
    # cut near the overext level. Captures the post-overext continuation (forensic: 76% of overext
    # tops keep running, 48.5% of profit-takes leave +16.7% upside) WITHOUT predicting fade-vs-run.
    # None = legacy hard overext sell. Causal.
    overext_trail_pct: float | None = None
    # VOL-ADAPTIVE TRAILING (magnitude axis): the fixed trailing_stop_pct whipsaws low-vol
    # names and gives back too much on high-vol ones — the same %-drop means different things
    # per stock. When set, the trail width becomes trailing_atr_mult * ATR14/close (clamped to
    # [trailing_atr_floor, trailing_atr_cap]), so a stock's own volatility sizes its give-back
    # band. None = use the fixed trailing_stop_pct (legacy). Causal (ATR up to the current bar).
    trailing_atr_mult: float | None = None
    trailing_atr_floor: float = 0.04
    trailing_atr_cap: float = 0.16
    # VOL-EXPANSION PROTECTIVE TRAIL (EXP-C, the MAGNITUDE axis): forensics (project_exit_predict_
    # ceiling, t1804 chain) — the exit head cannot predict DIRECTION (|IC|<=0.04) but VOL predicts
    # forward-drawdown MAGNITUDE (ATR IC -0.165). So instead of TIMING a top, react to RISK: when a
    # held position's ATR14/close z spikes >= vol_spike_z_threshold (a regime change / distribution
    # blow-off / instability), the expected give-back jumps, so ARM a tight trailing band
    # (vol_spike_trail_pct) regardless of the activation threshold AND overriding the trend-intact
    # suppression (a vol spike mid-uptrend is exactly the parabolic-climax top to protect). Gated to
    # in-profit positions (peak_gain >= vol_spike_min_gain) so it locks winners' gains and does NOT
    # knife a young underwater loser at a capitulation low (high vol there too). z = causal
    # per-symbol 252/60 z of ATR14/close. None = off. Causal (ATR up to the current bar). Own
    # exit reason 'vol_protect'.
    vol_spike_z_threshold: float | None = None
    vol_spike_trail_pct: float = 0.06
    vol_spike_min_gain: float = 0.05
    # DISTRIBUTION-ARMED PROTECTIVE TRAIL (the REALIZABILITY axis, forensic exit_giveback_1875):
    # the mechanical exits sell round-trippers (gave back >10% of a >10% MFE) ~6 bars AFTER the
    # in-trade peak while selling the legitimate riders ~1 bar after — and the two cohorts SEPARATE
    # on the 20-bar accumulation-minus-distribution balance at the peak (round-trippers ad_balance
    # ~1.2 vs riders ~2.3; IC(ad_balance, giveback) -0.06, IC(bull_trap, giveback) +0.16). So when a
    # held IN-PROFIT position (peak_gain >= dist_arm_min_gain) shows a distribution-heavy 20-bar
    # internal (ad_balance_20 <= dist_arm_neg_thresh), ARM a tight give-back band (dist_arm_trail_pct)
    # — protect the realizable gain on names whose internals say the advance is being distributed,
    # WITHOUT clipping the accumulation-strong riders (high ad_balance, left on the loose trail).
    # Orthogonal to vol_protect (vol MAGNITUDE) and downleg (price rollover): this reads the
    # multi-bar volume/close STRUCTURE the per-bar model is blind to. Causal (20-bar trailing sum up
    # to bar i). None = off. Own exit reason 'dist_protect'.
    dist_arm_neg_thresh: float | None = None
    dist_arm_trail_pct: float = 0.06
    dist_arm_min_gain: float = 0.10
    # ML-SWING-TOP TRAILING (Phase D, 2026-06-17): an ML swing-top detector (per-bar prob in the
    # 'swing_top_prob' signal column, trained on 'top within K bars' of a medium-term ATR-zigzag)
    # ARMS a tight give-back band (ml_swing_trail_pct) on an in-profit position when prob >= thr —
    # sell near the swing top to enable a lower rebuy, with LOW in-trade drawdown (the downleg/overext
    # gate as the always-on backstop). Mirrors overext_trail but ML-triggered. Causal (prob uses only
    # causal features, walk-forward OOS). None = off. Own exit reason 'ml_swing'.
    ml_swing_trail_thr: float | None = None
    ml_swing_trail_pct: float = 0.06
    ml_swing_min_gain: float = 0.05
    # DEAD-MONEY / STALE EXIT (per-bar lever, 2026-06-17): exit an IN-PROFIT trade that has not made
    # a NEW high in `stale_exit_bars` bars — cut "dead money" (peaked early then drifts to the exit),
    # boosting per-bar return (the LARGEST composite weight 0.27) WITHOUT predicting tops (purely
    # reactive: bars since last new high). Only when peak_gain >= stale_exit_min_gain (don't time-stop
    # a young/losing trade — leave that to the stops). Lowest priority. None = off. Reason 'stale'.
    stale_exit_bars: int | None = None
    stale_exit_min_gain: float = 0.05
    # ML-MAGNITUDE-SCALED TRAILING: the exit head's only genuine signal is forward drawdown
    # MAGNITUDE (IC ~0.27 vs realized |drawdown|), NOT direction — so use it to size the
    # give-back band, not to TIME the sell. When trailing_score_k is set, the trail width is
    # multiplied by clip(1 - trailing_score_k * z(exit_score), score_mult_floor, score_mult_cap):
    # a HIGH exit-score bar (big predicted drop) tightens the trail (protect), a LOW one widens
    # it (let the runner run). z(exit_score) is a causal per-symbol 252/60 z (same as recombine).
    # None = off (no modulation). Causal.
    trailing_score_k: float | None = None
    trailing_score_mult_floor: float = 0.6
    trailing_score_mult_cap: float = 1.5
    # BOTTOM-ride coupling (decoupling breakthrough 2026-06-19): widen the trailing give-back band
    # for trades ENTERED at a genuine zigzag bottom (high score6 from a zigzag_pivot bottom head).
    # A bottom entry has low MAE -> it can afford to ride to a better exit (the decoupling test:
    # true-bottom entry + aggressive exit is synergistic, capture 0.22->0.80, MAE controlled).
    # trail_pct *= clip(1 + k*(score6[entry] - mid), floor, cap). None = off. Causal (entry bar).
    # Needs entry_ensemble5 -> score6 in the signals frame (the bottom head, even z-gated off).
    bot_ride_k: float | None = None
    bot_ride_mid: float = 0.5
    bot_ride_floor: float = 1.0
    bot_ride_cap: float = 2.0
    # BOTTOM-deepen coupling (entry side, decoupling 2026-06-19): when a PRE-sided bottom head
    # (score6, one_sided='pre') anticipates a bottom, DEEPEN the required pullback so the fill lands
    # nearer the true low -> enter the SAME trade cheaper (oracle-entry capture 0.22->0.54 with the
    # REAL exit). Conditioned on the prediction so the deeper limit is reached (vs uniform deepening
    # which misses / buys runaways high). _depth *= clip(1 + k*score6[i], 1.0, cap). None = off.
    bot_deepen_k: float | None = None
    bot_deepen_cap: float = 2.5
    # BOTTOM-shallow coupling (quality-gated shallow fill, wave-start 2026-07-09): the INVERSE
    # knob. Wave starts only offer ~2% dips (median retest -2% below thrust-close; the champion's
    # 4.5% buffer exists in just 30% of them) — so when the bottom-structure head (score6) is HIGH
    # for a buy, SHRINK the required pullback so the genuine wave-start shallow dip fills; a low
    # score6 keeps the full depth (the knife filter). bot_deepen's clip floor 1.0 can only deepen,
    # never shallow — this is the gated shallow path. Applied AFTER bot_deepen (if both are set the
    # deepened depth is what gets shrunk; sweeps run deepen off). Conditioned on the causal 252/60
    # per-symbol z of score6 (same scale as the ensemble5 union z_threshold; the raw head score is
    # a penalized return, mean ~-0.25, so raw would never fire):
    # _depth *= max(floor, 1 - k*max(z6[i], 0)). 0 = off (byte-identical).
    bot_shallow_k: float = 0.0
    bot_shallow_floor: float = 0.4
    # CONFIRMATION-BREAKOUT B-ENTRY (wave-start final mechanism, 2026-07-09): every re-pricing
    # of the existing pending book failed via occupancy reshuffle (anchor/deepen/shallow/
    # miss-capture all negative) — so instead of changing core fills, ADD an entry when the
    # per-symbol slot is IDLE. Wave anatomy (W3-A4): buying the close that breaks the recent
    # thrust high captures 89% of real wave starts at +10.6%/21bar with 41% false-fill (vs 86%
    # false-fill for raw -2% limits) — the best EV/placement anchor tested. Trigger on a
    # no-position bar (checked BEFORE the core signal branch, so it preempts a same-bar core
    # signal): causal z of the bottom-structure head (score6, same 252/60 z as the ensemble5
    # union z_threshold) >= entry_bchannel_z AND close[i] > max(high[i-lb..i-1]) (thrust-high
    # confirmation) AND close[i] < SMA(below_ma) (the below-MA20 wave-start zone where the
    # main channel emits no signals -> slot usually free). Enters at-market close_next;
    # respects reentry_cooldown_bars. Optional B-only structural stop: exit when the low
    # breaks min(low over stop_lookback bars ending at the trigger bar), next-bar close fill,
    # reason 'bch_stop'. All other exits inherit the champion machinery unchanged.
    # None = off (byte-identical).
    entry_bchannel_z: float | None = None
    entry_bchannel_break_lookback: int = 4
    entry_bchannel_below_ma: int | None = 20
    entry_bchannel_stop_lookback: int | None = None
    # Conditional TAKE-PROFIT (loop iter-4): LOCK the small-mover pop the trailing gives back (capture
    # 0.22 root cause: median trade pops to MFE 15% then round-trips). Resting limit -> fills AT the
    # target intraday (no give-back, unlike trailing). Per-trade target conditioned on the mfe head
    # (score4 = predicted fwd peak): low predicted peak -> tight TP (lock the pop); high -> loose (ride
    # the runner). tp_pct = clip(take_profit_k * score4[entry], floor, cap). None = off. floor==cap = flat.
    take_profit_k: float | None = None
    take_profit_floor: float = 0.06
    take_profit_cap: float = 0.30
    # TREND-INTACT TRAIL SUPPRESSION (regime-conditional, mirror of overext_skip_bull): the fixed
    # trailing stop fires on ANY peak give-back, so it sells ~80% of the time MID-UPTREND (price
    # still above a rising MA) on a healthy pullback that then continues +5-13%. When
    # trailing_skip_above_ma>0, SUPPRESS the trailing_stop while close > SMA(window) AND that SMA is
    # rising — hold through the healthy pullback; the position is still protected by overext /
    # downleg / belowma20. The trail re-activates once price breaks the trend MA or the MA flattens.
    # The overext_trail handoff is exempt (a deliberate tight ride). None/0 = off. Causal.
    trailing_skip_above_ma: int | None = None
    trailing_skip_ma_slope_lb: int = 5
    # TREND-BREAK PROFIT-LOCK (chartist 2026-06-19: a pro TA locks a winner the moment price closes
    # below the trend MA — the mechanical % trail re-arms on the break then waits for ANOTHER %-drop
    # = ~4-7 bars late past the peak, giving back ~38% of MFE). FLEXIBLE/conditional: only on an
    # in-profit position (peak gain >= trend_break_lock_gain), sell IMMEDIATELY at the first close
    # below SMA(trend_break_lock_ma) instead of re-arming the lagging % trail. This protects winners
    # on a real trend break WITHOUT churning small/choppy trades (the failure of a global MA-swap).
    # None = off (champion). Causal. Own reason 'trend_break'.
    trend_break_lock_gain: float | None = None
    trend_break_lock_ma: int = 20
    # MARKET-REGIME ENTRY GATE (the per-symbol model is BLIND to the tape): entry forensics
    # (young_loser_hunt) — buys are monotonically better as the equal-weight market's recent
    # return rises (WR 0.51 @ market −6% 5d → 0.69 @ +9%; the model buys into a falling tape).
    # When enabled, a BUY is SKIPPED on dates where the EW-universe `window`-bar return (cumret)
    # or its z-score (zscore mode) is <= threshold — i.e. don't open new risk into a market
    # washout. Mirror of exit_market_gate; reuses the same causal market proxy. Off = legacy.
    entry_market_gate_enabled: bool = False
    entry_market_window: int = 5
    entry_market_threshold: float = -0.03
    entry_market_mode: str = "cumret"
    entry_market_z_lookback: int = 60
    # Absolute-drop floor UNIONED with the primary (zscore) gate: the adaptive zscore
    # DESENSITIZES during a sustained bear (its trailing distribution shifts down, so a deep
    # grind no longer reads as extreme) — mdd_attribution: the residual max-drawdown is a
    # cluster of entries in the June-2022 bear the zscore gate let through. When set, ALSO skip
    # buys on dates whose cumulative `window`-bar market return <= this absolute floor (a fixed
    # deep-drop catch the zscore misses). None = zscore gate only.
    entry_market_abs_floor: float | None = None
    # MARKET CHOP ENTRY GATE (EXP-A, forensic t1837): 49% of trades exit within 8 bars at net ~0
    # (804 trades, -1.9u) — pure cost+capital churn. It is REGIME-driven: fast-trade share & loss
    # concentrate in the choppy/bear years (2022 0.58, 2026 0.71 fast-share, all net-negative) and
    # is net-POSITIVE only in trending 2020-21; the per-symbol entry head is blind to it (escore on
    # fast-fails == escore on slow-winners, 0.260 vs 0.263). The washout gates catch CRASHES, not
    # CHOP. When enabled, a BUY is SKIPPED on dates where the EW-universe trend EFFICIENCY ratio
    # over `window` bars (|net index change| / sum|daily index change|, Kaufman ER in [0,1]) is
    # <= threshold — a directionless/whippy tape (a strong up OR down trend both score HIGH ER and
    # still trade). Causal (EW index up to the signal bar). Off = legacy.
    entry_market_chop_enabled: bool = False
    entry_market_chop_window: int = 20
    entry_market_chop_threshold: float = 0.30


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float  # effective fill price (after slippage)
    exit_date: pd.Timestamp
    exit_price: float  # effective fill price (after slippage)
    holding_days: int
    pnl_pct: float  # net of all costs, expressed as fraction (0.05 == +5%)
    exit_reason: str  # 'signal' | 'max_hold' | 'hard_stop' | 'trailing_stop' | 'end_of_data'
    entry_signal_date: pd.Timestamp  # the bar whose signal triggered entry (entry_date-1)
    # Portfolio-tier fields (default to the legacy per-symbol single-unit long position).
    side: int = 1  # +1 long, -1 short
    weight: float = 1.0  # target portfolio weight at entry
    notional: float = 0.0  # capital allocated at entry (0 == legacy unweighted run)


# Side-channel log of confirmation-breakout B-entries (symbol + signal/entry dates) for the
# current run_backtest call. The Trade schema is hash-frozen by the regression goldens
# (trades CSV byte-parity), so the B tag lives OUTSIDE the trade record; diagnostics join it
# to the exported trades on (symbol, entry_date). Cleared at the top of run_backtest ONLY
# when the B-channel is armed; empty and untouched otherwise.
BCH_ENTRY_LOG: list[dict] = []
MISSED_SIGNAL_LOG: list[dict] = []


def _ensure_signals(signals: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "signal"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing columns: {sorted(missing)}")
    out = signals.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["signal"] = out["signal"].astype(int)
    return out


def _ensure_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing columns: {sorted(missing)}")
    out = ohlcv.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)


def _run_symbol(
    sym: str,
    bars: pd.DataFrame,
    sig_map: dict[pd.Timestamp, int],
    cfg: EngineConfig,
    market_drop_dates: set[pd.Timestamp] | None = None,
    market_weak_dates: set[pd.Timestamp] | None = None,
    market_bull_dates: set[pd.Timestamp] | None = None,
    xscore: np.ndarray | None = None,
    escore: np.ndarray | None = None,
    ecsr: np.ndarray | None = None,
    emfe: np.ndarray | None = None,
    market_chop_dates: set[pd.Timestamp] | None = None,
    es3: np.ndarray | None = None,
    market_snr_dates: set[pd.Timestamp] | None = None,
    swing_prob: np.ndarray | None = None,
    bot: np.ndarray | None = None,
) -> list[Trade]:
    trades: list[Trade] = []
    n = len(bars)
    if n < 1:
        return trades

    dates = bars["date"].to_numpy()
    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)

    # Vol-adaptive trailing: precompute causal ATR14/close ratio for this symbol.
    atr_ratio = None
    if (cfg.trailing_atr_mult is not None or cfg.overext_atr_mult is not None
            or cfg.hard_stop_atr_mult is not None) and n > 1:
        prev_c = np.concatenate([[closes[0]], closes[:-1]])
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
        atr_ratio = atr / np.where(closes == 0.0, 1e-9, closes)

    # Protect-RELEASE vol (ATR(release_vol_window)/close): sizes the vol-scaled give-back that releases
    # the signal-exit protect band early. Computed only when the release lever is armed. Causal.
    protect_release_vol = None
    if cfg.signal_exit_protect_release_drop_k is not None and n > 1:
        prev_cr = np.concatenate([[closes[0]], closes[:-1]])
        trr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_cr), np.abs(lows - prev_cr)))
        wr = max(2, int(cfg.signal_exit_protect_release_vol_window))
        atrr = pd.Series(trr).rolling(wr, min_periods=1).mean().to_numpy()
        protect_release_vol = atrr / np.where(closes == 0.0, 1e-9, closes)

    # EXP-C vol-expansion protective trail: causal per-symbol z of ATR14/close (same 252/60 window
    # as the recombine z). A high z = the position's volatility just expanded vs its own norm.
    vol_z = None
    if cfg.vol_spike_z_threshold is not None and n > 1:
        prev_c = np.concatenate([[closes[0]], closes[:-1]])
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
        ar = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy() / np.where(closes == 0.0, 1e-9, closes)
        s = pd.Series(ar)
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        vol_z = ((s - mu) / (sd + 1e-9)).to_numpy()

    # ML-magnitude-scaled trailing: causal per-symbol z of the exit head's prediction (same
    # 252/60 window as the recombine z). NaN where the score is missing (pre-test bars) -> no
    # modulation there.
    xscore_z = None
    if cfg.trailing_score_k is not None and xscore is not None and n > 1:
        s = pd.Series(xscore, dtype=float)
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        xscore_z = ((s - mu) / (sd + 1e-9)).to_numpy()

    # EXP3 coherence veto: causal per-symbol z of the ENTRY head's score (same 252/60 window as
    # the recombine z). NaN on pre-test bars -> no veto there.
    emfe_z = None
    if cfg.entry_pullback_conv_mfe_w > 0 and emfe is not None and n > 1:
        s = pd.Series(emfe, dtype=float)
        emfe_z = ((s - s.rolling(252, min_periods=60).mean())
                  / (s.rolling(252, min_periods=60).std() + 1e-9)).to_numpy()

    escore_z = None
    if ((cfg.signal_exit_skip_if_entry_z is not None or cfg.entry_pullback_conv_head_w > 0)
            and escore is not None and n > 1):
        s = pd.Series(escore, dtype=float)
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        escore_z = ((s - mu) / (sd + 1e-9)).to_numpy()

    # BOTTOM-shallow gate: causal per-symbol z of the bottom-structure head (score6), same
    # 252/60 window as the recombine/union z so "high" means the same thing as the ensemble5
    # z_threshold. NOTE: raw score6 is NOT reusable here — the bottom_structure head predicts a
    # penalized return (mean ~-0.25), so max(raw,0) would never fire; bot_deepen keeps its raw
    # convention (built for the prob-like zigzag head) untouched. Shared by the B-channel
    # breakout entry (same z scale as its quality gate). Only computed when a consuming knob
    # is on -> off = byte-identical.
    bot_z = None
    if (cfg.bot_shallow_k > 0 or cfg.entry_bchannel_z is not None) and bot is not None and n > 1:
        s = pd.Series(bot, dtype=float)
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        bot_z = ((s - mu) / (sd + 1e-9)).to_numpy()

    # score3 continuation veto: causal per-symbol z of the continuation head (same 252/60 window).
    s3z = None
    if ((cfg.signal_exit_skip_if_score3_z is not None
         or cfg.signal_exit_hold_min_score3_z is not None
         or cfg.entry_pullback_conv_score3_w > 0) and es3 is not None and n > 1):
        s = pd.Series(es3, dtype=float)
        mu = s.rolling(252, min_periods=60).mean()
        sd = s.rolling(252, min_periods=60).std()
        s3z = ((s - mu) / (sd + 1e-9)).to_numpy()

    # EXP1 conditional pullback: causal SMA used to cancel a pending pullback limit once the
    # trend rolls over (close < SMA) while waiting for the fill.
    pb_ma = None
    if (cfg.entry_pullback_pct is not None and cfg.entry_pullback_cancel_below_ma is not None
            and cfg.entry_pullback_cancel_below_ma > 0 and n > 1):
        pb_ma = pd.Series(closes).rolling(
            int(cfg.entry_pullback_cancel_below_ma), min_periods=1).mean().to_numpy()

    # B-channel below-MA context: causal SMA defining the wave-start zone (below the trend MA
    # the core channel is quiet in). Only computed when the B-channel is armed.
    bch_ma = None
    if (cfg.entry_bchannel_z is not None and cfg.entry_bchannel_below_ma is not None
            and cfg.entry_bchannel_below_ma > 0 and n > 1):
        bch_ma = pd.Series(closes).rolling(
            int(cfg.entry_bchannel_below_ma), min_periods=1).mean().to_numpy()

    # EXP-D trend-scaled pullback depth: causal SMA used to FLOOR the pullback limit (keep the
    # fill above the trend MA — auto-shallows the effective pullback when price is near the MA).
    pb_floor_ma = None
    if (cfg.entry_pullback_pct is not None and cfg.entry_pullback_floor_ma is not None
            and cfg.entry_pullback_floor_ma > 0 and n > 1):
        pb_floor_ma = pd.Series(closes).rolling(
            int(cfg.entry_pullback_floor_ma), min_periods=1).mean().to_numpy()

    # Structural-low fill anchor: causal rolling MIN of lows over the lookback ending at the signal
    # bar -> the recent swing-low / base of the current leg. Used to DEEPEN the pullback limit toward
    # the leg bottom (the oracle FILL-PRICE lever). Off by default.
    pb_struct_low = None
    if (cfg.entry_pullback_pct is not None and cfg.entry_pullback_structural_lookback is not None
            and cfg.entry_pullback_structural_lookback > 0 and n > 1):
        pb_struct_low = pd.Series(lows).rolling(
            int(cfg.entry_pullback_structural_lookback), min_periods=1).min().to_numpy()

    # Regime-adaptive trail: per-bar consolidation_score (count of last trailing_cons_window bars whose
    # daily range < trailing_cons_range) -> tighten the trail band in a distribution/sideways regime.
    cons_trail = None
    if cfg.trailing_cons_tight_mult is not None and n > 1:
        _hlp = (highs - lows) / np.where(closes > 0, closes, np.nan)
        cons_trail = pd.Series((_hlp < cfg.trailing_cons_range).astype(float)).rolling(
            int(cfg.trailing_cons_window), min_periods=3).sum().to_numpy()

    # Vol-adaptive pullback depth (causal trailing return-vol * k, clamped). Off by default.
    pb_vol_depth = None
    if cfg.entry_pullback_pct is not None and cfg.entry_pullback_vol_scale and n > 1:
        _rv = pd.Series(closes).pct_change().rolling(
            int(cfg.entry_pullback_vol_window), min_periods=2).std()
        pb_vol_depth = (cfg.entry_pullback_vol_k * _rv).clip(
            lower=cfg.entry_pullback_vol_lo, upper=cfg.entry_pullback_vol_hi).to_numpy()

    # Top-structure adaptive EXTRA pullback depth (C', causal z 252/60 of Dow lower-high + upper-wick).
    pb_wk_down = None
    if cfg.entry_pullback_wk_deepen_k is not None:
        _wm = max(int(cfg.entry_pullback_wk_ma), 5)
        _wma = pd.Series(closes).rolling(_wm, min_periods=_wm).mean()
        _wk_up = (pd.Series(closes) > _wma) & (_wma > _wma.shift(5))
        pb_wk_down = (~_wk_up).to_numpy()

    pb_topstruct_depth = None
    if cfg.entry_pullback_pct is not None and cfg.entry_pullback_topstruct_scale and n > 1:
        _hi = pd.Series(highs); _lo = pd.Series(lows)
        _mx = np.maximum(opens, closes)
        _h20 = _hi.rolling(20, min_periods=1).max()
        _lower_high = (_h20 < _h20.shift(20)).astype(float)
        _rng = (_hi - _lo).replace(0.0, np.nan)
        _upwick = ((_hi - _mx) / _rng).rolling(5, min_periods=1).mean()

        def _cz252(x):
            mu = x.rolling(252, min_periods=60).mean(); sd = x.rolling(252, min_periods=60).std()
            return (x - mu) / (sd + 1e-9)
        _ts = (_cz252(_lower_high) + _cz252(_upwick)).clip(lower=0.0)
        if cfg.entry_pullback_topstruct_thr > 0.0:
            _ts = _ts.where(_ts >= cfg.entry_pullback_topstruct_thr, 0.0)
        pb_topstruct_depth = (cfg.entry_pullback_topstruct_k * _ts).clip(
            upper=cfg.entry_pullback_topstruct_cap).fillna(0.0).to_numpy()

    # MACD last-line shield (user 2026-06-18): per-bar 3-way decline-confirmation mask. Causal —
    # MACD hist from close EMAs, SMA(ma_win), and a red bar (close < prior close).
    macd_shield_arr = None
    if cfg.macd_shield_enabled and n > 1:
        _c = pd.Series(closes)
        _macd = _c.ewm(span=cfg.macd_shield_fast, adjust=False).mean() - \
            _c.ewm(span=cfg.macd_shield_slow, adjust=False).mean()
        _hist = (_macd - _macd.ewm(span=cfg.macd_shield_signal, adjust=False).mean()).to_numpy()
        _smN = _c.rolling(int(cfg.macd_shield_ma_win), min_periods=1).mean().to_numpy()
        _red = (_c < _c.shift(1)).to_numpy()
        macd_shield_arr = (_hist < 0.0) & (closes < _smN) & _red

    # RSI-slope shield (research: rsi_slope_5 = strongest top-signal). Wilder RSI14 + slope_win-bar slope.
    rsi_shield_arr = None
    if cfg.rsi_shield_enabled and n > 1:
        _c = pd.Series(closes); _d = _c.diff()
        _up = _d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        _dn = (-_d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        _rsi = 100.0 - 100.0 / (1.0 + _up / (_dn + 1e-9))
        _rslope = (_rsi - _rsi.shift(int(cfg.rsi_shield_slope_win))).to_numpy()
        _rma = _c.rolling(int(cfg.rsi_shield_ma_win), min_periods=1).mean().to_numpy()
        rsi_shield_arr = (_rslope <= cfg.rsi_shield_slope_thr) & (closes < _rma)

    # Regime-conditional entry/exit: precompute the per-bar "downtrend" mask (causal) —
    # close < SMA50 AND SMA50 falling over the trailing 20 bars. NaN warmup -> not downtrend
    # (keep the default uptrend behaviour). Only when a downtrend lever is active.
    downtrend_arr = None
    if (cfg.downtrend_skip_pullback or cfg.downtrend_hard_stop_pct is not None) and n > 1:
        sma50 = pd.Series(closes).rolling(50, min_periods=50).mean()
        slope50 = (sma50 / sma50.shift(20) - 1.0).to_numpy()
        sma50v = sma50.to_numpy()
        dt = (closes < sma50v) & (slope50 <= 0.0)
        dt[np.isnan(sma50v) | np.isnan(slope50)] = False
        downtrend_arr = dt

    # Trend-intact trail suppression: precompute "price above a RISING trend MA" (causal).
    trend_up = None
    if cfg.trailing_skip_above_ma is not None and cfg.trailing_skip_above_ma > 0 and n > 1:
        w = int(cfg.trailing_skip_above_ma)
        sma = pd.Series(closes).rolling(w, min_periods=1).mean()
        slope = sma - sma.shift(int(cfg.trailing_skip_ma_slope_lb))
        trend_up = ((closes > sma.to_numpy()) & (slope.to_numpy() > 0))

    # Vol-adaptive extension hold (signal_exit_hold_ext_atr): per-stock extension in ATR units + a
    # trend gate, so the signal-exit can be held on un-stretched trending winners (sell the stretch,
    # not the mean). Causal ATR14/MA up to the bar.
    ext_atr_hold = None
    hold_trend_ok = None
    atr_hold_ratio = None
    if cfg.signal_exit_hold_ext_atr is not None and n > 1:
        _wm = max(2, int(cfg.signal_exit_hold_ma))
        _ma = pd.Series(closes).rolling(_wm, min_periods=_wm).mean()
        _pc = np.concatenate([[closes[0]], closes[:-1]])
        _trh = np.maximum(highs - lows, np.maximum(np.abs(highs - _pc), np.abs(lows - _pc)))
        atr_hold_ratio = (pd.Series(_trh).rolling(14, min_periods=1).mean().to_numpy()
                          / np.where(closes == 0.0, 1e-9, closes))
        _mav = _ma.to_numpy()
        _exth = closes / np.where(np.isnan(_mav) | (_mav == 0), np.nan, _mav) - 1.0
        ext_atr_hold = _exth / np.where((atr_hold_ratio <= 0) | np.isnan(atr_hold_ratio), np.nan, atr_hold_ratio)
        _slh = (_ma - _ma.shift(5)).to_numpy()
        hold_trend_ok = (closes > _mav) & (_slh > 0)

    # RS-vs-market adaptive hold: per-bar RS-vs-VNINDEX (leader>0 / laggard<0) to scale the hold depth.
    rs_vsma_arr = None
    if (cfg.signal_exit_hold_rs_scale > 0 or cfg.signal_exit_release_rs_scale > 0) and n > 1:
        _vni = _load_vnindex()
        if _vni is not None:
            _vd = pd.to_datetime(pd.Series(dates)).dt.normalize()
            _vni_a = _vd.map(_vni).to_numpy(dtype=float)
            _rsl = closes / np.where((_vni_a <= 0) | np.isnan(_vni_a), np.nan, _vni_a)
            _rss = pd.Series(_rsl)
            _f = cfg.signal_exit_hold_rs_feature
            if _f == "rs_sl5":
                _rsf = (_rss / _rss.shift(5) - 1.0).to_numpy()
            elif _f == "rs_sl20":
                _rsf = (_rss / _rss.shift(20) - 1.0).to_numpy()
            elif _f == "rs_nh":
                _rsf = (_rss / _rss.rolling(60, min_periods=20).max() - 1.0).to_numpy()
            else:  # rs_vsma (RS vs its 50d trend)
                _rma = _rss.rolling(50, min_periods=20).mean().to_numpy()
                _rsf = _rsl / np.where(np.isnan(_rma) | (_rma == 0), np.nan, _rma) - 1.0
            rs_vsma_arr = np.nan_to_num(_rsf, nan=0.0)

    # MARKET big-trend adaptive hold: per-bar VNINDEX trend strength (same across symbols on a date),
    # causal (close + rolling stats up to the bar). Centered at 0 so clip(1 + mkt_scale*mkt, 0.5, 2.0)
    # holds deeper in a strong broad trend and exits sooner in a weak/late-wave market.
    mkt_trend_arr = None
    if cfg.signal_exit_hold_mkt_scale > 0 and n > 1:
        _vni = _load_vnindex()
        if _vni is not None:
            _vd = pd.to_datetime(pd.Series(dates)).dt.normalize()
            _vc = pd.Series(_vd.map(_vni).to_numpy(dtype=float))
            _mf = cfg.signal_exit_hold_mkt_feature
            if _mf == "ma200":
                _mma = _vc.rolling(200, min_periods=50).mean()
                _mt = (_vc / _mma - 1.0).to_numpy()
            elif _mf == "ma50":  # shakeout_vs_top forensic (2026-07-11): mkt>MA50 is the strongest
                #                  shakeout-vs-top discriminator (AUC 0.78 overall, 0.59-0.94
                #                  within-year) — faster regime flip than MA100/200.
                _mma = _vc.rolling(50, min_periods=20).mean()
                _mt = (_vc / _mma - 1.0).to_numpy()
            elif _mf == "pos120":
                _hi = _vc.rolling(120, min_periods=40).max()
                _lo = _vc.rolling(120, min_periods=40).min()
                _mt = (2.0 * (_vc - _lo) / (_hi - _lo + 1e-9) - 1.0).to_numpy()   # [-1,1], +1=at wave high
            elif _mf == "dd120":
                _mt = (_vc / _vc.rolling(120, min_periods=40).max() - 1.0).to_numpy()  # <=0 drawdown from peak
            else:  # ma100 (big-trend strength vs 100d MA)
                _mma = _vc.rolling(100, min_periods=30).mean()
                _mt = (_vc / _mma - 1.0).to_numpy()
            mkt_trend_arr = np.nan_to_num(_mt, nan=0.0)

    # MARKET-REGIME signal-exit skip (shakeout_vs_top): per-bar VNINDEX/MA(N) - 1 (bull >= 0), causal.
    mkt_skip_arr = None
    if cfg.signal_exit_skip_if_mkt_above_ma is not None and n > 1:
        _vni = _load_vnindex()
        if _vni is not None:
            _vd = pd.to_datetime(pd.Series(dates)).dt.normalize()
            _vc = pd.Series(_vd.map(_vni).to_numpy(dtype=float))
            _w = int(cfg.signal_exit_skip_if_mkt_above_ma)
            _mma = _vc.rolling(_w, min_periods=max(2, _w // 2)).mean()
            mkt_skip_arr = np.nan_to_num((_vc / _mma - 1.0).to_numpy(), nan=-1.0)

    # Wave-structure (current-leg age) adaptive hold: causal zigzag leg-age, normalized vs the
    # stock's typical up-leg length. Young leg -> hold deeper; aged leg -> exit sooner.
    legage_norm_arr = None
    legamp_norm_arr = None
    if (cfg.signal_exit_hold_legage_scale > 0 or cfg.signal_exit_hold_legamp_scale > 0) and n > 1:
        legage_norm_arr, legamp_norm_arr = _causal_leg_age(closes, cfg.signal_exit_hold_legage_pct)

    # 'Remaining-run' ML head hold modulator: per-bar causal OOF run-score-z for this symbol.
    runscore_arr = None
    if cfg.signal_exit_hold_runscore_scale > 0 and n > 1:
        _rsm = _load_runscore()
        if _rsm is not None and sym in _rsm:
            _ds = pd.to_datetime(pd.Series(dates)).dt.tz_localize(None).dt.normalize()
            runscore_arr = np.nan_to_num(_rsm[sym].reindex(_ds.to_numpy()).to_numpy(dtype=float), nan=0.0)

    # TREND-BREAK profit-lock MA (chartist trend-break exit) — precompute SMA(window).
    tb_ma_arr = None
    if cfg.trend_break_lock_gain is not None and n > 1:
        tb_ma_arr = pd.Series(closes).rolling(int(cfg.trend_break_lock_ma), min_periods=1).mean().to_numpy()

    # SLOW per-stock trend for the signal-exit protect band (regime-masking 2026-06-17): the protect's
    # default trend uses the SHORT MA10 (trailing_skip_above_ma), so a pullback breaking MA10 releases
    # the protect and the signal exit cuts a winner whose LONGER trend is still intact. When
    # signal_exit_protect_ma is set, the protect uses "close > rising MA(that window)" instead — holds
    # winners through deep pullbacks in a sustained up-trend (a per-stock regime gate: above the slow
    # MA = bull = hold; below = cut). None = use the default trend_up. Causal.
    trend_up_slow = None
    if cfg.signal_exit_protect_ma is not None and cfg.signal_exit_protect_ma > 0 and n > 1:
        ws = int(cfg.signal_exit_protect_ma)
        smas = pd.Series(closes).rolling(ws, min_periods=1).mean()
        slopes = smas - smas.shift(int(cfg.trailing_skip_ma_slope_lb))
        trend_up_slow = ((closes > smas.to_numpy()) & (slopes.to_numpy() > 0))

    # Strength-router skip-pullback: causal RSI14 + ext-vs-MA20 per bar; a STRONG signal bar skips the
    # pullback and fills at-market (capture the early-wave runner the dip-wait would otherwise drop).
    strength_skip_arr = None
    if cfg.entry_strength_skip_pullback and cfg.entry_pullback_pct is not None and n > 1:
        _c = pd.Series(closes)
        _d = _c.diff()
        _gain = _d.clip(lower=0).rolling(14, min_periods=14).mean()
        _loss = (-_d.clip(upper=0)).rolling(14, min_periods=14).mean()
        _rsi = (100.0 - 100.0 / (1.0 + _gain / (_loss + 1e-9))).to_numpy()
        _ext = (_c / _c.rolling(20, min_periods=20).mean() - 1.0).to_numpy()
        strength_skip_arr = (_rsi >= cfg.strength_skip_rsi) & (_ext >= cfg.strength_skip_ext)
        strength_skip_arr[np.isnan(_rsi) | np.isnan(_ext)] = False

    # Conviction-graded pullback depth: per-bar multiplier (<1 = shallower) scaling the limit depth DOWN
    # for strong signals (high RSI14 + extended above MA20), full depth (mult=1) for weak/warmup. Causal.
    pb_conv_mult = None
    if cfg.entry_pullback_conv_scale and cfg.entry_pullback_pct is not None and n > 1:
        _c2 = pd.Series(closes); _d2 = _c2.diff()
        _g2 = _d2.clip(lower=0).rolling(14, min_periods=14).mean()
        _l2 = (-_d2.clip(upper=0)).rolling(14, min_periods=14).mean()
        _rsi2 = (100.0 - 100.0 / (1.0 + _g2 / (_l2 + 1e-9))).to_numpy()
        _ext2 = (_c2 / _c2.rolling(20, min_periods=20).mean() - 1.0).to_numpy()
        _rsi_s = np.clip((_rsi2 - cfg.entry_pullback_conv_rsi_lo)
                         / (100.0 - cfg.entry_pullback_conv_rsi_lo + 1e-9), 0.0, 1.0)
        _ext_s = np.clip(_ext2 / (cfg.entry_pullback_conv_ext_cap + 1e-9), 0.0, 1.0)
        # Combined-signal conviction (2026-06-19, loop iter-5): a 3-fold-OOF blend of eff/ext/rsi/
        # range-pos predicts realized pnl at IC 0.184 (> any single feature), with a wide-but-all-
        # positive decile spread (D1 +3.5 .. D10 +12.5 — no skippable cohort, so MODULATE not skip).
        # conv already uses rsi+ext; blend in efficiency (trend cleanliness) + range-position so the
        # shallow fill concentrates on the cleanest, best-positioned setups. Pure price, causal.
        if cfg.entry_pullback_conv_use_combo:
            _eff2 = ((_c2 - _c2.shift(10)).abs()
                     / (_c2.diff().abs().rolling(10, min_periods=10).sum() + 1e-9)).to_numpy()
            _lo20 = pd.Series(lows).rolling(20, min_periods=10).min().to_numpy()
            _hi20 = pd.Series(highs).rolling(20, min_periods=10).max().to_numpy()
            _rpos2 = np.clip((closes - _lo20) / (_hi20 - _lo20 + 1e-9), 0.0, 1.0)
            _eff_s = np.clip(_eff2, 0.0, 1.0)
            _cw = cfg.entry_pullback_conv_combo_w  # (rsi, ext, eff, rpos)
            _strength = _cw[0] * _rsi_s + _cw[1] * _ext_s + _cw[2] * _eff_s + _cw[3] * _rpos2
            _strength = np.nan_to_num(_strength, nan=0.0)
            # Blend the ML head predictions (z -> [0,1] via sigmoid) into the price strength. Main
            # head (escore_z, IC_MFE 0.110) and/or mfe head (emfe_z = score4, IC_MFE 0.124).
            _hw = cfg.entry_pullback_conv_head_w if escore_z is not None else 0.0
            _mw = cfg.entry_pullback_conv_mfe_w if emfe_z is not None else 0.0
            _vw = cfg.entry_pullback_conv_vol_conf_w
            _ww = cfg.entry_pullback_conv_vwap_w
            _sw = cfg.entry_pullback_conv_score3_w if s3z is not None else 0.0
            _rw = cfg.entry_pullback_conv_rs_w
            _lw = cfg.entry_pullback_conv_leg_w
            _snw = cfg.entry_pullback_conv_snr_w
            _dmw = cfg.entry_pullback_conv_dma20_w
            _snr_z = None
            if _snw > 0:
                # per-symbol SNR = mean/std of 20-bar log returns (em_01/em_04 definition), then
                # the system-standard causal 252/60 rolling z. Data <= current bar only.
                _lr2 = pd.Series(np.log(np.where(closes > 0, closes, np.nan))).diff()
                _snr_raw = (_lr2.rolling(20, min_periods=20).mean()
                            / (_lr2.rolling(20, min_periods=20).std() + 1e-12))
                _snr_z = ((_snr_raw - _snr_raw.rolling(252, min_periods=60).mean())
                          / (_snr_raw.rolling(252, min_periods=60).std() + 1e-9)).to_numpy()
            _dma_z = None
            if _dmw > 0:
                # dist-MA20 = close/MA20 - 1 (= _ext2 already computed above), 252/60 causal z.
                _dma_s = pd.Series(_ext2)
                _dma_z = ((_dma_s - _dma_s.rolling(252, min_periods=60).mean())
                          / (_dma_s.rolling(252, min_periods=60).std() + 1e-9)).to_numpy()
            _leg_strength = None
            if _lw > 0:
                _lan, _lamp = _causal_leg_age(closes, cfg.entry_pullback_conv_leg_pct)
                _leg_strength = np.nan_to_num(np.clip(
                    0.5 + (_lamp - _lan) / (2.0 * cfg.entry_pullback_conv_leg_cap + 1e-9), 0.0, 1.0), nan=0.5)
            _rs_strength = None
            if _rw > 0:
                _vni = _load_vnindex()
                if _vni is not None:
                    _vd = pd.to_datetime(pd.Series(dates)).dt.normalize()
                    _vni_a = _vd.map(_vni).to_numpy(dtype=float)
                    _rsl = closes / np.where((_vni_a <= 0) | np.isnan(_vni_a), np.nan, _vni_a)
                    _rsma = pd.Series(_rsl).rolling(50, min_periods=20).mean().to_numpy()
                    _rsvsma = _rsl / np.where(np.isnan(_rsma) | (_rsma == 0), np.nan, _rsma) - 1.0
                    _rs_strength = np.nan_to_num(np.clip(
                        0.5 + _rsvsma / (2.0 * cfg.entry_pullback_conv_rs_cap + 1e-9), 0.0, 1.0), nan=0.5)
                if _rs_strength is None:
                    _rw = 0.0
            _vwap_sig = None
            if _ww > 0:
                _vv2 = bars["volume"].to_numpy(dtype=float)
                _vwin = max(2, int(cfg.entry_pullback_conv_vwap_win))
                _pv = pd.Series(closes * _vv2).rolling(_vwin, min_periods=_vwin // 2).sum().to_numpy()
                _vs = pd.Series(_vv2).rolling(_vwin, min_periods=_vwin // 2).sum().to_numpy()
                _vwap = _pv / np.where((_vs <= 0) | np.isnan(_vs), np.nan, _vs)
                _dvw = closes / _vwap - 1.0   # >0 price above the heavy-volume zone, <0 below (value)
                _cap = cfg.entry_pullback_conv_vwap_cap + 1e-9
                _vwap_sig = np.nan_to_num(np.clip(0.5 - _dvw / (2.0 * _cap), 0.0, 1.0), nan=0.0)
            _vol_acc = None
            if _vw > 0:
                _vv = bars["volume"].to_numpy(dtype=float)
                _rng = np.where((highs - lows) <= 0, 1e-9, highs - lows)
                _clpos = (closes - lows) / _rng
                _vavg = pd.Series(_vv).rolling(20, min_periods=10).mean().to_numpy()
                _volr = _vv / np.where((_vavg <= 0) | np.isnan(_vavg), np.nan, _vavg)
                _accb = ((_volr > 1.2) & (_clpos > 0.55)).astype(float)
                _distb = ((_volr > 1.2) & (_clpos < 0.45)).astype(float)
                _adb = (pd.Series(_accb).rolling(20, min_periods=10).sum()
                        - pd.Series(_distb).rolling(20, min_periods=10).sum()).to_numpy()
                _vol_acc = np.nan_to_num(
                    np.clip(_adb / (cfg.entry_pullback_conv_vol_conf_scale + 1e-9), 0.0, 1.0), nan=0.0)
            if (_hw > 0 or _mw > 0 or _vw > 0 or _ww > 0 or _sw > 0 or _rw > 0 or _lw > 0
                    or _snw > 0 or _dmw > 0):
                _blend = (1.0 - _hw - _mw - _vw - _ww - _sw - _rw - _lw - _snw - _dmw) * _strength
                if _hw > 0:
                    if cfg.entry_pullback_conv_head_raw and escore is not None:
                        _esr = np.nan_to_num(np.asarray(escore, dtype=float),
                                             nan=cfg.entry_pullback_conv_head_raw_b)
                        _harg = cfg.entry_pullback_conv_head_raw_a * (_esr - cfg.entry_pullback_conv_head_raw_b)
                    else:
                        _harg = np.nan_to_num(escore_z, nan=0.0)
                    _blend = _blend + _hw / (1.0 + np.exp(-_harg))
                if _mw > 0:
                    _blend = _blend + _mw / (1.0 + np.exp(-np.nan_to_num(emfe_z, nan=0.0)))
                if _vw > 0 and _vol_acc is not None:
                    _blend = _blend + _vw * _vol_acc
                if _ww > 0 and _vwap_sig is not None:
                    _blend = _blend + _ww * _vwap_sig
                if _sw > 0 and s3z is not None:
                    _blend = _blend + _sw / (1.0 + np.exp(-np.nan_to_num(s3z, nan=0.0)))
                if _rw > 0 and _rs_strength is not None:
                    _blend = _blend + _rw * _rs_strength
                if _lw > 0 and _leg_strength is not None:
                    _blend = _blend + _lw * _leg_strength
                if _snw > 0 and _snr_z is not None:
                    _blend = _blend + _snw / (1.0 + np.exp(-np.nan_to_num(_snr_z, nan=0.0)))
                if _dmw > 0 and _dma_z is not None:
                    _blend = _blend + _dmw / (1.0 + np.exp(-np.nan_to_num(_dma_z, nan=0.0)))
                _strength = _blend
        else:
            _strength = 0.5 * _rsi_s + 0.5 * _ext_s
        pb_conv_mult = np.clip(1.0 - cfg.entry_pullback_conv_k * _strength,
                               cfg.entry_pullback_conv_floor, 1.0)
        pb_conv_mult[np.isnan(_rsi2) | np.isnan(_ext2)] = 1.0   # warmup -> full depth
        # Regime-adaptive gate: in high-vol bars force full depth (disable the conv discount), since
        # the conv shallow-fill there is ~all drawdown and ~no PnL (probe_conv_regime_sep, 2409).
        if cfg.entry_pullback_conv_vol_z is not None:
            _rv2 = _c2.pct_change().rolling(20, min_periods=10).std()
            _lb = max(int(cfg.entry_pullback_conv_vol_lb), 5)
            _rvm = _rv2.rolling(_lb, min_periods=10).mean()
            _rvs = _rv2.rolling(_lb, min_periods=10).std()
            _volz = ((_rv2 - _rvm) / (_rvs + 1e-9)).to_numpy()
            pb_conv_mult[_volz > cfg.entry_pullback_conv_vol_z] = 1.0
        # Extension gate (2026-06-19, loop iter-2): the conv `strength` treats high extension as
        # CONVICTION (ext -> shallower fill) but the marginal-conv diagnostic (probe_conv_eff_sep)
        # found conv on ALREADY very-extended bars is ~0 PnL / +drawdown (chasing). Disable the conv
        # discount above a raw extension cap so extended setups must still wait for the full pullback.
        if cfg.entry_pullback_conv_ext_gate is not None:
            pb_conv_mult[_ext2 > cfg.entry_pullback_conv_ext_gate] = 1.0

    # COMBO exit-modulator array (2026-06-19, loop iter-6): the IC-0.184 entry-quality combo
    # (eff/ext/rsi/range-pos) also ranks realized MFE strongly (IC 0.248, Q1 12% .. Q5 21% MFE,
    # probe_combo_mfe) — high-combo trades genuinely run FURTHER. Build the same [0,1] combo per bar
    # so the trail can be WIDENED for high-combo entries (let the runner ride) / tightened for low.
    # Causal, pure price. Independent of the conv entry modulator.
    combo_arr = None
    if cfg.trailing_combo_k is not None:
        _cc = pd.Series(closes)
        _g = _cc.diff().clip(lower=0).rolling(14, min_periods=14).mean()
        _l = (-_cc.diff().clip(upper=0)).rolling(14, min_periods=14).mean()
        _rsi_c = (100.0 - 100.0 / (1.0 + _g / (_l + 1e-9))).to_numpy()
        _ext_c = (_cc / _cc.rolling(20, min_periods=20).mean() - 1.0).to_numpy()
        _eff_c = ((_cc - _cc.shift(10)).abs()
                  / (_cc.diff().abs().rolling(10, min_periods=10).sum() + 1e-9)).to_numpy()
        _lo = pd.Series(lows).rolling(20, min_periods=10).min().to_numpy()
        _hi = pd.Series(highs).rolling(20, min_periods=10).max().to_numpy()
        _rp = np.clip((closes - _lo) / (_hi - _lo + 1e-9), 0.0, 1.0)
        _rsi_cs = np.clip((_rsi_c - cfg.entry_pullback_conv_rsi_lo)
                          / (100.0 - cfg.entry_pullback_conv_rsi_lo + 1e-9), 0.0, 1.0)
        _ext_cs = np.clip(_ext_c / (cfg.entry_pullback_conv_ext_cap + 1e-9), 0.0, 1.0)
        combo_arr = 0.35 * _rsi_cs + 0.30 * _ext_cs + 0.20 * np.clip(_eff_c, 0, 1) + 0.15 * _rp
        combo_arr = np.nan_to_num(combo_arr, nan=cfg.trailing_combo_mid)

    # Reversal-confirmed overext: precompute the causal EMA once if the ema_cross mode is used.
    overext_ema = None
    if cfg.overext_ma_window > 0 and cfg.overext_reversal_mode == "ema_cross":
        overext_ema = pd.Series(closes).ewm(span=cfg.overext_ema_span, adjust=False).mean().to_numpy()

    # Bearish-divergence overext: causal TsRank(close,w) - TsRank(macd_hist,w), normalized to
    # [-1,1]. Positive = price ranks higher than momentum within the window = bearish divergence.
    overext_div = None
    if cfg.overext_ma_window > 0 and cfg.overext_reversal_mode == "bear_div":
        cs = pd.Series(closes)
        macd_hist = (cs.ewm(span=12, adjust=False).mean() - cs.ewm(span=26, adjust=False).mean())
        macd_hist = (macd_hist - macd_hist.ewm(span=9, adjust=False).mean())
        w = max(2, int(cfg.overext_div_window))
        rank = lambda s: s.rolling(w, min_periods=w).apply(
            lambda a: (a.argsort().argsort()[-1] + 1) / len(a), raw=True)
        overext_div = (rank(cs) - rank(macd_hist)).to_numpy()

    # DISTRIBUTION-CONDITIONED overext: causal accumulation-minus-distribution balance over the
    # trailing 20 bars. dist candle = high relative volume + weak close (close in lower 45% of the
    # bar range); accum candle = high relative volume + strong close (upper 45%). The 20-bar sum of
    # (accum - dist) shifts the effective overext_pct. Pure price/volume, causal.
    overext_ad_balance = None
    if (cfg.overext_dist_slope is not None
            or cfg.overext_dist_neg_thresh is not None
            or cfg.dist_arm_neg_thresh is not None):
        vol = bars["volume"].to_numpy(dtype=float)
        rng = highs - lows
        rng = np.where(rng <= 0, 1e-9, rng)
        clpos = (closes - lows) / rng
        vavg = pd.Series(vol).rolling(20, min_periods=10).mean().to_numpy()
        volr = vol / np.where((vavg <= 0) | np.isnan(vavg), np.nan, vavg)
        dist_bar = ((volr > 1.2) & (clpos < 0.45)).astype(float)
        accum_bar = ((volr > 1.2) & (clpos > 0.55)).astype(float)
        s_acc = pd.Series(accum_bar).rolling(20, min_periods=10).sum()
        s_dist = pd.Series(dist_bar).rolling(20, min_periods=10).sum()
        overext_ad_balance = (s_acc - s_dist).to_numpy()

    # STRUCTURE TRAIL: prior-N-bar low (Donchian lower band, causal: excludes current bar).
    don_low = None
    struct_trend_ok = None
    if cfg.trailing_struct_donch_win is not None:
        _w = max(int(cfg.trailing_struct_donch_win), 2)
        don_low = pd.Series(lows).rolling(_w, min_periods=_w).min().shift(1).to_numpy()
        if cfg.trailing_struct_trend_only:
            # micro-diff: the structure-ride helps in clean uptrends, gives back in faders. Gate the
            # wide structural trail to close>SMA50 AND SMA50 rising; else the tight %-trail exits the
            # fader sooner (cut give-back + free the symbol -> recover the blocked-entry velocity cost).
            _m50 = pd.Series(closes).rolling(50, min_periods=50).mean()
            struct_trend_ok = ((pd.Series(closes) > _m50) & (_m50 > _m50.shift(10))).to_numpy()

    def _overext_pct(i: int) -> float:
        """Effective overext extension threshold at bar i (distribution-conditioned if enabled)."""
        if (cfg.overext_atr_mult is not None and atr_ratio is not None
                and not np.isnan(atr_ratio[i])):
            return float(min(cfg.overext_atr_hi,
                             max(cfg.overext_atr_lo, cfg.overext_atr_mult * atr_ratio[i])))
        if overext_ad_balance is None or np.isnan(overext_ad_balance[i]):
            return cfg.overext_pct
        if cfg.overext_dist_neg_thresh is not None:
            # asymmetric: only LOWER the bar when distribution is heavy; never defer accum runners
            if overext_ad_balance[i] <= cfg.overext_dist_neg_thresh:
                return cfg.overext_dist_neg_pct
            return cfg.overext_pct
        if cfg.overext_dist_slope is None:
            # ad_balance was computed only for dist_arm (no overext distribution-shaping) -> base pct
            return cfg.overext_pct
        eff = cfg.overext_pct + cfg.overext_dist_slope * overext_ad_balance[i]
        return float(min(cfg.overext_dist_pct_hi, max(cfg.overext_dist_pct_lo, eff)))

    def _overext_rev_ok(i: int) -> bool:
        """Reversal confirmation for the overext top-sell (causal: bar i and earlier only)."""
        mode = cfg.overext_reversal_mode
        if mode is None:
            return (not cfg.overext_reversal) or (i >= 1 and closes[i] < closes[i - 1])
        if mode == "none":
            return True
        if i < 1:
            return False
        if mode == "down1":
            return closes[i] < closes[i - 1]
        if mode == "strong_down":
            return closes[i] < opens[i] and closes[i] <= closes[i - 1] * (1.0 - cfg.overext_strong_down_pct)
        if mode == "engulf2":
            prev_up = closes[i - 1] > opens[i - 1]
            cur_down = closes[i] < opens[i]
            engulf = opens[i] >= closes[i - 1] and closes[i] <= opens[i - 1]
            return prev_up and cur_down and engulf
        if mode == "three_down":
            return i >= 2 and closes[i] < closes[i - 1] < closes[i - 2]
        if mode == "ema_cross":
            return overext_ema is not None and closes[i] < overext_ema[i] and closes[i - 1] >= overext_ema[i - 1]
        if mode == "bear_div":
            if overext_div is None or np.isnan(overext_div[i]):
                return False
            return overext_div[i] >= cfg.overext_div_threshold
        return False

    in_pos = False
    b_pos = False  # current position was opened by the B-channel breakout entry
    entry_idx = -1
    entry_signal_idx = -1
    entry_fill = 0.0
    peak_high = 0.0  # highest high since entry — drives the trailing give-back stop
    pop_evaluated = False  # strength-gated pop lock: classified weak/strong yet?
    pop_is_weak = False    # True => this position pops weakly -> tight trail
    overext_armed = False  # overext->tight-trail: once extended, ride with a tight give-back band
    last_loss_exit_idx = -10**9  # re-entry cooldown: bar of the last LOSING exit on this symbol
    last_exit_idx = -10**9       # re-entry premium cap: bar of the last exit (any reason)
    last_exit_price: float | None = None  # fill price of that last exit
    last_exit_reason: str | None = None    # resumption re-entry: reason of the last exit
    last_new_high_idx = -10**9   # stale exit: bar of the most recent new peak high (this trade)

    # Determine loop boundary based on entry fill type
    loop_limit = n if cfg.entry_bar_fill_type == "close_same" else n - 1

    i = 0
    while i < loop_limit:
        sig = sig_map.get(pd.Timestamp(dates[i]), 0)
        if not in_pos:
            # CONFIRMATION-BREAKOUT B-ENTRY: additive entry at an idle slot on a quality-gated
            # wave start — z(score6) high AND the close breaks the recent thrust high AND price
            # is below the trend MA (the zone the core channel is silent in). Checked BEFORE
            # the core branch so it preempts a same-bar core signal (at-market close_next vs
            # the core's patient limit; W3-A4: capture 89%, +10.6%/21bar, false-fill 41% vs
            # 86% for raw -2% limits). Respects the loss cooldown like the core path; the
            # market gates are not applied (mirror of resume_reentry — react, not predict).
            if (cfg.entry_bchannel_z is not None and bot_z is not None
                    and i >= cfg.entry_bchannel_break_lookback and i < n - 1
                    and not np.isnan(bot_z[i])
                    and float(bot_z[i]) >= cfg.entry_bchannel_z
                    and closes[i] > float(highs[i - cfg.entry_bchannel_break_lookback:i].max())
                    and (bch_ma is None or closes[i] < bch_ma[i])
                    and (cfg.reentry_cooldown_bars <= 0
                         or i - last_loss_exit_idx >= cfg.reentry_cooldown_bars)):
                entry_signal_idx = i
                entry_idx = i + 1
                entry_fill = cfg.cost.fill_buy(closes[entry_idx])
                peak_high = float(highs[entry_idx])
                in_pos = True
                b_pos = True
                BCH_ENTRY_LOG.append({
                    "symbol": sym,
                    "signal_date": pd.Timestamp(dates[i]),
                    "entry_date": pd.Timestamp(dates[entry_idx]),
                })
                i = entry_idx
                continue
            if sig > 0:
                # Market-regime entry gate: don't open new risk into a market washout
                # (the per-symbol model is blind to the tape; buys into a falling market
                # are low-WR). Causal — market_weak_dates uses data up to the signal bar.
                if (
                    market_weak_dates is not None
                    and pd.Timestamp(dates[i]) in market_weak_dates
                ):
                    i += 1
                    continue
                # Market-chop entry gate: skip new entries while the tape is whippy/directionless
                # (low trend efficiency) — the fast-fail churn regime the per-symbol head is blind to.
                if (
                    market_chop_dates is not None
                    and pd.Timestamp(dates[i]) in market_chop_dates
                ):
                    i += 1
                    continue
                # Re-entry cooldown: skip a buy too soon after a losing exit (anti-whipsaw).
                if cfg.reentry_cooldown_bars > 0 and i - last_loss_exit_idx < cfg.reentry_cooldown_bars:
                    i += 1
                    continue
                # Re-entry premium cap for AT-MARKET fills (no pullback wait): skip a gap-up chase
                # of the same name within the window. The pullback path caps its limit instead.
                if (cfg.reentry_max_premium_pct is not None and last_exit_price is not None
                        and cfg.entry_pullback_pct is None
                        and i - last_exit_idx <= cfg.reentry_max_premium_bars
                        and closes[i] > last_exit_price * (1.0 + cfg.reentry_max_premium_pct)):
                    i += 1
                    continue
                entry_signal_idx = i
                b_pos = False  # core-channel entry (any fill path below)
                _skip_pb = (
                    cfg.downtrend_skip_pullback
                    and downtrend_arr is not None
                    and downtrend_arr[i]
                ) or (
                    strength_skip_arr is not None and strength_skip_arr[i]
                )
                if cfg.entry_pullback_pct is not None and not _skip_pb:
                    # Patient limit fill: wait for a pullback to close[i]*(1-pct) within window;
                    # skip the signal if price never trades down to it (don't chase the runaway).
                    _depth = cfg.entry_pullback_pct
                    if pb_vol_depth is not None and not np.isnan(pb_vol_depth[i]):
                        _depth = float(pb_vol_depth[i])
                    # C': deepen the dip in topping structure -> lower fill price or skip the toppy buy.
                    if pb_topstruct_depth is not None and not np.isnan(pb_topstruct_depth[i]):
                        _depth = _depth + float(pb_topstruct_depth[i])
                    # Conviction-graded: SHALLOW the depth for strong signals (fill the runner instead of
                    # waiting for a deep dip that never comes -> attacks the 48% fill-rate).
                    if pb_conv_mult is not None:
                        _depth = _depth * float(pb_conv_mult[i])
                    # WEEKLY-DOWN deepen: knives concentrate where the weekly trend is not up — demand
                    # a deeper buffer there (knife fills lower / skips); weekly-up keeps normal depth.
                    if pb_wk_down is not None and bool(pb_wk_down[i]):
                        _depth = _depth * min(1.0 + cfg.entry_pullback_wk_deepen_k,
                                              cfg.entry_pullback_wk_deepen_cap)
                    # BOTTOM-deepen: a PRE-sided bottom head (score6) anticipating a bottom DEEPENS the
                    # required pullback so the fill lands nearer the true low (enter the same trade
                    # cheaper). Conditioned -> the deeper limit is the level price is expected to reach.
                    if (cfg.bot_deepen_k is not None and bot is not None and not np.isnan(bot[i])):
                        _depth = _depth * float(np.clip(
                            1.0 + cfg.bot_deepen_k * bot[i], 1.0, cfg.bot_deepen_cap))
                    # BOTTOM-shallow: HIGH z(score6) (bottom-structure head) = a genuine wave start
                    # that only dips ~2% — SHRINK the depth so it fills; low z keeps the full
                    # knife-filter depth. After bot_deepen (shrinks the deepened depth if both on).
                    if cfg.bot_shallow_k > 0 and bot_z is not None and not np.isnan(bot_z[i]):
                        _depth = _depth * max(
                            cfg.bot_shallow_floor,
                            1.0 - cfg.bot_shallow_k * max(float(bot_z[i]), 0.0))
                    limit = closes[i] * (1.0 - _depth)
                    # EXP-D: floor the limit at the trend MA so the fill stays above it (the deep
                    # fixed pullback otherwise drags ~half the fills below MA20 = the weak cohort).
                    if pb_floor_ma is not None and not np.isnan(pb_floor_ma[i]):
                        floor_price = pb_floor_ma[i] * (1.0 + cfg.entry_pullback_floor_buffer)
                        limit = min(closes[i], max(limit, floor_price))
                    # FILL-PRICE: deepen the limit toward the recent structural swing-low (the leg base)
                    # so the fill lands nearer the bottom. min() => never shallower than the fixed dip;
                    # clamp <= close[i]. With fill_if_missed, legs that never return to the base still
                    # fill at-market window-end (re-prices fills WITHOUT cutting trades).
                    if pb_struct_low is not None and not np.isnan(pb_struct_low[i]):
                        struct_price = pb_struct_low[i] * (1.0 + cfg.entry_pullback_structural_buffer)
                        limit = min(limit, max(struct_price, 0.0))
                        if limit > closes[i]:
                            limit = closes[i]
                    # Re-entry premium cap (anti gap-up-chase): within the window after an exit, force
                    # the limit at/below prior_exit_price*(1+pct) so the wait fills only at a non-chase
                    # price; if price never dips that low the signal is skipped (no chase). Causal.
                    if (cfg.reentry_max_premium_pct is not None and last_exit_price is not None
                            and i - last_exit_idx <= cfg.reentry_max_premium_bars):
                        limit = min(limit, last_exit_price * (1.0 + cfg.reentry_max_premium_pct))
                    j_end = min(i + cfg.entry_pullback_window, n - 1)
                    fill_j = -1
                    for j in range(i + 1, j_end + 1):
                        # EXP1: cancel the pending limit once the trend rolls over while waiting.
                        if pb_ma is not None and closes[j] < pb_ma[j]:
                            break
                        if lows[j] <= limit:
                            # CANDLE-CONFIRMATION: skip a still-falling RED touch bar (the knife);
                            # keep scanning for a green (dip-then-recover) touch within the window.
                            if cfg.entry_pullback_confirm_reversal and closes[j] < opens[j]:
                                continue
                            fill_j = j
                            break
                    if fill_j < 0:
                        # Hybrid: take the runaway at-market at the window-end close instead of dropping.
                        if (cfg.entry_pullback_fill_if_missed and j_end > i
                                and (cfg.fill_if_missed_max_premium is None
                                     or closes[j_end] <= closes[i] * (1.0 + cfg.fill_if_missed_max_premium))):
                            entry_idx = j_end
                            entry_fill = cfg.cost.fill_buy(closes[entry_idx])
                            peak_high = float(highs[entry_idx])
                            in_pos = True
                            i = entry_idx
                            continue
                        fwd8 = closes[min(i + 8, n - 1)] / closes[i] - 1
                        fwd20 = closes[min(i + 20, n - 1)] / closes[i] - 1
                        MISSED_SIGNAL_LOG.append({
                            "symbol": sym,
                            "signal_date": pd.Timestamp(dates[i]),
                            "signal_close": float(closes[i]),
                            "limit_price": float(limit),
                            "pullback_depth": float(_depth),
                            "fwd8d_return": float(fwd8),
                            "fwd20d_return": float(fwd20),
                        })
                        i += 1  # missed — never pulled back; skip this signal
                        continue
                    entry_idx = fill_j
                    entry_fill = cfg.cost.fill_buy(limit)  # filled at the limit price
                    peak_high = float(highs[entry_idx])
                    in_pos = True
                    i = entry_idx
                    continue
                if cfg.entry_bar_fill_type == "close_same":
                    entry_idx = i
                    entry_fill = cfg.cost.fill_buy(closes[entry_idx])
                    peak_high = float(highs[entry_idx])
                    in_pos = True
                    i = entry_idx + 1  # next check exit at i+1
                elif cfg.entry_bar_fill_type == "close_next":
                    if i >= n - 1:  # can't do next-bar fill on last bar
                        i += 1
                        continue
                    entry_idx = i + 1
                    entry_fill = cfg.cost.fill_buy(closes[entry_idx])  # close, not open
                    peak_high = float(highs[entry_idx])
                    in_pos = True
                    i = entry_idx
                else:  # "open_next"
                    if i >= n - 1:  # can't do next-bar fill on last bar
                        i += 1
                        continue
                    entry_idx = i + 1
                    entry_fill = cfg.cost.fill_buy(opens[entry_idx])
                    peak_high = float(highs[entry_idx])
                    in_pos = True
                    i = entry_idx
                continue
            # RESUMPTION RE-ENTRY (no fresh signal needed): after a trailing exit, if the trend
            # reasserts (close breaks back above the exit bar's high) within the window, re-enter
            # AT-MARKET — recover the continuation the trail handed back. React, not predict.
            if (cfg.resume_reentry_win is not None
                    and last_exit_reason in ("overext_trail", "trailing_stop")
                    and 0 <= last_exit_idx < n
                    and 1 <= i - last_exit_idx <= cfg.resume_reentry_win
                    and closes[i] > highs[last_exit_idx]
                    and closes[i] <= highs[last_exit_idx] * (1.0 + cfg.resume_reentry_max_premium)
                    and (cfg.reentry_cooldown_bars <= 0
                         or i - last_loss_exit_idx >= cfg.reentry_cooldown_bars)
                    and i < n - 1):
                entry_signal_idx = i
                b_pos = False  # resumption re-entry rides the core machinery
                entry_idx = i + 1
                entry_fill = cfg.cost.fill_buy(closes[entry_idx])
                peak_high = float(highs[entry_idx])
                in_pos = True
                last_exit_reason = None  # one re-entry per exit
                i = entry_idx
                continue
            i += 1
            continue

        # In position. Check exit conditions in configured priority order.
        hold_bars = i - entry_idx
        if last_new_high_idx < entry_idx:           # baseline at entry (stale exit)
            last_new_high_idx = entry_idx
        # Track the running peak high so the trailing stop measures peak-to-current.
        if highs[i] > peak_high:
            peak_high = float(highs[i])
            last_new_high_idx = i                    # new peak -> reset stale counter
        reason: str | None = None

        # WR-expression csrank hard stop: cut a PREDICTED-LOSER (high entry-csrank) trade once
        # it drops below the shallow floor — checked ahead of exit_priority, only for the gated
        # cohort, so winners (which barely dip) are untouched. Causal.
        if (cfg.csr_hard_stop_pct is not None and cfg.csr_hard_stop_threshold is not None
                and hold_bars >= cfg.min_hold_bars and ecsr is not None
                and entry_signal_idx >= 0 and not np.isnan(ecsr[entry_signal_idx])
                and ecsr[entry_signal_idx] >= cfg.csr_hard_stop_threshold
                and lows[i] / entry_fill - 1.0 <= cfg.csr_hard_stop_pct):
            reason = "csr_hard_stop"

        # Regime-conditional tighter stop: cut a trade ENTERED in a downtrend once it drops
        # below the tight floor (failed bounce) — checked ahead of exit_priority, only for the
        # downtrend cohort, so uptrend trades keep the loose downleg force-gate. Causal.
        if (reason is None and cfg.downtrend_hard_stop_pct is not None
                and hold_bars >= cfg.min_hold_bars and downtrend_arr is not None
                and entry_signal_idx >= 0 and downtrend_arr[entry_signal_idx]
                and lows[i] / entry_fill - 1.0 <= cfg.downtrend_hard_stop_pct):
            reason = "downtrend_stop"

        # Structural stop (EXP-3 pattern-anchored buffer): exit if price breaks the pre-entry swing
        # low — the consolidation/base support that supplies the price buffer a waited-for dip gives.
        # Per-trade loss is capped at (entry - support)/entry. Checked ahead of exit_priority. Causal.
        if (reason is None and cfg.structural_stop_lookback is not None
                and hold_bars >= cfg.min_hold_bars and entry_signal_idx >= 0):
            _lo0 = max(0, entry_signal_idx - cfg.structural_stop_lookback + 1)
            _support = float(lows[_lo0:entry_signal_idx + 1].min())
            if lows[i] <= _support * (1.0 - cfg.structural_stop_buffer):
                reason = "structural_stop"

        # B-only structural stop: a B-entry buys a breakout above the thrust high with no
        # core-channel pullback buffer — exit when the low breaks the pre-trigger structural
        # low (min low over stop_lookback bars ending at the trigger bar). Next-bar close
        # fill like the other stops. Only for B-channel positions; reason 'bch_stop'.
        if (reason is None and b_pos and cfg.entry_bchannel_stop_lookback is not None
                and hold_bars >= cfg.min_hold_bars and entry_signal_idx >= 0):
            _blo0 = max(0, entry_signal_idx - cfg.entry_bchannel_stop_lookback + 1)
            if lows[i] <= float(lows[_blo0:entry_signal_idx + 1].min()):
                reason = "bch_stop"

        # Vol-adaptive hard stop: cut every name at the SAME ATR distance below entry (constant risk),
        # so the fixed-%% stop stops letting low-vol losers bleed deeper. Checked ahead of exit_priority.
        if (reason is None and cfg.hard_stop_atr_mult is not None and atr_ratio is not None
                and hold_bars >= cfg.min_hold_bars and entry_signal_idx >= 0
                and not np.isnan(atr_ratio[entry_signal_idx]) and atr_ratio[entry_signal_idx] > 0
                and lows[i] / entry_fill - 1.0 <= -cfg.hard_stop_atr_mult * atr_ratio[entry_signal_idx]):
            reason = "atr_stop"

        # Breakeven lock (non-selection RIDE mechanic): once peak gain proved >= breakeven_lock_mfe,
        # floor the trade at entry*(1+offset) — cut the winner-round-trip-to-loss cohort without
        # clipping runners (which stay well above the floor). Checked ahead of exit_priority. Causal.
        if (reason is None and cfg.breakeven_lock_mfe is not None
                and hold_bars >= cfg.min_hold_bars
                and peak_high / entry_fill - 1.0 >= cfg.breakeven_lock_mfe
                and lows[i] <= entry_fill * (1.0 + cfg.breakeven_lock_offset)):
            reason = "breakeven_lock"

        # MACD last-line shield (user 2026-06-18): force an exit on a 3-way decline confirmation
        # (MACD hist<0 AND close<MA AND red bar). Ahead of exit_priority so it bypasses the protect
        # band / market-washout deferral. Catches the du-dinh round-top the velocity head is blind to.
        if (reason is None and macd_shield_arr is not None
                and hold_bars >= cfg.min_hold_bars and macd_shield_arr[i]
                and (cfg.macd_shield_min_gain <= 0.0
                     or peak_high / entry_fill - 1.0 >= cfg.macd_shield_min_gain)):
            reason = "macd_shield"

        # RSI-slope last-line shield (research: rsi_slope_5 -0.155 = strongest top-signal). Backstop.
        if (reason is None and rsi_shield_arr is not None
                and hold_bars >= cfg.min_hold_bars and rsi_shield_arr[i]
                and (cfg.rsi_shield_min_gain <= 0.0
                     or peak_high / entry_fill - 1.0 >= cfg.rsi_shield_min_gain)):
            reason = "rsi_shield"

        for exit_rule in cfg.exit_priority:
            if reason is not None:
                break
            if exit_rule == "hard_stop":
                if cfg.hard_stop_pct is not None and hold_bars >= cfg.min_hold_bars:
                    mtm_low = lows[i] / entry_fill - 1.0
                    if mtm_low <= cfg.hard_stop_pct:
                        reason = "hard_stop"
                        break

            elif exit_rule == "trailing_stop":
                if (
                    cfg.trailing_stop_pct is not None
                    and hold_bars >= cfg.min_hold_bars
                    and peak_high > 0.0
                ):
                    # Entry-conditioned trail: cold entries (weak prior momentum) lock earlier.
                    act_pct = cfg.trailing_activate_pct
                    trail_pct = cfg.trailing_stop_pct
                    # EXP-B entry-peak-conditioned activation: set the arm threshold per-trade from
                    # the entry mfe-head's forward-peak prediction (score4). Low predicted peak ->
                    # arm early (protect the 10-27% deadband); big predicted runner -> high arm.
                    if (
                        cfg.mfe_act_k is not None
                        and emfe is not None
                        and entry_signal_idx >= 0
                        and not np.isnan(emfe[entry_signal_idx])
                    ):
                        act_pct = float(np.clip(
                            cfg.mfe_act_k * emfe[entry_signal_idx],
                            cfg.mfe_act_floor, cfg.mfe_act_cap))
                    if (
                        cfg.entry_momentum_window > 0
                        and cfg.cold_trailing_stop_pct is not None
                        and entry_idx - cfg.entry_momentum_window >= 0
                        and closes[entry_idx - cfg.entry_momentum_window] > 0.0
                    ):
                        runup = (
                            closes[entry_idx]
                            / closes[entry_idx - cfg.entry_momentum_window]
                            - 1.0
                        )
                        if runup <= cfg.entry_cold_threshold:
                            act_pct = cfg.cold_trailing_activate_pct
                            trail_pct = cfg.cold_trailing_stop_pct
                    peak_gain = peak_high / entry_fill - 1.0
                    # EXP2 two-tier: a looser FIXED band protects the mid (e.g. 10-27%) give-back
                    # zone; the tight default tier resumes above trailing_activate for big runners.
                    tier2_active = False
                    # WR-expression gate: when tier2_csr_threshold is set, the tier-2 (faster)
                    # trail fires ONLY for predicted-loser entries (entry-bar momentum csrank >=
                    # threshold). High-csrank winners keep the wide default trail.
                    tier2_ok = True
                    if cfg.tier2_csr_threshold is not None:
                        tier2_ok = (
                            ecsr is not None and entry_signal_idx >= 0
                            and not np.isnan(ecsr[entry_signal_idx])
                            and ecsr[entry_signal_idx] >= cfg.tier2_csr_threshold
                        )
                    if (cfg.trailing_tier2_activate_pct is not None
                            and cfg.trailing_tier2_stop_pct is not None
                            and act_pct is not None
                            and tier2_ok
                            and cfg.trailing_tier2_activate_pct <= peak_gain < act_pct):
                        act_pct = cfg.trailing_tier2_activate_pct
                        trail_pct = cfg.trailing_tier2_stop_pct
                        tier2_active = True
                    # Strength-gated pop lock: at the first bar peak-gain crosses the arm level,
                    # classify weak (close barely above short MA) vs strong; weak pops get the
                    # tight trail armed from here, strong pops keep the loose default trail.
                    if (
                        cfg.pop_lock_arm_pct is not None
                        and cfg.pop_lock_trail_pct is not None
                    ):
                        if not pop_evaluated and peak_gain >= cfg.pop_lock_arm_pct:
                            w = cfg.pop_lock_ext_window
                            lo_w = max(0, i - w + 1)
                            ma = closes[lo_w:i + 1].mean()
                            ext = closes[i] / ma - 1.0 if ma > 0 else 0.0
                            pop_is_weak = ext < cfg.pop_lock_ext_threshold
                            pop_evaluated = True
                        if pop_is_weak:
                            act_pct = cfg.pop_lock_arm_pct
                            trail_pct = cfg.pop_lock_trail_pct
                    # Vol-adaptive trail width: scale by the stock's own ATR/close.
                    if atr_ratio is not None and not tier2_active:
                        trail_pct = float(np.clip(
                            cfg.trailing_atr_mult * atr_ratio[i],
                            cfg.trailing_atr_floor, cfg.trailing_atr_cap))
                    # ML-magnitude modulation: tighten the band when the exit head predicts a
                    # big forward drop (high z), widen it when it predicts calm (low z).
                    if (xscore_z is not None and trail_pct is not None and not tier2_active
                            and not np.isnan(xscore_z[i])):
                        m = float(np.clip(1.0 - cfg.trailing_score_k * xscore_z[i],
                                          cfg.trailing_score_mult_floor, cfg.trailing_score_mult_cap))
                        trail_pct = trail_pct * m
                    armed = act_pct is None or peak_gain >= act_pct
                    exit_reason_trail = "trailing_stop"
                    # Overext-armed tight trail: once price hit the overext extension, ride the
                    # runner with a tight give-back band instead of a hard top-sell.
                    if overext_armed and cfg.overext_trail_pct is not None:
                        trail_pct = cfg.overext_trail_pct
                        armed = True
                        exit_reason_trail = "overext_trail"
                    # COMBO trail modulation (loop iter-6): scale the give-back band by the ENTRY-bar
                    # combo (IC 0.248 vs realized MFE) — widen for high-combo runners (let them ride
                    # to the 21% MFE they reach), tighten for low-combo (take the 12% sooner). Applies
                    # to whichever band is active (trailing_stop or overext_trail). Causal (entry bar).
                    if (cfg.trailing_combo_k is not None and combo_arr is not None and trail_pct is not None
                            and entry_signal_idx >= 0 and not np.isnan(combo_arr[entry_signal_idx])):
                        mc = float(np.clip(
                            1.0 + cfg.trailing_combo_k * (combo_arr[entry_signal_idx] - cfg.trailing_combo_mid),
                            cfg.trailing_combo_floor, cfg.trailing_combo_cap))
                        trail_pct = trail_pct * mc
                    # BOTTOM-ride coupling: a trade entered at a true zigzag bottom (high score6 =
                    # low MAE) rides a WIDER give-back band to a better exit (the entry bought the
                    # MDD budget). Widen only (floor 1.0); low/absent score6 -> no change. Causal.
                    if (cfg.bot_ride_k is not None and bot is not None and trail_pct is not None
                            and entry_signal_idx >= 0 and not np.isnan(bot[entry_signal_idx])):
                        mb = float(np.clip(
                            1.0 + cfg.bot_ride_k * (bot[entry_signal_idx] - cfg.bot_ride_mid),
                            cfg.bot_ride_floor, cfg.bot_ride_cap))
                        trail_pct = trail_pct * mb
                    # Trend-intact suppression: hold a healthy-uptrend pullback (price still above a
                    # rising trend MA) instead of trailing out mid-wave; the overext_trail handoff is
                    # exempt. Defers protection to overext / downleg / belowma20.
                    if (cfg.trailing_skip_above_ma is not None and exit_reason_trail == "trailing_stop"
                            and trend_up is not None and trend_up[i]):
                        armed = False
                    # EXP-C vol-expansion protective arm: a vol spike on an in-profit position arms a
                    # tight give-back band — react to risk MAGNITUDE, not top direction. Overrides the
                    # trend-intact suppression (a vol spike mid-uptrend = the parabolic climax to lock).
                    # The deliberate overext_trail tight ride is left as-is (already protective).
                    if (cfg.vol_spike_z_threshold is not None and vol_z is not None
                            and not overext_armed and not np.isnan(vol_z[i])
                            and vol_z[i] >= cfg.vol_spike_z_threshold
                            and (peak_high / entry_fill - 1.0) >= cfg.vol_spike_min_gain):
                        trail_pct = cfg.vol_spike_trail_pct
                        armed = True
                        exit_reason_trail = "vol_protect"
                    # REALIZABILITY arm: distribution-heavy 20-bar internals on an in-profit position
                    # arm a tight give-back band (round-trippers separate from riders on ad_balance).
                    if (cfg.dist_arm_neg_thresh is not None and overext_ad_balance is not None
                            and not overext_armed and not np.isnan(overext_ad_balance[i])
                            and overext_ad_balance[i] <= cfg.dist_arm_neg_thresh
                            and (peak_high / entry_fill - 1.0) >= cfg.dist_arm_min_gain):
                        trail_pct = cfg.dist_arm_trail_pct
                        armed = True
                        exit_reason_trail = "dist_protect"
                    # ML-SWING-TOP arm: the swing-top detector fires (prob>=thr) on an in-profit
                    # position -> arm a tight give-back band to sell near the top (Phase D). The
                    # downleg/overext gate stays the always-on backstop for tops it misses.
                    if (cfg.ml_swing_trail_thr is not None and swing_prob is not None
                            and not overext_armed and not np.isnan(swing_prob[i])
                            and swing_prob[i] >= cfg.ml_swing_trail_thr
                            and (peak_high / entry_fill - 1.0) >= cfg.ml_swing_min_gain):
                        trail_pct = cfg.ml_swing_trail_pct
                        armed = True
                        exit_reason_trail = "ml_swing"
                    # REGIME-ADAPTIVE tighten (un-hardcode the fixed trail band): in a distribution/
                    # sideways regime (consolidation_score >= thr) MULTIPLY the trail band tighter so the
                    # real top is locked faster (cuts LATE_EXIT); a clean trend (low consolidation) keeps
                    # the loose default and rides the pullback (holds SOLD_THEN_RAN). Applies to whatever
                    # band is active (default/overext/vol_protect). Causal.
                    if (cons_trail is not None and trail_pct is not None
                            and not np.isnan(cons_trail[i]) and cons_trail[i] >= cfg.trailing_cons_thr):
                        trail_pct = trail_pct * cfg.trailing_cons_tight_mult
                    # TREND-BREAK profit-lock (chartist): an in-profit winner that closes below the
                    # trend MA is locked NOW (don't wait for the lagging %-trail re-arm). Flexible —
                    # only winners (peak gain >= thr); the deliberate overext tight ride is exempt.
                    if (cfg.trend_break_lock_gain is not None and tb_ma_arr is not None
                            and not overext_armed
                            and (peak_high / entry_fill - 1.0) >= cfg.trend_break_lock_gain
                            and closes[i] < tb_ma_arr[i]):
                        reason = "trend_break"
                        break
                    if armed:
                        # STRUCTURE trail on the DEFAULT tier: exit on a structural break (close <
                        # prior-N-bar low) instead of the fixed %-giveback, so runners ride through
                        # shallow pullbacks. Protective arms (overext/vol/dist) keep their %-band.
                        _struct_tier = (exit_reason_trail == "trailing_stop"
                                        or (cfg.trailing_struct_apply_overext
                                            and exit_reason_trail == "overext_trail"))
                        if struct_trend_ok is not None and not bool(struct_trend_ok[i]):
                            _struct_tier = False  # trend weak -> use tight %-trail (exit fader sooner)
                        if (don_low is not None and _struct_tier
                                and not np.isnan(don_low[i])):
                            if closes[i] < don_low[i]:
                                reason = exit_reason_trail
                                break
                        else:
                            drop_from_peak = lows[i] / peak_high - 1.0
                            if drop_from_peak <= -trail_pct:
                                reason = exit_reason_trail
                                break

            elif exit_rule == "signal":
                if cfg.signal_exit_enabled and sig < 0 and hold_bars >= cfg.min_hold_bars:
                    # Age incubation: don't honor the signal exit while the trade is still
                    # young — give it time to reach the trail-arm instead of dying premature.
                    # Underwater guard: still cut a young trade that is already below the floor
                    # (the entry-driven losers) so incubation doesn't deepen the drawdown.
                    if hold_bars < cfg.signal_exit_min_age and (
                        cfg.signal_exit_incubate_floor is None
                        or closes[i] / entry_fill - 1.0 >= cfg.signal_exit_incubate_floor
                    ):
                        continue
                    # Market-regime gate: suppress the signal exit when the whole tape is in a
                    # sharp short-term drop (beta washout) — defer to the trailing/hard stop.
                    if (
                        market_drop_dates is not None
                        and pd.Timestamp(dates[i]) in market_drop_dates
                    ):
                        continue  # skip this exit rule, keep checking the rest / hold
                    # Vol-adaptive extension hold: don't sell an in-profit, trending winner into the MEAN
                    # — hold until it stretches K*ATR above its MA (sell the extension, not the mean). The
                    # un-stretched winner then rides toward overext/trail; protective gates still backstop.
                    _hold_k = cfg.signal_exit_hold_ext_atr
                    if (_hold_k is not None and cfg.signal_exit_hold_volscale
                            and atr_hold_ratio is not None and not np.isnan(atr_hold_ratio[i])
                            and atr_hold_ratio[i] > 0):
                        _hold_k = _hold_k * float(np.clip(
                            cfg.signal_exit_hold_volscale_ref / atr_hold_ratio[i], 0.5, 2.0))
                    if (_hold_k is not None and cfg.signal_exit_hold_rs_scale > 0
                            and rs_vsma_arr is not None):
                        _hold_k = _hold_k * float(np.clip(
                            1.0 + cfg.signal_exit_hold_rs_scale * rs_vsma_arr[i], 0.5, 2.0))
                    if (_hold_k is not None and cfg.signal_exit_hold_mkt_scale > 0
                            and mkt_trend_arr is not None):
                        _hold_k = _hold_k * float(np.clip(
                            1.0 + cfg.signal_exit_hold_mkt_scale * mkt_trend_arr[i], 0.5, 2.0))
                    if (_hold_k is not None and cfg.signal_exit_hold_legage_scale > 0
                            and legage_norm_arr is not None):
                        _hold_k = _hold_k * float(np.clip(
                            1.0 - cfg.signal_exit_hold_legage_scale * legage_norm_arr[i], 0.5, 2.0))
                    if (_hold_k is not None and cfg.signal_exit_hold_legamp_scale > 0
                            and legamp_norm_arr is not None):
                        _hold_k = _hold_k * float(np.clip(
                            1.0 + cfg.signal_exit_hold_legamp_scale * legamp_norm_arr[i], 0.5, 2.0))
                    if (_hold_k is not None and cfg.signal_exit_hold_runscore_scale > 0
                            and runscore_arr is not None):
                        _hold_k = _hold_k * float(np.clip(
                            1.0 + cfg.signal_exit_hold_runscore_scale * runscore_arr[i], 0.5, 2.0))
                    if (cfg.signal_exit_hold_ext_atr is not None and ext_atr_hold is not None
                            and closes[i] > entry_fill * (1.0 + cfg.signal_exit_hold_profit_floor)
                            and not np.isnan(ext_atr_hold[i])
                            and ext_atr_hold[i] < _hold_k
                            and hold_trend_ok is not None and hold_trend_ok[i]
                            and (cfg.signal_exit_hold_min_score3_z is None or (
                                s3z is not None and not np.isnan(s3z[i])
                                and s3z[i] >= cfg.signal_exit_hold_min_score3_z))
                            and (cfg.signal_exit_hold_max_atrpct is None or (
                                atr_hold_ratio is not None and not np.isnan(atr_hold_ratio[i])
                                and atr_hold_ratio[i] <= cfg.signal_exit_hold_max_atrpct))):
                        continue
                    # EXP3 MFE protect band: don't dump a winner mid-give-back in the unprotected
                    # 10-27% MFE zone; defer to trailing/overext/downleg so it can ride to the arm.
                    if (cfg.signal_exit_protect_lo is not None
                            and cfg.signal_exit_protect_hi is not None):
                        pg = peak_high / entry_fill - 1.0
                        _tr = trend_up_slow if trend_up_slow is not None else trend_up
                        if (cfg.signal_exit_protect_lo <= pg < cfg.signal_exit_protect_hi
                                and (not cfg.signal_exit_protect_require_trend
                                     or (_tr is not None and _tr[i]))):
                            # Early-release escape: a large vol-scaled give-back from the peak overrides
                            # the protect so the already-firing signal exit fires now (catch the down-leg
                            # in a few bars) instead of waiting ~20-30 bars for the slow MA to break.
                            released = False
                            if (cfg.signal_exit_protect_release_drop_k is not None
                                    and protect_release_vol is not None and peak_high > 0.0
                                    and not np.isnan(protect_release_vol[i])):
                                drop_pk = 1.0 - closes[i] / peak_high
                                _rel_k = cfg.signal_exit_protect_release_drop_k
                                if cfg.signal_exit_release_rs_scale > 0 and rs_vsma_arr is not None:
                                    _rel_k = _rel_k * float(np.clip(
                                        1.0 + cfg.signal_exit_release_rs_scale * rs_vsma_arr[i], 0.5, 2.0))
                                if drop_pk >= _rel_k * protect_release_vol[i]:
                                    released = True
                            if not released:
                                continue
                    # EXP3 coherence veto: don't sell while the entry head still says BUY.
                    if (cfg.signal_exit_skip_if_entry_z is not None and escore_z is not None
                            and not np.isnan(escore_z[i])
                            and escore_z[i] >= cfg.signal_exit_skip_if_entry_z):
                        continue
                    # score3 continuation veto: hold while the continuation head still predicts run.
                    if (cfg.signal_exit_skip_if_score3_z is not None and s3z is not None
                            and not np.isnan(s3z[i])
                            and s3z[i] >= cfg.signal_exit_skip_if_score3_z):
                        continue
                    # MARKET-REGIME veto (shakeout_vs_top hard rule): don't sell into a bull-tape
                    # shakeout — while VNINDEX/MA(N) - 1 >= margin (broad uptrend => dip likely
                    # recovers, AUC 0.78), defer to trail/overext/max_hold. Only holds in a BULL
                    # tape so it never rides into a bear-flip; winner_only keeps losers cutting.
                    if (cfg.signal_exit_skip_if_mkt_above_ma is not None and mkt_skip_arr is not None
                            and mkt_skip_arr[i] >= cfg.signal_exit_skip_if_mkt_margin
                            and (not cfg.signal_exit_skip_if_mkt_winner_only or closes[i] > entry_fill)):
                        continue
                    # SNR runner-extension: in a clean-trend universe regime, let winners ride the
                    # trail instead of dumping on the signal (only when the trade is already a winner).
                    if (cfg.exit_snr_extend_threshold is not None and market_snr_dates is not None
                            and peak_high / entry_fill - 1.0 >= cfg.exit_snr_min_gain
                            and (cfg.exit_snr_defer_min_giveback <= 0.0
                                 or (peak_high - closes[i]) / entry_fill
                                 >= cfg.exit_snr_defer_min_giveback)
                            and pd.Timestamp(dates[i]) in market_snr_dates):
                        continue
                    reason = "signal"
                    break

            elif exit_rule == "overext":
                # Sell-the-top: force a sell when price is over-extended above its SMA, the
                # signal the trained exit head is blind to at tops. Independent of the
                # signal-exit age/market gates by design. Causal (current+past closes only).
                if cfg.overext_ma_window > 0 and hold_bars >= cfg.min_hold_bars:
                    lo_w = max(0, i - cfg.overext_ma_window + 1)
                    ma = closes[lo_w:i + 1].mean()
                    if ma > 0.0:
                        ext = closes[i] / ma - 1.0
                        trig = ext >= _overext_pct(i) and _overext_rev_ok(i)
                        # Unconditional blow-off FLOOR: at extreme extension the top-sell fires
                        # regardless of the reversal/divergence confirmation (keeps the proven
                        # +14% blow-off exits), while overext_pct + bear_div handles the lower
                        # faded-mid-winner band. None = single-tier (legacy).
                        if cfg.overext_floor_pct is not None and ext >= cfg.overext_floor_pct:
                            trig = True
                        if trig and cfg.overext_skip_ma_slope_pct is not None:
                            p = i - cfg.overext_skip_lookback
                            if p - cfg.overext_ma_window + 1 >= 0:
                                ma_prev = closes[max(0, p - cfg.overext_ma_window + 1):p + 1].mean()
                                if ma_prev > 0.0 and (ma / ma_prev - 1.0) >= cfg.overext_skip_ma_slope_pct:
                                    trig = False  # strong uptrend -> let the runner run
                        if trig and market_bull_dates is not None and pd.Timestamp(dates[i]) in market_bull_dates:
                            trig = False  # strong MARKET bull -> let runners run with the tape
                        if trig:
                            if cfg.overext_trail_pct is not None:
                                overext_armed = True  # arm tight trail; let the runner run
                            else:
                                reason = "overext"
                                break

            elif exit_rule == "max_hold":
                if hold_bars >= cfg.max_hold_bars:
                    reason = "max_hold"
                    break

        # DEAD-MONEY / STALE exit (lowest priority): an in-profit trade that hasn't made a new high
        # in stale_exit_bars bars is dead money -> free it to lift per-bar. Reactive, causal.
        if (reason is None and cfg.stale_exit_bars is not None and hold_bars >= cfg.min_hold_bars
                and (peak_high / entry_fill - 1.0) >= cfg.stale_exit_min_gain
                and (i - last_new_high_idx) >= cfg.stale_exit_bars):
            reason = "stale"

        # Conditional TAKE-PROFIT: lock the pop at a per-trade target (resting limit, fills AT target
        # intraday). Target conditioned on the mfe head (score4): low predicted peak -> tight TP, high
        # -> loose (ride). Ahead of the next-bar exit fill so it does not give back the band. Causal.
        if (reason is None and cfg.take_profit_k is not None and hold_bars >= cfg.min_hold_bars
                and emfe is not None and entry_signal_idx >= 0
                and not np.isnan(emfe[entry_signal_idx])):
            tp_pct = float(np.clip(cfg.take_profit_k * emfe[entry_signal_idx],
                                   cfg.take_profit_floor, cfg.take_profit_cap))
            tp_price = entry_fill * (1.0 + tp_pct)
            if highs[i] >= tp_price:
                exit_fill = cfg.cost.fill_sell(tp_price)
                gross = exit_fill / entry_fill - 1.0
                net = gross - cfg.cost.round_trip_cost()
                trades.append(Trade(
                    symbol=sym, entry_date=pd.Timestamp(dates[entry_idx]),
                    entry_price=float(entry_fill), exit_date=pd.Timestamp(dates[i]),
                    exit_price=float(exit_fill), holding_days=int(i - entry_idx),
                    pnl_pct=float(net), exit_reason="take_profit",
                    entry_signal_date=pd.Timestamp(dates[entry_signal_idx])))
                in_pos = False
                pop_evaluated = False
                pop_is_weak = False
                overext_armed = False
                last_exit_idx = i
                last_exit_price = float(exit_fill)
                last_exit_reason = "take_profit"
                if net < 0:
                    last_loss_exit_idx = i
                i = i + 1
                continue

        if reason is not None:
            if cfg.exit_rally_pct is not None and (
                    not cfg.exit_rally_signal_only or reason == "signal"):
                # Sell into strength: limit above close[i]; fill when a later HIGH reaches it
                # within the window, else market-fill at the window-end close (must exit).
                limit = closes[i] * (1.0 + cfg.exit_rally_pct)
                j_end = min(i + cfg.exit_rally_window, n - 1)
                hit = -1
                for j in range(i + 1, j_end + 1):
                    if highs[j] >= limit:
                        hit = j
                        break
                if hit >= 0:
                    exit_idx = hit
                    exit_fill = cfg.cost.fill_sell(limit)
                else:
                    exit_idx = j_end
                    exit_fill = cfg.cost.fill_sell(closes[exit_idx])
                gross = exit_fill / entry_fill - 1.0
                net = gross - cfg.cost.round_trip_cost()
                trades.append(
                    Trade(
                        symbol=sym,
                        entry_date=pd.Timestamp(dates[entry_idx]),
                        entry_price=float(entry_fill),
                        exit_date=pd.Timestamp(dates[exit_idx]),
                        exit_price=float(exit_fill),
                        holding_days=int(exit_idx - entry_idx),
                        pnl_pct=float(net),
                        exit_reason=reason,
                        entry_signal_date=pd.Timestamp(dates[entry_signal_idx]),
                    )
                )
                in_pos = False
                pop_evaluated = False
                pop_is_weak = False
                overext_armed = False
                last_exit_idx = exit_idx
                last_exit_price = float(exit_fill)
                last_exit_reason = reason
                if net < 0:
                    last_loss_exit_idx = exit_idx
                i = exit_idx
                continue
            exit_idx = min(i + 1, n - 1)
            # Determine exit fill price: use exit_bar_fill_type if set, else use entry_bar_fill_type
            exit_fill_type = cfg.exit_bar_fill_type or cfg.entry_bar_fill_type
            if exit_fill_type in ["close_same", "close_next"]:
                exit_fill = cfg.cost.fill_sell(closes[exit_idx])
            else:
                exit_fill = cfg.cost.fill_sell(opens[exit_idx])
            gross = exit_fill / entry_fill - 1.0
            net = gross - cfg.cost.round_trip_cost()
            trades.append(
                Trade(
                    symbol=sym,
                    entry_date=pd.Timestamp(dates[entry_idx]),
                    entry_price=float(entry_fill),
                    exit_date=pd.Timestamp(dates[exit_idx]),
                    exit_price=float(exit_fill),
                    holding_days=int(exit_idx - entry_idx),
                    pnl_pct=float(net),
                    exit_reason=reason,
                    entry_signal_date=pd.Timestamp(dates[entry_signal_idx]),
                )
            )
            in_pos = False
            pop_evaluated = False
            pop_is_weak = False
            overext_armed = False
            last_exit_idx = exit_idx
            last_exit_price = float(exit_fill)
            if net < 0:
                last_loss_exit_idx = exit_idx
            i = exit_idx
            continue

        i += 1

    if in_pos:
        exit_idx = n - 1
        # Determine exit fill price for forced close
        exit_fill_type = cfg.exit_bar_fill_type or cfg.entry_bar_fill_type
        if exit_fill_type in ["close_same", "close_next"]:
            exit_fill = cfg.cost.fill_sell(closes[exit_idx])
        else:
            exit_fill = (
                cfg.cost.fill_sell(opens[exit_idx])
                if exit_idx < n - 1
                else cfg.cost.fill_sell(closes[exit_idx])
            )
        gross = exit_fill / entry_fill - 1.0
        net = gross - cfg.cost.round_trip_cost()
        trades.append(
            Trade(
                symbol=sym,
                entry_date=pd.Timestamp(dates[entry_idx]),
                entry_price=float(entry_fill),
                exit_date=pd.Timestamp(dates[exit_idx]),
                exit_price=float(exit_fill),
                holding_days=int(exit_idx - entry_idx),
                pnl_pct=float(net),
                exit_reason="end_of_data",
                entry_signal_date=pd.Timestamp(dates[entry_signal_idx]),
            )
        )

    # Pyramid-to-revealed-winner: add size to trades that proved an early surge by bar K. The add is
    # priced at that bar's close (+slippage), shares the base exit, and pays its own round-trip cost.
    if cfg.pyramid_add_units > 0 and trades and n > 1:
        _K = max(1, int(cfg.pyramid_add_bars))
        _thr = float(cfg.pyramid_add_min_ret)
        _a = float(cfg.pyramid_add_units)
        _rtc = cfg.cost.round_trip_cost()
        _didx = {pd.Timestamp(dates[i]): i for i in range(n)}
        for tr in trades:
            ei = _didx.get(pd.Timestamp(tr.entry_date))
            xi = _didx.get(pd.Timestamp(tr.exit_date))
            if ei is None or xi is None or ei + _K >= n or xi <= ei + _K:
                continue
            _ab = ei + _K
            if tr.entry_price <= 0 or closes[_ab] / tr.entry_price - 1.0 < _thr:
                continue
            _addfill = cfg.cost.fill_buy(float(closes[_ab]))
            if _addfill <= 0:
                continue
            _addnet = tr.exit_price / _addfill - 1.0 - _rtc
            tr.pnl_pct = float(tr.pnl_pct + _a * _addnet)
            tr.weight = 1.0 + _a
            tr.notional = 1.0 + _a
    return trades


def _assert_viable_exit(cfg: EngineConfig, sig: pd.DataFrame) -> None:
    """Fail loud if no exit mechanism can ever close a position.

    A position is closed only by an exit rule listed in ``cfg.exit_priority`` that is
    actually able to fire:
      - ``max_hold``: always fires (bar-count timer, finite ``max_hold_bars``)
      - ``hard_stop``: only if ``hard_stop_pct`` is not None
      - ``trailing_stop``: only if ``trailing_stop_pct`` is not None
      - ``signal``: only if ``signal_exit_enabled`` AND the signals contain a sell (-1)

    Without any of these every position rides to ``end_of_data`` — a silent
    buy-and-hold that inflates returns (the ``*_noexit`` bug: exit_priority=["signal"]
    cloned onto a strategy with no exit component, so no -1 signal ever fires).
    Reject it loudly instead of producing a misleading run.
    """
    has_sell_signal = bool((sig["signal"] < 0).any())
    for rule in cfg.exit_priority:
        if rule == "max_hold":
            return
        if rule == "hard_stop" and cfg.hard_stop_pct is not None:
            return
        if rule == "trailing_stop" and cfg.trailing_stop_pct is not None:
            return
        if rule == "signal" and cfg.signal_exit_enabled and has_sell_signal:
            return
    raise ValueError(
        "No viable exit path: every position would ride to end_of_data "
        "(silent buy-and-hold). "
        f"exit_priority={cfg.exit_priority}, hard_stop_pct={cfg.hard_stop_pct}, "
        f"signal_exit_enabled={cfg.signal_exit_enabled}, "
        f"sell_signals_present={has_sell_signal}. "
        "Add 'max_hold' or 'hard_stop' (with hard_stop_pct set) to exit_priority, "
        "or give the strategy an exit component that emits sell (-1) signals."
    )


def run_backtest(
    signals: pd.DataFrame,
    ohlcv: pd.DataFrame,
    cfg: EngineConfig | None = None,
) -> list[Trade]:
    """Execute backtest. Signals and OHLCV must be aligned on (symbol, date)."""
    if cfg is None:
        raise ValueError(
            "run_backtest requires an explicit EngineConfig — no silent default. "
            "Declare max_hold_bars and hard_stop_pct (hard_stop_pct=None disables the stop)."
        )
    sig = _ensure_signals(signals)
    bars = _ensure_ohlcv(ohlcv)
    _assert_viable_exit(cfg, sig)

    # B-channel side log: reset per run so it reflects THIS call only (see BCH_ENTRY_LOG).
    # Guarded so the knob-off path never touches module state (byte-parity).
    if cfg.entry_bchannel_z is not None:
        BCH_ENTRY_LOG.clear()
    MISSED_SIGNAL_LOG.clear()

    market_drop_dates: set[pd.Timestamp] | None = None
    if cfg.exit_market_gate_enabled:
        market_drop_dates = _market_drop_dates(
            bars,
            cfg.exit_market_drop_window,
            cfg.exit_market_drop_threshold,
            cfg.exit_market_drop_mode,
            cfg.exit_market_z_lookback,
        )

    market_weak_dates: set[pd.Timestamp] | None = None
    if cfg.entry_market_gate_enabled:
        market_weak_dates = _market_drop_dates(
            bars,
            cfg.entry_market_window,
            cfg.entry_market_threshold,
            cfg.entry_market_mode,
            cfg.entry_market_z_lookback,
        )
        if cfg.entry_market_abs_floor is not None:
            # Union an absolute-drop catch for the sustained bear the adaptive zscore misses.
            market_weak_dates = market_weak_dates | _market_drop_dates(
                bars, cfg.entry_market_window, cfg.entry_market_abs_floor, "cumret",
                cfg.entry_market_z_lookback,
            )

    market_bull_dates: set[pd.Timestamp] | None = None
    if cfg.overext_skip_bull_enabled:
        market_bull_dates = _market_strong_dates(
            bars,
            cfg.overext_bull_window,
            cfg.overext_bull_threshold,
            cfg.overext_bull_mode,
            cfg.overext_bull_z_lookback,
        )

    market_chop_dates: set[pd.Timestamp] | None = None
    if cfg.entry_market_chop_enabled:
        market_chop_dates = _market_chop_dates(
            bars,
            cfg.entry_market_chop_window,
            cfg.entry_market_chop_threshold,
        )

    market_snr_dates: set[pd.Timestamp] | None = None
    if cfg.exit_snr_extend_threshold is not None:
        market_snr_dates = _market_snr_dates(
            bars, cfg.exit_snr_extend_window, cfg.exit_snr_extend_threshold
        )


    use_xscore = cfg.trailing_score_k is not None and "exit_score" in sig.columns
    use_escore = ((cfg.signal_exit_skip_if_entry_z is not None or cfg.entry_pullback_conv_head_w > 0)
                  and "score" in sig.columns)
    use_ecsr = (
        (cfg.tier2_csr_threshold is not None or cfg.csr_hard_stop_pct is not None)
        and "entry_csr" in sig.columns
    )
    use_emfe = (cfg.mfe_act_k is not None or cfg.entry_pullback_conv_mfe_w > 0
                or cfg.take_profit_k is not None) and "score4" in sig.columns
    use_es3 = (cfg.signal_exit_skip_if_score3_z is not None
               or cfg.signal_exit_hold_min_score3_z is not None
               or cfg.entry_pullback_conv_score3_w > 0) and "score3" in sig.columns
    use_swing = cfg.ml_swing_trail_thr is not None and "swing_top_prob" in sig.columns
    use_bot = (cfg.bot_ride_k is not None or cfg.bot_deepen_k is not None
               or cfg.bot_shallow_k > 0 or cfg.entry_bchannel_z is not None) and "score6" in sig.columns
    trades: list[Trade] = []
    for sym, g in bars.groupby("symbol", sort=False):
        g = g.reset_index(drop=True)
        sym_sig = sig[sig["symbol"] == sym]
        sig_map = dict(zip(sym_sig["date"], sym_sig["signal"]))
        xscore = None
        if use_xscore:
            xmap = dict(zip(sym_sig["date"], sym_sig["exit_score"]))
            xscore = g["date"].map(xmap).to_numpy(dtype=float)
        escore = None
        if use_escore:
            emap = dict(zip(sym_sig["date"], sym_sig["score"]))
            escore = g["date"].map(emap).to_numpy(dtype=float)
        ecsr = None
        if use_ecsr:
            cmap = dict(zip(sym_sig["date"], sym_sig["entry_csr"]))
            ecsr = g["date"].map(cmap).to_numpy(dtype=float)
        emfe = None
        if use_emfe:
            mmap = dict(zip(sym_sig["date"], sym_sig["score4"]))
            emfe = g["date"].map(mmap).to_numpy(dtype=float)
        es3 = None
        if use_es3:
            s3map = dict(zip(sym_sig["date"], sym_sig["score3"]))
            es3 = g["date"].map(s3map).to_numpy(dtype=float)
        swing_prob = None
        if use_swing:
            swmap = dict(zip(sym_sig["date"], sym_sig["swing_top_prob"]))
            swing_prob = g["date"].map(swmap).to_numpy(dtype=float)
        bot = None
        if use_bot:
            bbmap = dict(zip(sym_sig["date"], sym_sig["score6"]))
            bot = g["date"].map(bbmap).to_numpy(dtype=float)
        trades.extend(
            _run_symbol(str(sym), g, sig_map, cfg, market_drop_dates, market_weak_dates,
                        market_bull_dates, xscore, escore, ecsr, emfe, market_chop_dates, es3=es3,
                        market_snr_dates=market_snr_dates, swing_prob=swing_prob, bot=bot)
        )
    return trades


def _market_snr_dates(
    bars: pd.DataFrame,
    window: int,
    threshold: float,
) -> set[pd.Timestamp]:
    """Causal set of dates the universe is in a clean-TREND (high signal-to-noise) regime.
    SNR = rolling(window) universe-mean return / cross-sectional dispersion of per-symbol window
    returns. High SNR = broad market trending up with low dispersion = winners run. Causal."""
    piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    rets = piv.pct_change()
    uni = rets.mean(axis=1).rolling(window).sum()        # universe trend (window return)
    disp = rets.rolling(window).sum().std(axis=1)         # cross-sectional dispersion (noise)
    snr = uni / (disp + 1e-9)
    return set(snr.index[snr >= threshold])


def _market_strong_dates(
    bars: pd.DataFrame,
    window: int,
    threshold: float,
    mode: str = "zscore",
    z_lookback: int = 60,
) -> set[pd.Timestamp]:
    """Causal set of dates the EW universe is in a strong UP regime (mirror of _market_drop_dates).
    zscore: window-return z >= threshold; trend: EW index level >= z_lookback-bar SMA*(1+threshold).
    """
    piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    mret = piv.pct_change().mean(axis=1)
    roll = mret.rolling(window).sum()
    if mode == "zscore":
        mu = roll.rolling(z_lookback).mean()
        sd = roll.rolling(z_lookback).std()
        z = (roll - mu) / (sd + 1e-9)
        return set(z.index[z >= threshold])
    if mode == "trend":
        lvl = (1.0 + mret.fillna(0.0)).cumprod()
        ma = lvl.rolling(z_lookback).mean()
        return set(lvl.index[lvl >= ma * (1.0 + threshold)])
    raise ValueError(f"overext_bull_mode must be 'zscore'|'trend', got {mode!r}")


def _market_drop_dates(
    bars: pd.DataFrame,
    window: int,
    threshold: float,
    mode: str = "cumret",
    z_lookback: int = 60,
) -> set[pd.Timestamp]:
    """Causal set of dates on which the equal-weight universe is in a sharp drop.

    Builds an equal-weight market proxy (mean daily close-to-close return across all
    symbols available on each date) and takes its rolling `window`-bar sum. In "cumret"
    mode the dates where that cumulative return is <= `threshold` (a fixed % drop) are
    returned. In "zscore" mode the window-return is standardized against its own trailing
    `z_lookback`-bar distribution and dates where z <= `threshold` fire — a vol-normalized
    washout that adapts across calm vs crisis regimes. Both are causal: every rolling stat
    at date t uses only data up to and including t (same-bar info, like the signal itself).
    """
    piv = bars.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    rets = piv.pct_change()
    mret = rets.mean(axis=1)  # equal-weight across symbols trading that day
    roll = mret.rolling(window).sum()
    if mode == "zscore":
        mu = roll.rolling(z_lookback).mean()
        sd = roll.rolling(z_lookback).std()
        z = (roll - mu) / (sd + 1e-9)
        return set(z.index[z <= threshold])
    if mode == "trend":
        # market-TREND regime (distinct from a sharp washout): the EW index LEVEL sits below
        # its trailing z_lookback-bar SMA by >= |threshold| — a slow grind-down bear, which the
        # window-return washout modes miss. Causal cumulative index + rolling MA.
        lvl = (1.0 + mret.fillna(0.0)).cumprod()
        ma = lvl.rolling(z_lookback).mean()
        return set(lvl.index[lvl <= ma * (1.0 + threshold)])
    if mode != "cumret":
        raise ValueError(
            f"market_drop_mode must be 'cumret'|'zscore'|'trend', got {mode!r}"
        )
    return set(roll.index[roll <= threshold])


def _market_chop_dates(
    bars: pd.DataFrame,
    window: int,
    threshold: float,
) -> set[pd.Timestamp]:
    """Causal set of dates the EW universe is in a CHOPPY (low trend-efficiency) regime.

    Builds the equal-weight market index level (cumprod of mean daily return) and computes
    Kaufman's Efficiency Ratio over `window` bars: ER = |level[t]-level[t-w]| / sum|level diff|
    over the window. ER -> 1 = a clean directional trend (up OR down); ER -> 0 = a whippy,
    directionless tape that round-trips. Dates with ER <= threshold are flagged choppy so the
    entry gate can skip new risk there (the fast-fail churn cohort). Causal: every rolling stat
    at date t uses only data up to and including t.
    """
    piv = bars.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    # Replace inf (a 0/near-0 close poisons the cumprod) and clip extremes, mirroring the
    # EW-index build in _market_drop_dates' caller — without this one bad bar NaNs the whole tail.
    mret = (piv.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .clip(-0.5, 0.5)
            .mean(axis=1))  # equal-weight across symbols trading that day
    lvl = (1.0 + mret.fillna(0.0)).cumprod()
    direction = (lvl - lvl.shift(window)).abs()
    volatility = lvl.diff().abs().rolling(window).sum()
    eff = direction / (volatility + 1e-9)
    return set(eff.index[eff <= threshold])


def trades_to_dataframe(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = [asdict(t) for t in trades]
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "entry_date",
                "entry_price",
                "exit_date",
                "exit_price",
                "holding_days",
                "pnl_pct",
                "exit_reason",
                "entry_signal_date",
                "side",
                "weight",
                "notional",
            ]
        )
    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["entry_signal_date"] = pd.to_datetime(df["entry_signal_date"])
    return df


# ── Legacy compatibility shim ──────────────────────────────────────────────────
# backtest_unified is the pre-refactor entry point used by lineage_backtests.py
# runners (v28–v42). Kept here so legacy runners work without modification.

import numpy as _np_compat
from collections import defaultdict as _defaultdict_compat

from .defaults import (
    DEFAULT_PARAMS as _DEFAULT_PARAMS_COMPAT,
    FEATURE_DEFAULTS as _FEATURE_DEFAULTS,
    RULE_PRIORITY_SYMBOLS as _RULE_PRIORITY_SYMBOLS,
    SCORE5_RISKY_SYMBOLS as _SCORE5_RISKY_SYMBOLS,
    SYMBOL_PROFILES as _SYMBOL_PROFILES,
)
from .indicators import (
    compute_indicators as _compute_indicators,
    detect_trend_strength as _detect_trend_strength,
    get_regime_adapter as _get_regime_adapter,
)
def backtest_unified(y_pred, returns, df_test, feature_cols, y_pred_exit=None, **config):
    cfg = {**_DEFAULT_PARAMS_COMPAT, **config}

    initial_capital = cfg["initial_capital"]
    commission = cfg["commission"]
    tax = cfg["tax"]
    record_trades = cfg["record_trades"]
    use_model_b_exit = cfg.get("use_model_b_exit", False)
    exit_mode = cfg.get("exit_mode", "rule") if use_model_b_exit else "rule"
    mod_a = cfg["mod_a"]; mod_b = cfg["mod_b"]; mod_c = cfg["mod_c"]; mod_d = cfg["mod_d"]
    mod_e = cfg["mod_e"]; mod_f = cfg["mod_f"]; mod_g = cfg["mod_g"]; mod_h = cfg["mod_h"]
    mod_i = cfg["mod_i"]; mod_j = cfg["mod_j"]

    fast_exit_strong = cfg["fast_exit_strong"]
    fast_exit_moderate = cfg["fast_exit_moderate"]
    fast_exit_weak = cfg["fast_exit_weak"]
    fast_exit_hb_buffer = cfg["fast_exit_hb_buffer"]
    peak_protect_strong_threshold = cfg["peak_protect_strong_threshold"]
    peak_protect_normal_threshold = cfg["peak_protect_normal_threshold"]
    hard_cap_weak = cfg["hard_cap_weak"]
    hard_cap_moderate_mult = cfg["hard_cap_moderate_mult"]
    hard_cap_strong_mult = cfg["hard_cap_strong_mult"]
    hard_cap_strong_floor = cfg["hard_cap_strong_floor"]
    hard_cap_moderate_floor = cfg["hard_cap_moderate_floor"]
    time_decay_bars = cfg["time_decay_bars"]
    time_decay_mult = cfg["time_decay_mult"]

    v22_mode = cfg["v22_mode"]

    patch_smart_hardcap = cfg["patch_smart_hardcap"]
    patch_pp_restore = cfg["patch_pp_restore"]
    patch_long_horizon = cfg["patch_long_horizon"]
    patch_symbol_tuning = cfg["patch_symbol_tuning"]
    patch_rule_ensemble = cfg["patch_rule_ensemble"]
    patch_noise_filter = cfg["patch_noise_filter"]
    patch_adaptive_hardcap = cfg["patch_adaptive_hardcap"]
    patch_pp_2of3 = cfg["patch_pp_2of3"]

    v26_wider_hardcap = cfg["v26_wider_hardcap"]
    v26_relaxed_entry = cfg["v26_relaxed_entry"]
    v26_skip_choppy = cfg["v26_skip_choppy"]
    v26_extended_hold = cfg["v26_extended_hold"]
    v26_strong_rule_ensemble = cfg["v26_strong_rule_ensemble"]
    v26_min_position = cfg["v26_min_position"]
    v26_score5_penalty = cfg["v26_score5_penalty"]
    v26_hardcap_confirm_strong = cfg["v26_hardcap_confirm_strong"]

    v27_selective_choppy = cfg["v27_selective_choppy"]
    v27_hardcap_two_step = cfg["v27_hardcap_two_step"]
    v27_rule_priority = cfg["v27_rule_priority"]
    v27_dynamic_score5_penalty = cfg["v27_dynamic_score5_penalty"]
    v27_trend_persistence_hold = cfg["v27_trend_persistence_hold"]

    v28_early_wave_filter = cfg["v28_early_wave_filter"]
    v28_crash_guard = cfg["v28_crash_guard"]
    v28_wave_acceleration_entry = cfg["v28_wave_acceleration_entry"]
    v28_early_loss_cut = cfg["v28_early_loss_cut"]
    v28_cycle_peak_exit = cfg["v28_cycle_peak_exit"]
    v28_early_loss_cut_threshold = cfg["v28_early_loss_cut_threshold"]
    v28_early_loss_cut_days = cfg["v28_early_loss_cut_days"]

    v29_adaptive_peak_lock = cfg["v29_adaptive_peak_lock"]
    v29_apl_trigger = cfg["v29_adaptive_peak_lock_trigger"]
    v29_apl_keep = cfg["v29_adaptive_peak_lock_keep"]
    v29_atr_velocity_exit = cfg["v29_atr_velocity_exit"]
    v29_atr_velocity_k = cfg["v29_atr_velocity_k"]
    v29_atr_velocity_min_profit = cfg["v29_atr_velocity_min_profit"]
    v29_tighter_trail_high_profit = cfg["v29_tighter_trail_high_profit"]
    v29_high_profit_trigger = cfg["v29_high_profit_trigger"]
    v29_high_profit_trail = cfg["v29_high_profit_trail"]
    v29_reversal_after_peak = cfg["v29_reversal_after_peak"]
    v29_reversal_peak_trigger = cfg["v29_reversal_peak_trigger"]
    v29_reversal_ret2_threshold = cfg["v29_reversal_ret2_threshold"]
    v29_breakout_strength_entry = cfg["v29_breakout_strength_entry"]
    v29_relstrength_filter = cfg["v29_relstrength_filter"]
    v29_rs_ret20_threshold = cfg["v29_rs_ret20_threshold"]
    v29_peak_lock_high_beta_only = cfg["v29_peak_lock_high_beta_only"]
    v29_profit_safety_net = cfg["v29_profit_safety_net"]
    v29_profit_safety_trigger = cfg["v29_profit_safety_trigger"]
    v29_hardcap_after_peak = cfg["v29_hardcap_after_peak"]
    v29_hardcap_after_peak_trigger = cfg["v29_hardcap_after_peak_trigger"]
    v29_hardcap_after_peak_floor = cfg["v29_hardcap_after_peak_floor"]

    # V30 flags
    v30_peak_proximity_filter  = cfg["v30_peak_proximity_filter"]
    v30_peak_prox_dist_threshold = cfg["v30_peak_prox_dist_threshold"]
    v30_peak_prox_rally10_min  = cfg["v30_peak_prox_rally10_min"]
    v30_rally_extension_filter = cfg["v30_rally_extension_filter"]
    v30_rally10_hard_block     = cfg["v30_rally10_hard_block"]
    v30_rally20_hard_block     = cfg["v30_rally20_hard_block"]
    v30_pullback_only_entry    = cfg["v30_pullback_only_entry"]
    v30_pullback_min_pct       = cfg["v30_pullback_min_pct"]
    v30_rally_position_scaling = cfg["v30_rally_position_scaling"]
    v30_rps_tier1_rally        = cfg["v30_rps_tier1_rally"]
    v30_rps_tier2_rally        = cfg["v30_rps_tier2_rally"]
    v30_signal_exit_defer      = cfg["v30_signal_exit_defer"]
    v30_sed_defer_bars         = cfg["v30_sed_defer_bars"]
    v30_sed_min_cum_ret        = cfg["v30_sed_min_cum_ret"]
    v30_momentum_hold_override = cfg["v30_momentum_hold_override"]
    v30_mho_min_profit         = cfg["v30_mho_min_profit"]
    v30_mho_rsi_max            = cfg["v30_mho_rsi_max"]
    v30_chandelier_trail       = cfg["v30_chandelier_trail"]
    v30_chand_atr_mult         = cfg["v30_chand_atr_mult"]
    v30_chand_profit_trigger   = cfg["v30_chand_profit_trigger"]
    v30_atr_aware_hardcap      = cfg["v30_atr_aware_hardcap"]
    v30_atr_hc_mult            = cfg["v30_atr_hc_mult"]
    v30_atr_hc_floor           = cfg["v30_atr_hc_floor"]
    v30_atr_hc_ceiling         = cfg["v30_atr_hc_ceiling"]
    v30_hardcap_two_step_v2    = cfg["v30_hardcap_two_step_v2"]
    v30_hc2_step1_loss         = cfg["v30_hc2_step1_loss"]
    v30_hc2_step2_loss         = cfg["v30_hc2_step2_loss"]
    v30_regime_aware_hardcap   = cfg["v30_regime_aware_hardcap"]
    v30_rah_choppy_cap         = cfg["v30_rah_choppy_cap"]
    v30_rah_trending_cap       = cfg["v30_rah_trending_cap"]

    # V31 flags
    v31_peak_chasing_guard     = cfg.get("v31_peak_chasing_guard", False)
    v31_pcg_ret5d_thresh       = cfg.get("v31_pcg_ret5d_thresh", 0.08)
    v31_pcg_dist_thresh        = cfg.get("v31_pcg_dist_thresh", 10.0)
    v31_pcg_action             = cfg.get("v31_pcg_action", "half_size")  # "half_size" | "skip"
    v31_adaptive_defer         = cfg.get("v31_adaptive_defer", False)
    v31_ad_min_cum_ret         = cfg.get("v31_ad_min_cum_ret", 0.02)
    v31_ad_max_bars            = cfg.get("v31_ad_max_bars", 7)
    v31_ad_use_ema_confirm     = cfg.get("v31_ad_use_ema_confirm", True)
    v31_hardcap_after_profit   = cfg.get("v31_hardcap_after_profit", False)
    v31_hap_profit_trigger     = cfg.get("v31_hap_profit_trigger", 0.05)
    v31_hap_floor              = cfg.get("v31_hap_floor", -0.03)
    v31_profile_sizing         = cfg.get("v31_profile_sizing", False)
    v31_ps_momentum_mult       = cfg.get("v31_ps_momentum_mult", 1.4)
    v31_ps_highbeta_mult       = cfg.get("v31_ps_highbeta_mult", 1.2)
    v31_ps_defensive_mult      = cfg.get("v31_ps_defensive_mult", 0.75)
    v31_ps_bank_mult           = cfg.get("v31_ps_bank_mult", 0.85)
    v31_short_hold_exit_filter = cfg.get("v31_short_hold_exit_filter", False)
    v31_shef_min_hold          = cfg.get("v31_shef_min_hold", 10)
    v31_shef_min_pnl           = cfg.get("v31_shef_min_pnl", -0.03)
    v31_enriched_log           = cfg.get("v31_enriched_log", False)

    # V32 flags
    # A: HAP preempt — thoát TRƯỚC hard_cap nếu đã từng có lãi >= trigger rồi rớt về floor
    v32_hap_preempt            = cfg.get("v32_hap_preempt", False)
    v32_hap_pre_trigger        = cfg.get("v32_hap_pre_trigger", 0.05)   # max_profit >= 5%
    v32_hap_pre_floor          = cfg.get("v32_hap_pre_floor", -0.03)    # cur_ret <= -3%
    # B: Trend-weak oversold exit — khi trend=weak + dist_sma20 < threshold + còn lãi → exit sớm
    v32_weak_oversold_exit     = cfg.get("v32_weak_oversold_exit", False)
    v32_woe_dist_thresh        = cfg.get("v32_woe_dist_thresh", -0.07)  # dist_sma20 < -7%
    v32_woe_min_profit         = cfg.get("v32_woe_min_profit", 0.0)     # only when pnl > 0
    v32_woe_hold_min           = cfg.get("v32_woe_hold_min", 5)         # hold >= 5 days
    # C: Dynamic hard_cap tighten — khi dist_sma20 < -8% thì cap floor lỏng hơn không còn giá trị
    v32_dynamic_hc_dist        = cfg.get("v32_dynamic_hc_dist", False)
    v32_dhc_dist_thresh        = cfg.get("v32_dhc_dist_thresh", -0.08)  # dist_sma20 < -8%
    v32_dhc_tight_cap          = cfg.get("v32_dhc_tight_cap", -0.07)    # exit at -7% nếu thỏa mãn
    # D: Profit ratchet exit — khi đạt max_profit >= trigger rồi rớt về keep * max_profit
    v32_profit_ratchet         = cfg.get("v32_profit_ratchet", False)
    v32_pr_trigger             = cfg.get("v32_pr_trigger", 0.08)        # max_profit >= 8%
    v32_pr_keep                = cfg.get("v32_pr_keep", 0.30)           # sàn = 30% của max_profit
    # E: Signal exit weak-trend early — khi signal + trend=weak + dist_sma20 < -5% → không block
    v32_signal_weak_exit       = cfg.get("v32_signal_weak_exit", False)
    v32_swe_dist_thresh        = cfg.get("v32_swe_dist_thresh", -0.05)  # dist_sma20 < -5%

    # V33 flags
    # A: Multi-tier trailing ratchet — tighten floor progressively as max_profit grows
    v33_trailing_ratchet       = cfg.get("v33_trailing_ratchet", False)
    v33_tr_tier1_trigger       = cfg.get("v33_tr_tier1_trigger", 0.12)  # >= 12%: floor = tier1_keep * max_profit
    v33_tr_tier1_keep          = cfg.get("v33_tr_tier1_keep", 0.40)
    v33_tr_tier2_trigger       = cfg.get("v33_tr_tier2_trigger", 0.20)  # >= 20%: floor = tier2_keep * max_profit
    v33_tr_tier2_keep          = cfg.get("v33_tr_tier2_keep", 0.55)
    v33_tr_tier3_trigger       = cfg.get("v33_tr_tier3_trigger", 0.35)  # >= 35%: floor = tier3_keep * max_profit
    v33_tr_tier3_keep          = cfg.get("v33_tr_tier3_keep", 0.65)

    # B: Trend-reversal exit — khi max_profit>threshold VÀ close<ema8 VÀ rsi thấp → thoát ngay
    v33_trend_rev_exit         = cfg.get("v33_trend_rev_exit", False)
    v33_tre_min_profit         = cfg.get("v33_tre_min_profit", 0.08)    # chỉ sau khi đạt >= 8% profit
    v33_tre_rsi_thresh         = cfg.get("v33_tre_rsi_thresh", 50.0)    # rsi14 < 50
    v33_tre_hold_min           = cfg.get("v33_tre_hold_min", 5)         # hold >= 5 ngày

    # C: Recovery-peak entry filter — block khi giá đã tăng nhanh + dist_sma20 vẫn dương mà trend không mạnh
    v33_recovery_peak_filter   = cfg.get("v33_recovery_peak_filter", False)
    v33_rpf_ret10_thresh       = cfg.get("v33_rpf_ret10_thresh", 0.12)  # ret 10 ngày > 12%
    v33_rpf_dist_sma20_thresh  = cfg.get("v33_rpf_dist_sma20_thresh", 0.03)  # dist_sma20 > 3%
    v33_rpf_require_weak       = cfg.get("v33_rpf_require_weak", True)  # chỉ block nếu trend != strong

    # D: HAP consecutive drop — HAP preempt chỉ trigger sau N ngày giảm liên tiếp (tránh bán đáy spike)
    v33_hap_consec_drop        = cfg.get("v33_hap_consec_drop", False)
    v33_hcd_min_days           = cfg.get("v33_hcd_min_days", 2)         # phải giảm ít nhất 2 ngày liên tiếp

    # E: RSI oversold exit block — không thoát bởi signal/hap khi rsi đang oversold (đáy điều chỉnh)
    v33_rsi_oversold_block     = cfg.get("v33_rsi_oversold_block", False)
    v33_rob_rsi_thresh         = cfg.get("v33_rob_rsi_thresh", 32.0)    # block nếu rsi14 < 32 (oversold)
    v33_rob_max_hold           = cfg.get("v33_rob_max_hold", 5)         # chỉ block trong N ngày đầu oversold

    # F: Signal exit confirm — require 2 consecutive signal bars to exit (giảm false exits)
    v33_signal_confirm_exit    = cfg.get("v33_signal_confirm_exit", False)
    v33_sce_min_pnl            = cfg.get("v33_sce_min_pnl", -0.02)     # chỉ block nếu pnl > -2%
    v33_sce_min_profit_seen    = cfg.get("v33_sce_min_profit_seen", 0.03)  # và đã từng có profit >= 3%
    v33_sce_max_hold           = cfg.get("v33_sce_max_hold", 90)        # không block khi hold >= 90d (tránh zombie)

    # V35 flags
    v35_relax_cooldown         = cfg.get("v35_relax_cooldown", False)
    v35_cooldown_after_big_loss = cfg.get("v35_cooldown_after_big_loss", 1)
    v35_cooldown_after_loss    = cfg.get("v35_cooldown_after_loss", 0)
    v35_skip_price_proximity   = cfg.get("v35_skip_price_proximity", False)
    v35_single_bar_signal      = cfg.get("v35_single_bar_signal", False)
    v35_rule_override          = cfg.get("v35_rule_override", False)
    v35_rule_override_min_score = cfg.get("v35_rule_override_min_score", 1)
    v35_hybrid_entry           = cfg.get("v35_hybrid_entry", False)
    v35_hybrid_size            = cfg.get("v35_hybrid_size", 0.5)

    # === V38 flags ===
    # V38b: stall-exit — exit khi giu N ngay ma chua dat dc profit threshold
    v38b_stall_exit            = cfg.get("v38b_stall_exit", False)
    v38b_stall_min_hold        = cfg.get("v38b_stall_min_hold", 6)
    v38b_stall_max_profit      = cfg.get("v38b_stall_max_profit", 0.02)  # max_price_profit < 2%
    v38b_stall_pnl_thresh      = cfg.get("v38b_stall_pnl_thresh", -0.02) # cur_ret < -2%
    # V38c: HA-driven exit — bearish_reversal_signal hoac late_wave + body_shrinking + cur_ret<0
    v38c_ha_exit               = cfg.get("v38c_ha_exit", False)
    v38c_ha_min_hold           = cfg.get("v38c_ha_min_hold", 3)
    v38c_ha_pnl_thresh         = cfg.get("v38c_ha_pnl_thresh", 0.0)     # exit khi cur_ret < 0%
    # V38d: anti-fomo entry filter
    v38d_fomo_filter           = cfg.get("v38d_fomo_filter", False)
    v38d_fomo_ret5d_thresh     = cfg.get("v38d_fomo_ret5d_thresh", 0.06)   # entry_ret_5d > 6% block
    v38d_fomo_dist_thresh      = cfg.get("v38d_fomo_dist_thresh", 0.06)    # dist_sma20 > 6% block
    # V38d: rule co-pilot exit — neu rule_signal=False sau N ngay giu va co loi -> chot
    v38d_copilot_exit          = cfg.get("v38d_copilot_exit", False)
    v38d_copilot_min_hold      = cfg.get("v38d_copilot_min_hold", 4)
    v38d_copilot_min_profit    = cfg.get("v38d_copilot_min_profit", 0.03) # da co profit >=3%

    # === V39 flags ===
    # V39a: signal exit defer mở rộng — signal exit chỉ kích hoạt sau min_hold ngày
    # Giải quyết bucket 21-30d WR=12% (signal exit quá sớm với early_wave fw=8d target)
    v39a_signal_exit_min_hold  = cfg.get("v39a_signal_exit_min_hold", 0)  # 0 = off

    # V39a2: kết hợp thêm rule confirm — signal exit cần MACD<0 AND Close<MA20
    v39a_rule_confirm_exit     = cfg.get("v39a_rule_confirm_exit", False)
    # V39g: chỉ defer khi price_max_profit đã đạt threshold (tránh defer trades stall)
    v39g_rule_confirm_min_maxprofit = cfg.get("v39g_rule_confirm_min_maxprofit", 0.0)

    # V39b: HAP reform — raise trigger + require min_hold trước khi HAP active
    # Giải quyết 87 trades HAP preempt 100% lỗ do bắt đầu sóng quá sớm
    v39b_hap_min_hold          = cfg.get("v39b_hap_min_hold", 0)          # 0 = off (tương thích ngược)
    v39b_hap_trigger           = cfg.get("v39b_hap_trigger", None)        # None = dùng v32_hap_pre_trigger

    # V39d: per-symbol rule-exit hybrid — các mã stable-trend dùng rule exit thay signal exit
    v39d_rule_exit_symbols     = cfg.get("v39d_rule_exit_symbols", set())  # set of symbols

    # --- Indicators ---
    ind = _compute_indicators(df_test, mod_e=mod_e)
    n = ind["n"]
    close = ind["close"]; opn = ind["opn"]; high = ind["high"]; low = ind["low"]; volume = ind["volume"]
    sma10 = ind["sma10"]; sma20 = ind["sma20"]; sma50 = ind["sma50"]; sma100 = ind["sma100"]
    ema8 = ind["ema8"]; macd_line = ind["macd_line"]; macd_hist = ind["macd_hist"]
    atr14 = ind["atr14"]; avg_vol20 = ind["avg_vol20"]
    local_low_20 = ind["local_low_20"]
    ret_2d = ind["ret_2d"]; ret_3d = ind["ret_3d"]
    ret_5d = ind["ret_5d"]; ret_20d = ind["ret_20d"]; ret_60d = ind["ret_60d"]
    ret_acceleration = ind["ret_acceleration"]; days_since_low_10 = ind["days_since_low_10"]
    dist_sma20 = ind["dist_sma20"]
    drop_from_peak_20 = ind["drop_from_peak_20"]; rsi14 = ind["rsi14"]
    stabilized_sideways = ind["stabilized_sideways"]
    consolidation_breakout = ind["consolidation_breakout"]
    secondary_breakout = ind["secondary_breakout"]
    vshape_bypass = ind["vshape_bypass"]
    days_above_ma20 = ind["days_above_ma20"]
    days_above_sma50 = ind["days_above_sma50"]
    rule_signal = ind["rule_signal"]; rule_consecutive = ind["rule_consecutive"]
    dist_from_52w_high = ind["dist_from_52w_high"]
    dates = ind["dates"]; symbols = ind["symbols"]
    feat_arrays = ind["feat_arrays"]

    # V38c: extract HA columns from df_test if available (leading_v4)
    _HA_COLS = ["ha_bearish_reversal_signal", "ha_late_wave",
                "ha_upper_shadow_growing", "ha_body_shrinking",
                "ha_green_streak"]
    ha_arrays = {}
    for c in _HA_COLS:
        if c in df_test.columns:
            arr = df_test[c].values.astype(float)
            arr = np.where(np.isnan(arr), 0.0, arr)
            ha_arrays[c] = arr
        else:
            ha_arrays[c] = np.zeros(n)

    MODEL_B_MIN_HOLD = cfg.get("model_b_min_hold", 3)
    MODEL_B_REQUIRE_TREND_BREAK = cfg.get("model_b_require_trend_break", False)
    MODEL_B_USE_SMA20 = cfg.get("model_b_trend_break_use_sma20", True)
    MODEL_B_USE_EMA8 = cfg.get("model_b_trend_break_use_ema8", True)
    MODEL_B_USE_MACD = cfg.get("model_b_trend_break_use_macd", False)
    MODEL_B_USE_RED_CANDLE = cfg.get("model_b_trend_break_use_red_candle", False)
    MODEL_B_MIN_NEG_SIGNALS = cfg.get("model_b_min_negative_signals", 2)
    MODEL_B_PROFIT_ONLY_IF_LOSING_STRUCTURE = cfg.get("model_b_profit_only_if_losing_structure", False)
    MODEL_B_PROFIT_THRESHOLD = cfg.get("model_b_profit_threshold", 0.0)

    # --- State ---
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    max_equity_in_trade = 0
    max_price_in_trade = 0
    hold_days = 0
    position_size = 1.0
    consecutive_exit_signals = 0
    consecutive_below_ema8 = 0

    hard_cap_pending_bars = 0
    HARD_CAP_CONFIRM_STRONG = 2 if v26_hardcap_confirm_strong else 1
    HARD_CAP_CONFIRM_MODERATE = 1
    HARD_CAP_CONFIRM_WEAK = 0

    pp_pending_bars = 0

    MIN_HOLD = 6
    ZOMBIE_BARS = 14
    PROFIT_LOCK_THRESHOLD = 0.12
    PROFIT_LOCK_MIN = 0.06
    HARD_STOP = 0.08
    ATR_MULT = 1.8
    COOLDOWN_AFTER_BIG_LOSS = 5
    QUICK_REENTRY_WINDOW = 3
    STRONG_TREND_TRAIL_MULT = 0.45
    V22_SIGNAL_HARD_CAP = 0.12

    cooldown_remaining = 0
    last_exit_price = 0
    last_exit_reason = ""
    last_exit_bar = -999
    entry_close = 0
    v30_hc2_halved = False       # track if we already halved position (C2 two-step)
    v30_defer_bars_remaining = 0  # bars left to defer signal exit (B1)

    # V31 state
    v31_defer_bars_remaining = 0  # adaptive defer counter

    # V33 state
    v33_consec_below_ema8 = 0     # consecutive days close < ema8 (for trend_rev_exit)
    v33_prev_signal_exit = False  # last bar also had signal exit (for signal_confirm_exit)

    def gf(name, idx):
        return feat_arrays[name][idx] if idx < n else _FEATURE_DEFAULTS.get(name, 0)

    entry_features = {}
    counters = _defaultdict_compat(int)

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        raw_signal = 1 if pred == 1 else 0
        new_position = raw_signal
        exit_reason = "signal"

        wp = gf("range_position_20d", i)
        dp = gf("dist_to_resistance", i)
        rs = gf("rsi_slope_5d", i)
        vs = gf("vol_surge_ratio", i)
        bs = gf("breakout_setup_score", i)
        hl = gf("higher_lows_count", i)
        od = gf("obv_price_divergence", i)
        bb = gf("bb_width_percentile", i)

        trend = _detect_trend_strength(i, ind)
        regime_cfg = _get_regime_adapter(i, trend, ind, patch_symbol_tuning=patch_symbol_tuning)
        dp_floor = regime_cfg["dp_floor"]
        ret5_hot = regime_cfg["ret5_hot"]
        sym = str(symbols[i]) if i < n else "?"
        profile = _SYMBOL_PROFILES.get(sym, "balanced")

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        # V35: rule trigger (used by hybrid entry / rule_override below)
        rule_trigger_now = (
            i >= 26
            and not np.isnan(macd_hist[i]) and macd_hist[i] > 0
            and not np.isnan(sma20[i]) and close[i] > sma20[i]
            and close[i] > opn[i]
        )

        # === ENTRY LOGIC ===
        quick_reentry = False
        breakout_entry = False
        vshape_entry = False
        v35_hybrid_now = False  # set True when entry triggered by V35 hybrid path

        if new_position == 0 and position == 0 and last_exit_reason == "trailing_stop":
            bars_since_exit = i - last_exit_bar
            if (bars_since_exit <= QUICK_REENTRY_WINDOW and trend in ("strong", "moderate") and
                macd_line[i] > 0 and not np.isnan(sma20[i]) and close[i] > sma20[i]):
                new_position = 1; quick_reentry = True

        bo_quality_ok = True
        if mod_f:
            bo_quality_ok = (macd_hist[i] > 0 and close[i] > opn[i] and
                            not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])

        if new_position == 0 and position == 0 and consolidation_breakout[i] and bo_quality_ok:
            new_position = 1; breakout_entry = True

        if mod_e and new_position == 0 and position == 0 and secondary_breakout[i] and bo_quality_ok:
            new_position = 1; breakout_entry = True; counters["secondary_bo"] += 1

        if mod_a and new_position == 0 and position == 0 and vshape_bypass[i]:
            if not np.isnan(ema8[i]) and close[i] >= ema8[i] * 0.99:
                new_position = 1; vshape_entry = True; counters["vshape"] += 1

        # V35c: hybrid entry — fire when ML didn't but rule trigger fires
        if (v35_hybrid_entry and new_position == 0 and position == 0
                and rule_trigger_now and cooldown_remaining == 0):
            new_position = 1
            v35_hybrid_now = True

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if cooldown_remaining > 0:
                new_position = 0
        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if last_exit_price > 0 and last_exit_reason != "trailing_stop":
                if abs(close[i] / last_exit_price - 1) < 0.03:
                    if not (v35_skip_price_proximity and rule_trigger_now) and not v35_hybrid_now:
                        new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not breakout_entry and not vshape_entry and not v35_hybrid_now:
            prev_pred = int(y_pred[i - 2]) if i >= 2 else 0
            if bs >= 4 and vs > 1.2: pass
            elif trend == "strong" and rs > 0: pass
            elif v26_relaxed_entry and trend == "strong" and rule_consecutive[i] >= 3:
                pass
            elif v35_single_bar_signal and rule_trigger_now:
                pass
            elif prev_pred != 1: new_position = 0

        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            if not np.isnan(sma50[i]) and not np.isnan(sma20[i]):
                if close[i] < sma50[i] and close[i] < sma20[i] and rs <= 0:
                    if bs < 3 and not breakout_entry: new_position = 0

        strong_breakout_context = (trend == "strong" and (bs >= 3 or vs > 1.5 or breakout_entry))
        entry_alpha_ok = True
        entry_score = 0
        if new_position == 1 and position == 0 and not quick_reentry and not vshape_entry:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            near_sma_support = (not np.isnan(sma20[i]) and close[i] <= sma20[i] * 1.02 and close[i] >= sma20[i] * 0.97)
            near_local_low = (not np.isnan(local_low_20[i]) and close[i] <= local_low_20[i] * 1.05)
            in_uptrend_macro = (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] > sma50[i])
            if trend == "strong": min_score = 1
            elif (near_sma_support or near_local_low) and in_uptrend_macro: min_score = 2
            elif in_uptrend_macro and rs > 0: min_score = 2
            else: min_score = 3
            if entry_score < min_score and not breakout_entry: entry_alpha_ok = False
            if wp > 0.9 and rs <= 0 and bs < 2 and trend != "strong" and not breakout_entry: entry_alpha_ok = False
            if bb > 0.85 and bs < 2 and entry_score < 4 and trend != "strong" and not breakout_entry: entry_alpha_ok = False
            if entry_alpha_ok and wp > 0.78 and bb < 0.35 and trend == "weak" and not breakout_entry: entry_alpha_ok = False
            if entry_alpha_ok and dp < dp_floor:
                if entry_score < 4 and not strong_breakout_context: entry_alpha_ok = False

        if new_position == 1 and position == 0 and not vshape_entry:
            if ret_5d[i] > ret5_hot and not strong_breakout_context:
                if not (v35_rule_override and rule_trigger_now and entry_score >= v35_rule_override_min_score) and not v35_hybrid_now:
                    entry_alpha_ok = False
        if new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if drop_from_peak_20[i] <= -0.15 and not stabilized_sideways[i]: entry_alpha_ok = False
        if new_position == 1 and position == 0 and entry_alpha_ok:
            vol_floor = 0.7 * avg_vol20[i] if not np.isnan(avg_vol20[i]) else 0
            if vol_floor > 0 and volume[i] < vol_floor: entry_alpha_ok = False
        if mod_g and new_position == 1 and position == 0 and not vshape_entry and entry_alpha_ok:
            if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and sma20[i] < sma50[i] and
                close[i] < sma50[i] and ret_60d[i] < -0.10):
                entry_alpha_ok = False; counters["bear_blocked"] += 1
        if mod_j and new_position == 1 and position == 0 and not vshape_entry and not breakout_entry and entry_alpha_ok:
            if (not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and
                abs(sma20[i] / sma50[i] - 1) < 0.02 and abs(ret_20d[i]) < 0.06 and
                bb < 0.45 and trend == "weak"):
                entry_alpha_ok = False; counters["chop_blocked"] += 1

        # V26-C: Skip choppy regime entirely
        if v26_skip_choppy and new_position == 1 and position == 0 and entry_alpha_ok:
            if regime_cfg["choppy_regime"]:
                if not (v35_rule_override and rule_trigger_now) and not v35_hybrid_now:
                    entry_alpha_ok = False; counters["v26_choppy_skipped"] += 1

        # V27: selective choppy filter
        if (not v26_skip_choppy) and v27_selective_choppy and new_position == 1 and position == 0 and entry_alpha_ok:
            if regime_cfg["choppy_regime"]:
                vol_ok = (not np.isnan(avg_vol20[i]) and volume[i] >= 0.95 * avg_vol20[i])
                quality_ok = (
                    breakout_entry or
                    (trend in ("strong", "moderate") and bs >= 3 and vs > 1.15 and vol_ok) or
                    (rule_consecutive[i] >= 3 and rs > 0 and vol_ok)
                )
                if not quality_ok:
                    entry_alpha_ok = False
                    counters["v27_choppy_low_quality_skipped"] += 1
                else:
                    counters["v27_choppy_quality_kept"] += 1

        if patch_noise_filter and new_position == 1 and position == 0 and entry_alpha_ok:
            entry_score_nf = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            if trend == "weak" and entry_score_nf < 3 and ret_5d[i] > 0.03:
                entry_alpha_ok = False; counters["noise_filtered"] += 1

        # V28-K1: Early wave filter — block when wave is already mature
        # (price has been rising for >7 of last 10 days AND ret_5d > 8%)
        if v28_early_wave_filter and new_position == 1 and position == 0 and entry_alpha_ok and not vshape_entry:
            wave_mature = (days_since_low_10[i] > 7 and ret_5d[i] > 0.08)
            # Also block if ret_2d is much smaller than ret_5d/2 (momentum slowing)
            wave_exhausted = (ret_5d[i] > 0.06 and ret_2d[i] < 0.005 and ret_2d[i] < ret_5d[i] * 0.15)
            if (wave_mature or wave_exhausted) and not breakout_entry:
                if not (v35_rule_override and rule_trigger_now) and not v35_hybrid_now:
                    entry_alpha_ok = False
                    counters["v28_late_wave_blocked"] += 1

        # V28-K2: Crash guard — block new entries when broad market is crashing
        # Proxy: ret_20d of current symbol < -12% (systemic crash period)
        if v28_crash_guard and new_position == 1 and position == 0 and entry_alpha_ok:
            market_crash = (ret_20d[i] < -0.12)
            if market_crash and not vshape_entry:
                entry_alpha_ok = False
                counters["v28_crash_guard_blocked"] += 1

        # V38d: anti-fomo entry filter — block khi gia da chay nong (entry_ret_5d/dist_sma20 cao)
        # Tru breakout_entry / vshape_entry (giu special-case)
        if v38d_fomo_filter and new_position == 1 and position == 0 and entry_alpha_ok \
                and not breakout_entry and not vshape_entry:
            ret5_now = ret_5d[i] if not np.isnan(ret_5d[i]) else 0.0
            dist_now = dist_sma20[i] if not np.isnan(dist_sma20[i]) else 0.0
            if ret5_now > v38d_fomo_ret5d_thresh or dist_now > v38d_fomo_dist_thresh:
                entry_alpha_ok = False
                counters["v38d_fomo_blocked"] += 1

        # V28-K3: Wave acceleration bonus entry
        # Allow entry when wave is just starting: ret_2d > 0 but ret_5d still low
        if v28_wave_acceleration_entry and new_position == 0 and position == 0:
            early_wave = (ret_2d[i] > 0.015 and ret_3d[i] > 0.02 and
                          ret_5d[i] < 0.05 and ret_acceleration[i] > 0.005)
            accel_quality = (trend in ("strong", "moderate") and rs > 0 and
                             not np.isnan(sma20[i]) and close[i] > sma20[i] and
                             not np.isnan(avg_vol20[i]) and volume[i] > 0.9 * avg_vol20[i])
            if early_wave and accel_quality and not regime_cfg["choppy_regime"]:
                new_position = 1
                position_size = 0.40
                counters["v28_wave_accel_entry"] += 1

        # V30-A1: Peak-proximity filter — block entry when near 20d high AND already rallied
        if v30_peak_proximity_filter and new_position == 1 and position == 0 and not vshape_entry:
            peak_prox = drop_from_peak_20[i] >= v30_peak_prox_dist_threshold  # close to peak
            already_rallied = ret_5d[i] > v30_peak_prox_rally10_min * 0.65 or (
                i >= 10 and close[i] > 0 and close[max(0,i-10)] > 0 and
                close[i] / close[max(0,i-10)] - 1 > v30_peak_prox_rally10_min
            )
            if peak_prox and already_rallied and not breakout_entry:
                entry_alpha_ok = False
                counters["v30_peak_prox_blocked"] += 1

        # V30-A2: Rally-extension filter — block when price has run too far too fast
        if v30_rally_extension_filter and new_position == 1 and position == 0 and entry_alpha_ok and not vshape_entry:
            ret10 = (close[i] / close[max(0,i-10)] - 1) if i >= 10 and close[max(0,i-10)] > 0 else 0
            ret20_val = ret_20d[i]
            if ret10 > v30_rally10_hard_block or ret20_val > v30_rally20_hard_block:
                entry_alpha_ok = False
                counters["v30_rally_ext_blocked"] += 1

        # V30-A3: Pullback-only entry — only enter after at least v30_pullback_min_pct pullback from 5d high
        if v30_pullback_only_entry and new_position == 1 and position == 0 and entry_alpha_ok and not vshape_entry and not breakout_entry:
            high_5d = np.max(high[max(0,i-5):i+1]) if i >= 5 else high[i]
            pullback_pct = (high_5d - close[i]) / high_5d if high_5d > 0 else 0
            if pullback_pct < v30_pullback_min_pct:
                entry_alpha_ok = False
                counters["v30_pullback_blocked"] += 1

        # V29-P5: Breakout strength early entry — clear recent high with dry-then-spike volume
        if v29_breakout_strength_entry and new_position == 0 and position == 0:
            if i >= 20 and not np.isnan(avg_vol20[i]):
                recent_high_10 = np.max(high[max(0, i - 10):i])
                breakout_clean = close[i] > recent_high_10 * 1.005
                dry_vol = np.nanmean(volume[max(0, i - 10):max(0, i - 3)])
                vol_spike = dry_vol > 0 and volume[i] > 1.6 * dry_vol
                acceptable = (ret_5d[i] < 0.08 and trend in ("strong", "moderate") and
                              not np.isnan(sma20[i]) and close[i] > sma20[i])
                if breakout_clean and vol_spike and acceptable and not regime_cfg["choppy_regime"]:
                    new_position = 1
                    position_size = 0.45
                    breakout_entry = True
                    counters["v29_bo_strength_entry"] += 1

        # V29-P6: Relative-strength filter — block new entries when symbol has been weak
        if v29_relstrength_filter and new_position == 1 and position == 0 and not vshape_entry:
            if trend != "strong" and ret_20d[i] < v29_rs_ret20_threshold:
                new_position = 0
                counters["v29_rs_filter_blocked"] += 1

        if new_position == 1 and position == 0 and not entry_alpha_ok:
            new_position = 0; counters["alpha_blocked"] += 1

        # V31-A: Peak-chasing guard — reduce/skip when ret5d very hot + dist_sma20 large
        if v31_peak_chasing_guard and new_position == 1 and position == 0 and not vshape_entry:
            pcg_trigger = (
                ret_5d[i] > v31_pcg_ret5d_thresh and
                dist_sma20[i] * 100 > v31_pcg_dist_thresh
            )
            if pcg_trigger:
                if v31_pcg_action == "skip":
                    new_position = 0; counters["v31_pcg_skipped"] += 1
                else:  # half_size
                    position_size = position_size * 0.50 if position_size > 0 else 0.25
                    counters["v31_pcg_halved"] += 1

        # V33-C: Recovery-peak filter — block entry khi giá đã hồi phục nhanh từ đáy
        # Mục đích: tránh mua đỉnh sóng hồi (73-74% lệnh thua có giá thấp hơn 5d/10d trước)
        # Chỉ block khi: ret10d > threshold VÀ dist_sma20 > threshold VÀ trend != strong
        if (v33_recovery_peak_filter and new_position == 1 and position == 0
                and not vshape_entry and not breakout_entry):
            ret10_val = (close[i] / close[max(0, i - 10)] - 1) if i >= 10 and close[max(0, i - 10)] > 0 else 0
            dist_ok = not np.isnan(dist_sma20[i]) and dist_sma20[i] > v33_rpf_dist_sma20_thresh
            ret_ok = ret10_val > v33_rpf_ret10_thresh
            trend_ok = (not v33_rpf_require_weak) or (trend != "strong")
            if ret_ok and dist_ok and trend_ok:
                new_position = 0; counters["v33_rpf_blocked"] += 1

        # Rule ensemble
        if patch_rule_ensemble:
            ml_buy = (new_position == 1 and position == 0)
            rule_buy = (rule_signal[i] == 1)
            if rule_buy and not ml_buy and position == 0 and trend == "strong":
                new_position = 1
                position_size = 0.30
                counters["rule_only_entry"] += 1

        # V26-E: Stronger rule ensemble
        if v26_strong_rule_ensemble and not patch_rule_ensemble:
            ml_buy = (new_position == 1 and position == 0)
            if not ml_buy and position == 0 and rule_consecutive[i] >= 3:
                if trend in ("strong", "moderate"):
                    new_position = 1
                    position_size = 0.35
                    counters["v26_strong_rule_entry"] += 1
        elif v26_strong_rule_ensemble and patch_rule_ensemble:
            if position == 0 and new_position == 0 and rule_consecutive[i] >= 3:
                if trend == "moderate":
                    new_position = 1
                    position_size = 0.30
                    counters["v26_strong_rule_moderate"] += 1

        # V27: rule-priority entry
        if v27_rule_priority and position == 0 and new_position == 0:
            if sym in _RULE_PRIORITY_SYMBOLS and rule_consecutive[i] >= 2 and trend in ("strong", "moderate"):
                new_position = 1
                position_size = 0.35 if trend == "moderate" else 0.40
                counters["v27_rule_priority_entry"] += 1

        # Position sizing
        if new_position == 1 and position == 0:
            entry_score = sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2])
            atr_ratio = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03
            if vshape_entry: position_size = 0.50
            elif v35_hybrid_now: position_size = v35_hybrid_size
            elif trend == "strong" and entry_score >= 4: position_size = 0.95
            elif trend == "strong" and entry_score >= 3: position_size = 0.90
            elif trend == "moderate" and entry_score >= 3: position_size = 0.50
            elif trend == "weak": position_size = 0.30
            else: position_size = 0.50
            if atr_ratio > 0.055: position_size = min(position_size, 0.35)
            elif atr_ratio > 0.040: position_size = min(position_size, 0.50)
            if trend == "weak": position_size = min(position_size, 0.40)
            elif trend == "moderate": position_size = min(position_size, 0.70)
            if close[i] <= opn[i] and not vshape_entry: position_size *= 0.75
            if ret_5d[i] > ret5_hot: position_size = min(position_size, 0.40)
            position_size *= regime_cfg["size_mult"]

            # V26-G: Score-5 penalty
            if v26_score5_penalty and entry_score == 5:
                position_size *= 0.75
                counters["v26_score5_penalized"] += 1

            # V27: dynamic score-5 penalty
            if v27_dynamic_score5_penalty and entry_score == 5:
                score5_penalty = 1.0
                if sym in _SCORE5_RISKY_SYMBOLS:
                    score5_penalty *= 0.70
                if trend == "weak":
                    score5_penalty *= 0.85
                if bb < 0.40 and trend != "strong":
                    score5_penalty *= 0.90
                if rule_consecutive[i] >= 3 and trend in ("strong", "moderate"):
                    score5_penalty = max(score5_penalty, 0.80)
                if score5_penalty < 0.999:
                    position_size *= score5_penalty
                    counters["v27_score5_dynamic_penalized"] += 1

            # V31-D: Profile-aware position sizing
            if v31_profile_sizing:
                if profile == "momentum":
                    position_size *= v31_ps_momentum_mult
                elif profile == "high_beta":
                    position_size *= v31_ps_highbeta_mult
                elif profile == "defensive":
                    position_size *= v31_ps_defensive_mult
                elif profile == "bank":
                    position_size *= v31_ps_bank_mult
                counters["v31_profile_sized"] += 1

            # V30-A4: Rally-aware position scaling
            if v30_rally_position_scaling and not vshape_entry and not breakout_entry:
                ret10_ps = (close[i] / close[max(0,i-10)] - 1) if i >= 10 and close[max(0,i-10)] > 0 else 0
                if ret10_ps > v30_rps_tier2_rally:
                    new_position = 0; counters["v30_rps_blocked"] += 1
                elif ret10_ps > v30_rps_tier1_rally:
                    position_size *= 0.70; counters["v30_rps_scaled"] += 1

            position_size = max(0.25, min(position_size, 1.0))

            # V26-F: Minimum position threshold
            if v26_min_position and position_size < 0.28:
                new_position = 0; counters["v26_min_pos_blocked"] += 1

        # === EXIT LOGIC ===
        if position == 1:
            projected = equity[i - 1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            if close[i] > max_price_in_trade: max_price_in_trade = close[i]
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0
            price_max_profit = (max_price_in_trade / entry_close - 1) if entry_close > 0 else 0
            price_cur_ret = (close[i] / entry_close - 1) if entry_close > 0 else 0
            strong_uptrend = trend == "strong"
            atr_ratio_now = (atr14[i] / close[i]) if (close[i] > 0 and not np.isnan(atr14[i])) else 0.03

            if not np.isnan(atr14[i]) and close[i] > 0:
                atr_stop = ATR_MULT * atr14[i] / close[i]
                atr_stop = max(0.025, min(atr_stop, 0.06))
            else:
                atr_stop = 0.04

            # 0) HARD STOP
            if cum_ret <= -HARD_STOP:
                new_position = 0; exit_reason = "hard_stop"

            # Model B exit — primary exit signal for model_b/hybrid modes
            elif (use_model_b_exit and exit_mode in ("model_b", "hybrid") and y_pred_exit is not None
                    and hold_days >= MODEL_B_MIN_HOLD and y_pred_exit[i - 1] == 1):
                model_b_allow_exit = True
                if MODEL_B_REQUIRE_TREND_BREAK:
                    model_b_negative_signals = 0
                    if MODEL_B_USE_SMA20 and not np.isnan(sma20[i]) and close[i] < sma20[i]:
                        model_b_negative_signals += 1
                    if MODEL_B_USE_EMA8 and not np.isnan(ema8[i]) and close[i] < ema8[i]:
                        model_b_negative_signals += 1
                    if MODEL_B_USE_MACD and not np.isnan(macd_hist[i]) and macd_hist[i] <= 0:
                        model_b_negative_signals += 1
                    if MODEL_B_USE_RED_CANDLE and close[i] < opn[i]:
                        model_b_negative_signals += 1

                    model_b_allow_exit = model_b_negative_signals >= MODEL_B_MIN_NEG_SIGNALS
                    if (MODEL_B_PROFIT_ONLY_IF_LOSING_STRUCTURE
                            and price_cur_ret >= MODEL_B_PROFIT_THRESHOLD
                            and not model_b_allow_exit):
                        model_b_allow_exit = False

                if model_b_allow_exit:
                    new_position = 0; exit_reason = "model_b_exit"
                    counters["model_b_exit"] += 1

            elif exit_mode == "model_b":
                pass

            # V38b: Stall-exit — giu N ngay nhung max profit qua thap, dang lo nho -> chot
            elif v38b_stall_exit and hold_days >= v38b_stall_min_hold and \
                    price_max_profit < v38b_stall_max_profit and \
                    price_cur_ret <= v38b_stall_pnl_thresh:
                new_position = 0; exit_reason = "v38b_stall_exit"
                counters["v38b_stall_exit"] += 1

            # V38c: HA-driven exit — bearish reversal HOAC late_wave + body_shrinking, dang lo
            elif v38c_ha_exit and hold_days >= v38c_ha_min_hold and \
                    price_cur_ret < v38c_ha_pnl_thresh:
                ha_bear = ha_arrays["ha_bearish_reversal_signal"][i] >= 0.5
                ha_late = (ha_arrays["ha_late_wave"][i] >= 0.5 and
                           ha_arrays["ha_body_shrinking"][i] >= 0.5)
                if ha_bear or ha_late:
                    new_position = 0; exit_reason = "v38c_ha_exit"
                    counters["v38c_ha_exit"] += 1

            # V38d: Rule co-pilot exit — co profit nhung rule_signal het hoat dong (close<MA20 hoac MACD<=0)
            elif v38d_copilot_exit and hold_days >= v38d_copilot_min_hold and \
                    price_max_profit >= v38d_copilot_min_profit:
                rule_off = (
                    (np.isnan(macd_hist[i]) or macd_hist[i] <= 0) or
                    (not np.isnan(sma20[i]) and close[i] < sma20[i])
                )
                if rule_off and price_cur_ret < price_max_profit * 0.6:
                    # giu it nhat 60% peak; neu duoi thi co-pilot exit
                    new_position = 0; exit_reason = "v38d_copilot_exit"
                    counters["v38d_copilot_exit"] += 1

            # V28-K4: Early loss cut — must run BEFORE hard_cap to intercept crash entries
            # Cut loss at a tighter threshold within first N hold days
            elif v28_early_loss_cut and hold_days <= v28_early_loss_cut_days and not vshape_entry:
                if price_cur_ret <= v28_early_loss_cut_threshold:
                    new_position = 0; exit_reason = "v28_early_loss_cut"
                    counters["v28_early_loss_cut"] += 1

            # V32-A: HAP preempt — chạy TRƯỚC hard_cap: nếu đã từng có >= trigger% lãi rồi rớt về floor
            # Mục đích: bắt các trade "lên +5-15% rồi rớt -12%" mà V31-C miss do ordering bug
            # V33-D variant: chỉ trigger sau N ngày giảm liên tiếp (tránh bán đáy spike 1 ngày)
            # V39b variant: require hold_days >= v39b_hap_min_hold + optional tighter trigger
            elif v32_hap_preempt:
                hap_trigger_eff = v39b_hap_trigger if v39b_hap_trigger is not None else v32_hap_pre_trigger
                hap_price_ok = (price_max_profit >= hap_trigger_eff and
                                price_cur_ret <= v32_hap_pre_floor)
                if v39b_hap_min_hold > 0 and hold_days < v39b_hap_min_hold:
                    hap_price_ok = False  # HAP chưa active do hold quá ngắn
                if hap_price_ok:
                    if v33_hap_consec_drop:
                        # Chỉ trigger nếu đã giảm >= v33_hcd_min_days ngày liên tiếp
                        consec_drop_ok = (v33_consec_below_ema8 >= v33_hcd_min_days)
                        if consec_drop_ok:
                            new_position = 0; exit_reason = "v32_hap_preempt"
                            counters["v32_hap_preempt"] += 1
                        # else: giữ vị thế thêm, chờ xác nhận
                    else:
                        new_position = 0; exit_reason = "v32_hap_preempt"
                        counters["v32_hap_preempt"] += 1

            # V30-C1: ATR-aware hard cap — replaces fixed floor when enabled
            elif v30_atr_aware_hardcap:
                atr_cap = max(v30_atr_hc_floor, min(v30_atr_hc_ceiling, v30_atr_hc_mult * atr_ratio_now))
                if price_cur_ret <= -atr_cap:
                    new_position = 0; exit_reason = "signal_hard_cap"
                    counters["v30_atr_hc"] += 1

            # V30-C2: Two-step hard cap — partial halve then full exit
            elif v30_hardcap_two_step_v2:
                if price_cur_ret <= v30_hc2_step2_loss:
                    new_position = 0; exit_reason = "signal_hard_cap"
                    v30_hc2_halved = False
                    counters["v30_hc2_full"] += 1
                elif price_cur_ret <= v30_hc2_step1_loss and not v30_hc2_halved:
                    position_size = position_size * 0.50
                    v30_hc2_halved = True
                    counters["v30_hc2_half"] += 1

            # === HARD CAP / SIGNAL HARD CAP ===
            elif v22_mode:
                # V22: simple adaptive or flat hard cap
                if cfg["v22_adaptive_hard_cap"]:
                    if profile == "high_beta":
                        cap = max(cfg["v22_hard_cap_floor_hb"], cfg["v22_hard_cap_mult_hb"] * atr_ratio_now)
                    else:
                        cap = max(cfg["v22_hard_cap_floor"], cfg["v22_hard_cap_mult_std"] * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                elif price_cur_ret <= -V22_SIGNAL_HARD_CAP:
                    new_position = 0; exit_reason = "signal_hard_cap"
                    counters["signal_hard_cap"] += 1
            elif patch_smart_hardcap:
                if trend == "weak":
                    if price_cur_ret <= hard_cap_weak:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        hard_cap_pending_bars = 0
                        counters["signal_hard_cap"] += 1
                elif trend == "moderate":
                    if v26_wider_hardcap:
                        cap = max(hard_cap_moderate_floor * 1.3, hard_cap_moderate_mult * 1.3 * atr_ratio_now)
                    elif patch_adaptive_hardcap:
                        cap = max(0.10 if atr_ratio_now < 0.025 else hard_cap_moderate_floor,
                                  hard_cap_moderate_mult * atr_ratio_now)
                    else:
                        cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        hard_cap_pending_bars += 1
                        hc_confirm_bars = 1 + HARD_CAP_CONFIRM_MODERATE
                        if v27_hardcap_two_step and profile in ("high_beta", "momentum"):
                            hc_confirm_bars += 1
                        if hard_cap_pending_bars >= hc_confirm_bars:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            hard_cap_pending_bars = 0
                            counters["signal_hard_cap"] += 1
                    else:
                        hard_cap_pending_bars = 0
                else:  # strong
                    if v26_wider_hardcap:
                        cap = max(hard_cap_strong_floor * 1.4, hard_cap_strong_mult * 1.4 * atr_ratio_now)
                        if atr_ratio_now > 0.04:
                            cap = max(0.25, 4.0 * atr_ratio_now)
                        elif profile == "high_beta":
                            cap = max(0.22, 4.0 * atr_ratio_now)
                    elif patch_adaptive_hardcap:
                        if atr_ratio_now > 0.04:
                            cap = max(0.20, 3.5 * atr_ratio_now)
                        elif profile == "high_beta":
                            cap = max(0.18, 3.5 * atr_ratio_now)
                        else:
                            cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    else:
                        if profile == "high_beta":
                            cap = max(0.18, 3.5 * atr_ratio_now)
                        else:
                            cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        hard_cap_pending_bars += 1
                        hc_confirm_bars = 1 + HARD_CAP_CONFIRM_STRONG
                        if v27_hardcap_two_step and profile in ("high_beta", "momentum"):
                            hc_confirm_bars += 1
                        if hard_cap_pending_bars >= hc_confirm_bars:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            hard_cap_pending_bars = 0
                            counters["signal_hard_cap"] += 1
                    else:
                        hard_cap_pending_bars = 0
            else:
                # V23 original hard_cap (no confirm bars)
                if trend == "weak":
                    if price_cur_ret <= hard_cap_weak:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                elif trend == "moderate":
                    cap = max(hard_cap_moderate_floor, hard_cap_moderate_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1
                else:
                    if profile == "high_beta":
                        cap = max(hard_cap_strong_floor, hard_cap_strong_mult * atr_ratio_now)
                    else:
                        cap = max(hard_cap_moderate_floor, hard_cap_strong_mult * atr_ratio_now)
                    if price_cur_ret <= -cap:
                        new_position = 0; exit_reason = "signal_hard_cap"
                        counters["signal_hard_cap"] += 1

            if exit_mode != "model_b":
                # Fast exit loss
                if new_position == 1:
                    # V30-C3: regime-aware hard cap post-check (overrides standard cap if tighter needed)
                    if v30_regime_aware_hardcap:
                        choppy_now = regime_cfg["choppy_regime"]
                        rah_cap = v30_rah_choppy_cap if choppy_now else (
                            v30_rah_trending_cap if trend == "strong" else -0.09)
                        if price_cur_ret <= rah_cap:
                            new_position = 0; exit_reason = "signal_hard_cap"
                            counters["v30_rah_cap"] += 1
                    if v22_mode:
                        # V22: smart fast exit with trend_healthy check
                        do_fast_exit = False
                        ft = cfg["v22_fast_exit_threshold_hb"] if profile == "high_beta" else cfg["v22_fast_exit_threshold_std"]
                        mt = ft + 0.02

                        trend_healthy = (cfg["v22_fast_exit_skip_strong"] and strong_uptrend and
                                        macd_line[i] > 0 and
                                        not np.isnan(sma20[i]) and close[i] > sma20[i] * 0.97)

                        vol_selling = (cfg["v22_fast_exit_vol_confirm"] and
                                      not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i] and
                                      close[i] < opn[i])

                        if price_cur_ret < ft and hold_days > 3:
                            if trend_healthy and not vol_selling:
                                counters["fast_exit_saved"] += 1
                            else:
                                do_fast_exit = True
                        elif (price_cur_ret < mt and hold_days > 2 and
                              macd_hist[i] < 0 and not np.isnan(ema8[i]) and close[i] < ema8[i]):
                            if trend_healthy and not vol_selling:
                                counters["fast_exit_saved"] += 1
                            else:
                                do_fast_exit = True

                        if do_fast_exit:
                            new_position = 0; exit_reason = "fast_exit_loss"
                            counters["fast_exit_loss"] += 1
                    else:
                        # V23+: graduated fast_exit_loss by trend
                        hb_buf = fast_exit_hb_buffer if profile == "high_beta" else 0.0
                        if trend == "strong":
                            ft = fast_exit_strong - hb_buf
                        elif trend == "moderate":
                            ft = fast_exit_moderate - hb_buf
                        else:
                            ft = fast_exit_weak - hb_buf
                        mt = ft + 0.02

                        do_fast_exit = False
                        if price_cur_ret < ft and hold_days > 3:
                            do_fast_exit = True
                        elif (price_cur_ret < mt and hold_days > 2 and
                              macd_hist[i] < 0 and not np.isnan(ema8[i]) and close[i] < ema8[i]):
                            do_fast_exit = True

                        if do_fast_exit:
                            new_position = 0; exit_reason = "fast_exit_loss"
                            counters["fast_exit_loss"] += 1

                # ATR stop
                if new_position == 1 and cum_ret <= -atr_stop:
                    new_position = 0; exit_reason = "stop_loss"

                # V29-P1: Adaptive peak lock — once max_profit reaches trigger,
                #     ratchet a stop floor = keep * max_profit. Prevents "ran +20% then closed -10%".
                if new_position == 1 and v29_adaptive_peak_lock and price_max_profit >= v29_apl_trigger:
                    apply_lock = True
                    if v29_peak_lock_high_beta_only and profile not in ("high_beta", "momentum"):
                        apply_lock = False
                    if apply_lock:
                        floor_ret = price_max_profit * v29_apl_keep
                        if price_cur_ret <= floor_ret:
                            new_position = 0; exit_reason = "v29_peak_lock"
                            counters["v29_peak_lock"] += 1

                # V31-C: Hardcap-after-profit — once we've had >=5% profit, tighten exit floor to -3%
                # Prevents "ran +5-15% then closed -13%" which is the main hard_cap loss pattern
                if (new_position == 1 and v31_hardcap_after_profit and
                    price_max_profit >= v31_hap_profit_trigger and
                    price_cur_ret <= v31_hap_floor):
                    new_position = 0; exit_reason = "v31_hap_exit"
                    counters["v31_hap_exit"] += 1

                # V32-B: Trend-weak oversold exit — khi trend=weak + giá quá xa SMA20 + còn có lãi → exit
                # Logic: nếu trend đã yếu + giá dưới SMA20 quá 7% → thị trường đang sập, không nên giữ
                if (new_position == 1 and v32_weak_oversold_exit and
                    trend == "weak" and hold_days >= v32_woe_hold_min and
                    not np.isnan(dist_sma20[i]) and dist_sma20[i] < v32_woe_dist_thresh and
                    price_cur_ret >= v32_woe_min_profit):
                    new_position = 0; exit_reason = "v32_weak_oversold"
                    counters["v32_weak_oversold"] += 1

                # V32-C: Dynamic hard_cap tighten — khi dist_sma20 < -8% → exit ở -7% thay vì -12%
                # Ghi chú: chạy khi new_position vẫn = 1 (hard_cap chưa bắt vì trade chưa đến ngưỡng cũ)
                if (new_position == 1 and v32_dynamic_hc_dist and
                    not np.isnan(dist_sma20[i]) and dist_sma20[i] < v32_dhc_dist_thresh and
                    price_cur_ret <= v32_dhc_tight_cap):
                    new_position = 0; exit_reason = "v32_dynamic_hc"
                    counters["v32_dynamic_hc"] += 1

                # V32-D: Profit ratchet exit — khi đã đạt >= trigger% profit rồi rớt về keep * max_profit
                # Nhẹ hơn V29-P1 vì dùng price-based thay vì equity-based, và keep thấp hơn
                if (new_position == 1 and v32_profit_ratchet and
                    price_max_profit >= v32_pr_trigger):
                    pr_floor = price_max_profit * v32_pr_keep
                    if price_cur_ret <= pr_floor:
                        new_position = 0; exit_reason = "v32_profit_ratchet"
                        counters["v32_profit_ratchet"] += 1

                # V33-A: Multi-tier trailing ratchet — tighten floor progressively theo mức profit đạt được
                # Mục đích: fix -2177% drag từ signal_hard_cap (401 lệnh max_profit>5% nhưng kết thúc lỗ>5%)
                # Khác V32-D: multi-tier, sàn tăng dần theo profit lớn; tập trung vào big winners
                if new_position == 1 and v33_trailing_ratchet:
                    if price_max_profit >= v33_tr_tier3_trigger:
                        tr_floor = price_max_profit * v33_tr_tier3_keep
                    elif price_max_profit >= v33_tr_tier2_trigger:
                        tr_floor = price_max_profit * v33_tr_tier2_keep
                    elif price_max_profit >= v33_tr_tier1_trigger:
                        tr_floor = price_max_profit * v33_tr_tier1_keep
                    else:
                        tr_floor = None
                    if tr_floor is not None and price_cur_ret <= tr_floor:
                        new_position = 0; exit_reason = "v33_trailing_ratchet"
                        counters["v33_trailing_ratchet"] += 1

                # V33-B: Trend-reversal exit — bán ngay khi max_profit đạt ngưỡng + trend đảo chiều
                # Mục đích: bắt "bán ở đỉnh phân phối" sớm hơn signal/hard_cap
                # Logic: price_max_profit >= min_profit VÀ close < ema8 N ngày VÀ rsi < thresh
                if new_position == 1 and v33_trend_rev_exit:
                    if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                        v33_consec_below_ema8 += 1
                    else:
                        v33_consec_below_ema8 = 0
                    if (price_max_profit >= v33_tre_min_profit and
                        hold_days >= v33_tre_hold_min and
                        v33_consec_below_ema8 >= 2 and
                        not np.isnan(rsi14[i]) and rsi14[i] < v33_tre_rsi_thresh):
                        new_position = 0; exit_reason = "v33_trend_rev_exit"
                        counters["v33_trend_rev_exit"] += 1
                else:
                    # track ema8 below streak dù v33_trend_rev_exit off (cần cho v33_hap_consec_drop)
                    if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                        v33_consec_below_ema8 += 1
                    else:
                        v33_consec_below_ema8 = 0

                # V29-P8: Profit safety net — once max_profit >= trigger, never let it close negative
                if (new_position == 1 and v29_profit_safety_net and
                    price_max_profit >= v29_profit_safety_trigger and price_cur_ret < 0):
                    new_position = 0; exit_reason = "v29_profit_safety"
                    counters["v29_profit_safety"] += 1

                # V29-P9: Hardcap-after-peak — tighten hard cap once we've had a meaningful peak
                if (new_position == 1 and v29_hardcap_after_peak and
                    price_max_profit >= v29_hardcap_after_peak_trigger and
                    price_cur_ret <= v29_hardcap_after_peak_floor):
                    new_position = 0; exit_reason = "v29_hardcap_after_peak"
                    counters["v29_hardcap_after_peak"] += 1

                # V29-P4: Reversal after peak — exit fast when ret_2d collapses after big run-up
                if (new_position == 1 and v29_reversal_after_peak and
                    price_max_profit >= v29_reversal_peak_trigger and
                    ret_2d[i] <= v29_reversal_ret2_threshold):
                    new_position = 0; exit_reason = "v29_reversal_after_peak"
                    counters["v29_reversal_after_peak"] += 1

                # V29-P2: ATR velocity exit — exit on a fast 2-day decline > k*ATR after meaningful profit
                if (new_position == 1 and v29_atr_velocity_exit and
                    price_max_profit >= v29_atr_velocity_min_profit and
                    not np.isnan(atr14[i]) and close[i] > 0 and i >= 2):
                    two_d_drop = close[i - 2] - close[i]
                    atr_threshold = v29_atr_velocity_k * atr14[i]
                    if two_d_drop >= atr_threshold:
                        new_position = 0; exit_reason = "v29_atr_velocity"
                        counters["v29_atr_velocity"] += 1

                # V30-B3: Chandelier trailing — ATR×k trailing from max_high when in profit
                if (new_position == 1 and v30_chandelier_trail and
                    price_max_profit >= v30_chand_profit_trigger and
                    not np.isnan(atr14[i]) and close[i] > 0):
                    chand_stop = max_price_in_trade - v30_chand_atr_mult * atr14[i]
                    if close[i] < chand_stop:
                        new_position = 0; exit_reason = "v30_chandelier"
                        counters["v30_chandelier"] += 1

                # Peak protection
                if new_position == 1 and mod_b:
                    if v22_mode:
                        # V22: simple threshold + 3-in-1
                        if price_max_profit >= 0.20:
                            price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                            bearish_candle = close[i] < opn[i]
                            if price_below_sma10 and heavy_vol and bearish_candle:
                                new_position = 0; exit_reason = "peak_protect_dist"
                                counters["peak_protect"] += 1
                    elif patch_pp_restore:
                        if price_max_profit >= 0.25:
                            pp_threshold_active = 0.10
                        elif strong_uptrend:
                            pp_threshold_active = peak_protect_strong_threshold
                        else:
                            pp_threshold_active = peak_protect_normal_threshold

                        pp_bonus = regime_cfg.get("pp_sensitivity_bonus", 0)
                        pp_threshold_active = max(0.08, pp_threshold_active - pp_bonus)

                        if price_max_profit >= pp_threshold_active:
                            price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                            if price_below_sma10:
                                pp_pending_bars += 1
                                heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.3 * avg_vol20[i])
                                if pp_pending_bars >= 2 or heavy_vol:
                                    new_position = 0; exit_reason = "peak_protect_dist"
                                    pp_pending_bars = 0
                                    counters["peak_protect"] += 1
                            else:
                                pp_pending_bars = 0
                    elif patch_pp_2of3:
                        pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
                        if price_max_profit >= pp_threshold:
                            price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                            bearish_candle = close[i] < opn[i]
                            checks = sum([price_below_sma10, heavy_vol, bearish_candle])
                            if checks >= 2:
                                new_position = 0; exit_reason = "peak_protect_dist"
                                counters["peak_protect"] += 1
                    else:
                        pp_threshold = peak_protect_strong_threshold if strong_uptrend else peak_protect_normal_threshold
                        if price_max_profit >= pp_threshold:
                            price_below_sma10 = (not np.isnan(sma10[i]) and close[i] < sma10[i])
                            heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.5 * avg_vol20[i])
                            bearish_candle = close[i] < opn[i]
                            if price_below_sma10 and heavy_vol and bearish_candle:
                                new_position = 0; exit_reason = "peak_protect_dist"
                                counters["peak_protect"] += 1

                # EMA8 peak protect
                if mod_b and new_position == 1 and position == 1:
                    if price_max_profit >= 0.15:
                        if not np.isnan(ema8[i]) and close[i] < ema8[i]:
                            consecutive_below_ema8 += 1
                        else:
                            consecutive_below_ema8 = 0
                        if consecutive_below_ema8 >= 2 and price_cur_ret < price_max_profit * 0.75:
                            new_position = 0; exit_reason = "peak_protect_ema"
                            counters["peak_protect"] += 1

                # V28-K5: Cycle peak exit — if ret_3d turns negative after being up >8%, exit quickly
                if v28_cycle_peak_exit and new_position == 1 and position == 1:
                    if price_max_profit >= 0.08 and ret_3d[i] < -0.02 and price_cur_ret < price_max_profit * 0.70:
                        heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.2 * avg_vol20[i])
                        if heavy_vol or ret_3d[i] < -0.04:
                            new_position = 0; exit_reason = "v28_cycle_peak"
                            counters["v28_cycle_peak"] += 1

                # Hybrid exit
                if exit_mode != "model_b" and new_position == 1 and strong_uptrend and cum_ret > 0.05 and max_profit > 0.08:
                    macd_bearish = macd_hist[i] < 0 and macd_hist[i - 1] >= 0 if i > 0 else False
                    price_below_ma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                    if macd_bearish and price_below_ma20:
                        new_position = 0; exit_reason = "hybrid_exit"
                    elif price_below_ma20 and cum_ret < max_profit * 0.5:
                        new_position = 0; exit_reason = "hybrid_exit"

                # Adaptive trailing
                elif new_position == 1 and max_profit > 0.03:
                    if max_profit > 0.25: trail_pct = 0.18
                    elif max_profit > 0.15: trail_pct = 0.25
                    elif max_profit > 0.08: trail_pct = 0.40
                    else: trail_pct = 0.65
                    # V29-P3: tighten when max_profit is very high (lock more profit)
                    if v29_tighter_trail_high_profit and max_profit >= v29_high_profit_trigger:
                        trail_pct = min(trail_pct, v29_high_profit_trail)
                    if strong_uptrend: trail_pct *= STRONG_TREND_TRAIL_MULT
                    elif trend == "moderate": trail_pct *= 0.7
                    giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                    if giveback >= trail_pct:
                        new_position = 0; exit_reason = "trailing_stop"

                # Profit lock
                if new_position == 1 and max_profit >= PROFIT_LOCK_THRESHOLD:
                    if cum_ret < PROFIT_LOCK_MIN and not strong_uptrend:
                        if patch_symbol_tuning and regime_cfg.get("disable_profit_lock_in_strong") and trend == "strong":
                            pass
                        else:
                            new_position = 0; exit_reason = "profit_lock"

                # Zombie
                if new_position == 1 and hold_days >= ZOMBIE_BARS and cum_ret < 0.01:
                    if not strong_uptrend:
                        new_position = 0; exit_reason = "zombie_exit"

            # Extended hold / trend persistence
            extended_min_hold = MIN_HOLD
            if v26_extended_hold and strong_uptrend and cum_ret > 0.05:
                extended_min_hold = 12
            if v27_trend_persistence_hold and strong_uptrend and cum_ret > 0.03:
                trend_supported = (
                    (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.995) and
                    (macd_hist[i] > -0.01)
                )
                if trend_supported:
                    extended_min_hold = max(extended_min_hold, 10)
                    counters["v27_persistence_hold"] += 1

            # Min hold
            if new_position == 0 and exit_reason not in (
                    "stop_loss", "hard_stop", "hybrid_exit", "peak_protect_dist",
                    "peak_protect_ema", "fast_loss_cut", "signal_hard_cap",
                    "fast_exit_loss", "v31_hap_exit", "model_b_exit") and hold_days < extended_min_hold:
                if cum_ret > -atr_stop: new_position = 1

            # V39a: signal exit min hold — block signal exit trước khi đủ hold_days
            # Root cause: early_wave fw=8d target tạo bias thoát sớm ở 21-30d (WR=12%)
            # Nếu rule_confirm bật: thêm điều kiện MACD<0 AND Close<MA20 mới cho exit
            if (new_position == 0 and exit_reason == "signal" and
                    v39a_signal_exit_min_hold > 0 and hold_days < v39a_signal_exit_min_hold):
                new_position = 1; counters["v39a_min_hold_blocked"] += 1
            if (new_position == 0 and exit_reason == "signal" and v39a_rule_confirm_exit):
                macd_bearish_now = (not np.isnan(macd_hist[i]) and macd_hist[i] < 0)
                below_ma20_now = (not np.isnan(sma20[i]) and close[i] < sma20[i])
                # V39g: chỉ defer khi max_profit đã đạt threshold (trades stall thì không defer)
                min_mp_ok = (price_max_profit >= v39g_rule_confirm_min_maxprofit)
                if not (macd_bearish_now and below_ma20_now) and min_mp_ok:
                    new_position = 1; counters["v39a_rule_confirm_blocked"] += 1

            # V31-E: Short-hold exit filter — block signal exits if hold too short and pnl not bad enough
            # Prevents the flood of hold<10d signal exits at -4% that drag WR down
            if (new_position == 0 and exit_reason == "signal" and v31_short_hold_exit_filter and
                hold_days < v31_shef_min_hold and cum_ret > v31_shef_min_pnl):
                new_position = 1; counters["v31_shef_blocked"] += 1

            # V32-E: Signal-weak-oversold passthrough — KHÔNG block signal exit khi trend=weak + dist<thresh
            # Bổ trợ V31-E: dù hold chưa đủ, nếu market oversold nặng thì vẫn cho exit
            if (new_position == 1 and exit_reason == "signal" and v32_signal_weak_exit and
                trend == "weak" and
                not np.isnan(dist_sma20[i]) and dist_sma20[i] < v32_swe_dist_thresh):
                new_position = 0; counters["v32_swe_pass"] += 1

            # V33-E: RSI oversold block — không thoát bởi signal/hap_preempt khi rsi oversold sâu
            # Mục đích: tránh bán đúng đáy điều chỉnh (85 lệnh giá tăng >8% trong 5 ngày sau bán)
            if (new_position == 0 and v33_rsi_oversold_block and
                exit_reason in ("signal", "v32_hap_preempt", "v32_weak_oversold") and
                not np.isnan(rsi14[i]) and rsi14[i] < v33_rob_rsi_thresh and
                price_cur_ret > -0.08):  # chỉ block khi chưa lỗ quá nặng
                new_position = 1; counters["v33_rob_blocked"] += 1

            # V33-F: Signal confirm exit — require 2 consecutive signal bars to exit
            # Mục đích: lọc false signal exit (776 exits WR 57% nhưng avg +7.9% — khá nhiều false exits)
            # Safety: chỉ block khi hold_days < 90 để tránh zombie trades quá dài
            if (new_position == 0 and exit_reason == "signal" and v33_signal_confirm_exit and
                cum_ret > v33_sce_min_pnl and price_max_profit >= v33_sce_min_profit_seen and
                hold_days < v33_sce_max_hold):
                if not v33_prev_signal_exit:
                    new_position = 1  # chờ bar tiếp theo confirm
                    counters["v33_sce_deferred"] += 1
                # else: bar trước cũng signal → cho exit

            # V30-B2: Momentum-hold override — run before confirmed exit scoring
            if (new_position == 0 and exit_reason == "signal" and
                v30_momentum_hold_override and cum_ret >= v30_mho_min_profit and
                not np.isnan(rsi14[i]) and rsi14[i] < v30_mho_rsi_max and
                not np.isnan(sma20[i]) and close[i] > sma20[i] and
                not np.isnan(ema8[i]) and close[i] > ema8[i] and
                macd_hist[i] > 0):
                new_position = 1; counters["v30_mho_saved"] += 1

            # V30-B1: Signal-exit defer — hold extra bars when still in uptrend and profitable
            if new_position == 0 and exit_reason == "signal" and v30_signal_exit_defer:
                if cum_ret >= v30_sed_min_cum_ret and trend in ("strong", "moderate"):
                    if v30_defer_bars_remaining > 0:
                        v30_defer_bars_remaining -= 1
                        new_position = 1; counters["v30_sed_defer"] += 1
                    else:
                        v30_defer_bars_remaining = v30_sed_defer_bars  # arm for next time
                else:
                    v30_defer_bars_remaining = 0  # reset when not qualifying

            # V31-B: Adaptive defer — defer signal exit while momentum confirmed (up to max_bars)
            # Smarter than V30's flat-N-bars: resets counter when EMA confirm fails
            if new_position == 0 and exit_reason == "signal" and v31_adaptive_defer:
                ema_ok = (not np.isnan(ema8[i]) and close[i] >= ema8[i]) if v31_ad_use_ema_confirm else True
                if cum_ret >= v31_ad_min_cum_ret and trend in ("strong", "moderate") and ema_ok:
                    if v31_defer_bars_remaining < v31_ad_max_bars:
                        v31_defer_bars_remaining += 1
                        new_position = 1; counters["v31_ad_defer"] += 1
                    else:
                        v31_defer_bars_remaining = 0  # max bars reached — let exit through
                else:
                    v31_defer_bars_remaining = 0

            # V39d: per-symbol rule-exit hybrid — với các mã stable-trend đã thua rule nhiều
            # Thay thế signal exit bằng rule-based exit: thoát khi Close<MA20 OR MACD_hist<=0
            # Mục đích: 21 mã V37a thua Rule do signal exit sớm, rule bắt trend 30-60d tốt hơn
            if (new_position == 0 and exit_reason == "signal" and v39d_rule_exit_symbols and
                    sym in v39d_rule_exit_symbols):
                rule_exit_now = (
                    (not np.isnan(macd_hist[i]) and macd_hist[i] <= 0) or
                    (not np.isnan(sma20[i]) and close[i] < sma20[i])
                )
                if not rule_exit_now:
                    new_position = 1; counters["v39d_rule_exit_held"] += 1
                # else: rule đồng thuận → cho signal exit thông qua

            # Confirmed signal exit
            if new_position == 0 and exit_reason == "signal" and mod_h:
                below_ma20 = (not np.isnan(sma20[i]) and close[i] < sma20[i])
                below_ma50 = (not np.isnan(sma50[i]) and close[i] < sma50[i])
                heavy_vol = (not np.isnan(avg_vol20[i]) and volume[i] > 1.4 * avg_vol20[i])
                bearish_candle = close[i] < opn[i]
                macd_falling = macd_hist[i] < macd_hist[i - 1] if i > 0 else False
                below_ema8 = (not np.isnan(ema8[i]) and close[i] < ema8[i] * 0.997)
                weak_rebound = (ret_5d[i] < 0.01 and rs <= 0)
                bearish_score = 0.0
                bearish_score += 2.0 if below_ma50 else 0.0
                bearish_score += 1.0 if below_ma20 else 0.0
                bearish_score += 1.0 if macd_hist[i] < -0.03 else 0.0
                bearish_score += 0.8 if (macd_hist[i] < 0 and macd_falling) else 0.0
                bearish_score += 0.8 if (bearish_candle and heavy_vol) else 0.0
                bearish_score += 0.7 if below_ema8 else 0.0
                bearish_score += 0.5 if weak_rebound else 0.0
                score_threshold = regime_cfg["exit_score_threshold"]
                if cum_ret > 0.06 and max_profit > 0.10 and trend == "strong": score_threshold += 0.7
                if hold_days < 7 and cum_ret > -0.02: score_threshold += 0.4
                if cum_ret < -0.03: score_threshold -= 0.4
                if not v22_mode:
                    if hold_days > time_decay_bars and cum_ret < 0.02:
                        score_threshold *= time_decay_mult
                        counters["time_decay_exit"] += 1
                    elif hold_days > 15 and cum_ret < 0.03:
                        score_threshold *= 0.60
                    elif hold_days > 10 and cum_ret < 0.01:
                        score_threshold *= 0.75
                else:
                    if hold_days > 15 and cum_ret < 0.03:
                        score_threshold *= 0.60
                    elif hold_days > 10 and cum_ret < 0.01:
                        score_threshold *= 0.75
                if bearish_score < score_threshold:
                    new_position = 1; counters["confirmed_exit_blocked"] += 1

            if new_position == 0 and exit_reason == "signal":
                if cum_ret < 0: confirm_bars = 0
                else:
                    confirm_bars = regime_cfg["base_confirm_bars"]
                    if cum_ret < -0.03: confirm_bars = max(1, confirm_bars - 1)
                if raw_signal == 0: consecutive_exit_signals += 1
                else: consecutive_exit_signals = 0
                if consecutive_exit_signals < confirm_bars: new_position = 1
                else: consecutive_exit_signals = 0

            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0.03 and trend == "strong": new_position = 1

            if mod_i and new_position == 0 and exit_reason == "signal":
                still_supported = (not np.isnan(sma20[i]) and close[i] >= sma20[i] * 0.99)
                trend_ok = trend in ("strong", "moderate")
                if (cum_ret > 0.03 and max_profit > 0.06 and trend_ok and still_supported and
                    macd_hist[i] > -0.02):
                    new_position = 1; counters["trend_carry_saved"] += 1

            # Long-horizon carry
            if patch_long_horizon:
                if (new_position == 0 and exit_reason in ("signal", "peak_protect_dist", "peak_protect_ema",
                                                           "profit_lock", "trailing_stop")):
                    long_horizon_regime = (
                        ret_60d[i] > 0.30 and
                        not np.isnan(sma20[i]) and not np.isnan(sma50[i]) and not np.isnan(sma100[i]) and
                        sma20[i] > sma50[i] > sma100[i] and
                        days_above_sma50[i] >= 20 and
                        cum_ret > 0.15
                    )
                    if long_horizon_regime:
                        hard_breakdown = (
                            (close[i] < sma50[i] * 0.97) or
                            (i >= 3 and macd_hist[i] < 0 and macd_hist[i-1] < 0 and macd_hist[i-2] < 0)
                        )
                        if not hard_breakdown:
                            new_position = 1
                            counters["long_horizon_carry"] += 1

        else:
            consecutive_exit_signals = 0
            consecutive_below_ema8 = 0
            v33_consec_below_ema8 = 0  # reset when not in position

        # Track prev signal exit for V33-F confirm
        v33_prev_signal_exit = (new_position == 0 and exit_reason == "signal") if position == 1 else False

        # EXECUTE
        cost = 0
        if new_position != position:
            if new_position == 1:
                deploy = equity[i - 1] * position_size
                cost = deploy * commission
                entry_equity = deploy - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i; hold_days = 0
                consecutive_exit_signals = 0; consecutive_below_ema8 = 0
                hard_cap_pending_bars = 0; pp_pending_bars = 0
                entry_close = close[i]; max_price_in_trade = close[i]
                v30_hc2_halved = False
                v30_defer_bars_remaining = 0
                v31_defer_bars_remaining = 0
                v33_consec_below_ema8 = 0
                v33_prev_signal_exit = False
                entry_features = {
                    "entry_wp": wp, "entry_dp": dp, "entry_rs": rs,
                    "entry_vs": vs, "entry_bs": bs, "entry_hl": hl,
                    "entry_od": od, "entry_bb": bb,
                    "entry_score": sum([wp < 0.75, dp > 0.02, rs > 0, vs > 1.1, hl >= 2]),
                    "entry_date": str(dates[i])[:10], "entry_symbol": str(symbols[i]),
                    "position_size": position_size, "entry_trend": trend,
                    "quick_reentry": quick_reentry, "breakout_entry": breakout_entry,
                    "vshape_entry": vshape_entry,
                    "entry_ret_5d": round(ret_5d[i] * 100, 2),
                    "entry_drop20d": round(drop_from_peak_20[i] * 100, 2),
                    "entry_dist_sma20": round(dist_sma20[i] * 100, 2),
                    "entry_profile": regime_cfg["profile"],
                    "entry_choppy_regime": regime_cfg["choppy_regime"],
                }
            else:
                cost = equity[i - 1] * position_size * (commission + tax)
                pnl_pct_now = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                if v35_relax_cooldown:
                    cooldown_remaining = v35_cooldown_after_big_loss if pnl_pct_now < -5 else v35_cooldown_after_loss
                else:
                    cooldown_remaining = COOLDOWN_AFTER_BIG_LOSS if pnl_pct_now < -5 else 3
                last_exit_price = close[i]; last_exit_reason = exit_reason; last_exit_bar = i
                if record_trades and entry_equity > 0:
                    pnl_pct = (close[i] / entry_close - 1) * 100 if entry_close > 0 else 0
                    max_pnl_pct = (max_equity_in_trade - entry_equity) / entry_equity * 100
                    trade_rec = {
                        "entry_day": current_entry_day, "exit_day": i,
                        "holding_days": i - current_entry_day,
                        "pnl_pct": round(pnl_pct, 2), "max_profit_pct": round(max_pnl_pct, 2),
                        "exit_reason": exit_reason, "exit_date": str(dates[i])[:10],
                        **entry_features,
                    }
                    if v31_enriched_log:
                        trade_rec["exit_trend"] = trend
                        trade_rec["exit_dist_sma20"] = round(dist_sma20[i] * 100, 2)
                        trade_rec["exit_ret_5d"] = round(ret_5d[i] * 100, 2)
                        trade_rec["exit_rsi14"] = round(float(rsi14[i]), 2) if not np.isnan(rsi14[i]) else 50.0
                        trade_rec["exit_macd_hist"] = round(float(macd_hist[i]), 5) if not np.isnan(macd_hist[i]) else 0.0
                        trade_rec["price_max_profit_pct"] = round(price_max_profit * 100, 2)
                        trade_rec["exit_above_sma20"] = int(not np.isnan(sma20[i]) and close[i] >= sma20[i])
                        trade_rec["exit_above_ema8"] = int(not np.isnan(ema8[i]) and close[i] >= ema8[i])
                        trade_rec["exit_vol_ratio"] = round(float(volume[i] / avg_vol20[i]), 2) if (not np.isnan(avg_vol20[i]) and avg_vol20[i] > 0) else 1.0
                    trades.append(trade_rec)
                entry_equity = 0; max_equity_in_trade = 0; max_price_in_trade = 0; position_size = 1.0
                hard_cap_pending_bars = 0; pp_pending_bars = 0

        if position == 1:
            equity[i] = equity[i - 1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i - 1] - cost
        position = new_position

    if position == 1 and entry_equity > 0 and record_trades:
        pnl_pct = (close[-1] / entry_close - 1) * 100 if entry_close > 0 else 0
        trades.append({
            "entry_day": current_entry_day, "exit_day": n - 1,
            "holding_days": n - 1 - current_entry_day,
            "pnl_pct": round(pnl_pct, 2), "exit_reason": "end",
            "exit_date": str(dates[-1])[:10], **entry_features,
        })

    return {
        "equity_curve": equity, "trades": trades,
        "total_return_pct": round((equity[-1] / initial_capital - 1) * 100, 2),
        "final_equity": round(equity[-1]),
        **{f"n_{k}": v for k, v in counters.items()},
    }
