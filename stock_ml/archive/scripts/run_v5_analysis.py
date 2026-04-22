"""
V5 EQUITY CURVE FILTER — Reduce MaxDD from -60% to target -30%
===============================================================
Builds on V4 (regime filter, dynamic trailing, ATR stops, position sizing).

V4 PROBLEM:
  MaxDD still -59 to -70% because during prolonged bear (2022), even filtered trades
  compound losses. No mechanism to STOP trading when the strategy itself is losing.

V5 ADDITIONS:
  A. EQUITY CURVE MA FILTER: Track 20-day EMA of equity curve.
     - When equity < EMA → "cold mode" (reduce size to 40% or skip)
     - When equity > EMA → "normal mode"
  B. DRAWDOWN CIRCUIT BREAKER: If portfolio DD > -15% from peak → STOP trading entirely
     - Resume only when equity recovers above 20-day EMA of equity
  C. PROGRESSIVE RECOVERY: After circuit breaker, first 5 trades at 50% size
  D. DAILY LOSS LIMIT: If 3 consecutive losses, pause for 5 days (bars)
"""
import argparse, sys, os, numpy as np, pandas as pd
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model, get_available_models
from src.evaluation.metrics import compute_metrics

# Import V4 backtest for comparison
from run_v4_analysis import backtest_v4, print_window_result, print_aggregate


def backtest_v5(y_pred, returns, df_test, feature_cols, initial_capital=100_000_000,
                commission=0.0015, tax=0.001, mode="v5",
                eq_ema_span=30, dd_circuit_breaker=-0.30, recovery_trades=3,
                cold_size=0.5, consec_loss_pause=4, pause_bars=3):
    """
    V5 backtest: V4 + SOFT equity curve filter + drawdown size reduction.
    
    KEY DESIGN: Never STOP trading entirely. Instead:
    - Cold mode (equity < EMA): reduce size to cold_size (50%)
    - Deep DD mode (DD > threshold): reduce size to 30%
    - Consecutive losses: brief pause (3 bars)
    
    Parameters:
    -----------
    eq_ema_span : int — EMA span for equity curve smoothing (30 = slower, less whipsaw)
    dd_circuit_breaker : float — DD threshold for max size reduction (e.g., -0.30 = -30%)
    recovery_trades : int — # trades at reduced size after deep DD recovery
    cold_size : float — position size when equity < EMA (0.5 = 50%)
    consec_loss_pause : int — consecutive losses before pausing
    pause_bars : int — bars to pause after consecutive losses
    """
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0
    trades = []
    current_entry_day = 0
    entry_equity = 0
    max_equity_in_trade = 0
    total_commission = 0
    hold_days = 0
    position_size = 1.0

    # ── V5 STATE ──
    equity_peak = initial_capital
    equity_ema = initial_capital  # EMA of equity curve
    ema_alpha = 2.0 / (eq_ema_span + 1)
    circuit_breaker_active = False
    recovery_count = 0  # trades since circuit breaker lifted
    consecutive_losses = 0
    pause_until = 0  # bar index to pause until
    cold_mode = False

    stats = {
        "n_filtered": 0, "n_regime_filtered": 0, "n_sl": 0,
        "n_trail": 0, "n_trend_hold": 0, "n_min_hold_skip": 0,
        "n_partial_size": 0,
        # V5 stats
        "n_eq_filter": 0, "n_circuit_breaker": 0, "n_consec_pause": 0,
        "n_recovery_half": 0, "circuit_breaker_days": 0,
    }

    # Extract features (same as V4)
    has_context = len(df_test) == n
    feat_arrays = {}
    if has_context:
        feat_names = ["rsi_slope_5d", "vol_surge_ratio", "range_position_20d",
                       "dist_to_resistance", "breakout_setup_score", "bb_width_percentile",
                       "higher_lows_count", "obv_price_divergence"]
        defaults = {"rsi_slope_5d": 0, "vol_surge_ratio": 1.0, "range_position_20d": 0.5,
                     "dist_to_resistance": 0.05, "breakout_setup_score": 0, "bb_width_percentile": 0.5,
                     "higher_lows_count": 0, "obv_price_divergence": 0}
        for fn in feat_names:
            if fn in df_test.columns:
                arr = df_test[fn].values.copy()
                arr = np.where(np.isnan(arr), defaults[fn], arr)
                feat_arrays[fn] = arr
            else:
                feat_arrays[fn] = np.full(n, defaults[fn])

    # SMAs and ATR for regime/stops (same as V4)
    close = sma20 = sma50 = atr14 = None
    if "close" in df_test.columns:
        close = df_test["close"].values
        sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
        sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values
        if "high" in df_test.columns and "low" in df_test.columns:
            high = df_test["high"].values
            low = df_test["low"].values
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
            tr[0] = high[0] - low[0]
            atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values
        else:
            atr14 = np.full(n, close.mean() * 0.02)

    def get_feat(name, idx):
        return feat_arrays[name][idx] if name in feat_arrays and idx < n else 0

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        new_position = 1 if pred == 1 else 0
        exit_reason = "signal"

        # ══════════════════════════════════════════════════════════════
        # V5: UPDATE EQUITY CURVE EMA (on previous bar's equity)
        # ══════════════════════════════════════════════════════════════
        equity_ema = ema_alpha * equity[i-1] + (1 - ema_alpha) * equity_ema
        equity_peak = max(equity_peak, equity[i-1])
        current_dd = (equity[i-1] - equity_peak) / equity_peak
        cold_mode = equity[i-1] < equity_ema

        # ══════════════════════════════════════════════════════════════
        # V5: DEEP DD MODE — Reduce size, don't stop trading
        # ══════════════════════════════════════════════════════════════
        deep_dd_mode = current_dd <= dd_circuit_breaker
        if deep_dd_mode:
            stats["circuit_breaker_days"] += 1
            if not circuit_breaker_active:
                circuit_breaker_active = True
                stats["n_circuit_breaker"] += 1
        else:
            if circuit_breaker_active and equity[i-1] >= equity_ema:
                circuit_breaker_active = False
                recovery_count = 0  # start recovery phase

        # ══════════════════════════════════════════════════════════════
        # V5: CONSECUTIVE LOSS PAUSE
        # ══════════════════════════════════════════════════════════════
        if not circuit_breaker_active and i < pause_until:
            if new_position == 1 and position == 0:
                new_position = 0
                stats["n_consec_pause"] += 1

        # ══════════════════════════════════════════════════════════════
        # V4 REGIME FILTER (inherited)
        # ══════════════════════════════════════════════════════════════
        if new_position == 1 and position == 0:
            if close is not None and sma50 is not None:
                price_below_sma50 = close[i] < sma50[i] if not np.isnan(sma50[i]) else False
                price_below_sma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                rs = get_feat("rsi_slope_5d", i)
                bs = get_feat("breakout_setup_score", i)
                if price_below_sma50 and price_below_sma20 and rs <= 0:
                    if bs < 3:
                        new_position = 0
                        stats["n_regime_filtered"] += 1

        # ══════════════════════════════════════════════════════════════
        # V4 ENTRY QUALITY FILTER (inherited)
        # ══════════════════════════════════════════════════════════════
        if new_position == 1 and position == 0:
            wp = get_feat("range_position_20d", i)
            dp = get_feat("dist_to_resistance", i)
            rs = get_feat("rsi_slope_5d", i)
            vs = get_feat("vol_surge_ratio", i)
            bs = get_feat("breakout_setup_score", i)
            hl = get_feat("higher_lows_count", i)
            bb = get_feat("bb_width_percentile", i)

            entry_score = 0
            if wp < 0.75: entry_score += 1
            if dp > 0.02: entry_score += 1
            if rs > 0: entry_score += 1
            if vs > 1.1: entry_score += 1
            if hl >= 2: entry_score += 1

            min_score = 3
            if entry_score < min_score:
                new_position = 0
                stats["n_filtered"] += 1

            if wp > 0.9 and rs <= 0 and bs < 2:
                new_position = 0
                stats["n_filtered"] += 1

            if bb > 0.85 and bs < 2 and entry_score < 4:
                new_position = 0
                stats["n_filtered"] += 1

        # ══════════════════════════════════════════════════════════════
        # V5: EQUITY CURVE POSITION SIZING
        # ══════════════════════════════════════════════════════════════
        if new_position == 1 and position == 0:
            bb = get_feat("bb_width_percentile", i)

            if deep_dd_mode:
                # Deep DD: trade at 30% size
                position_size = 0.3
                stats["n_circuit_breaker"] += 1
            elif recovery_count < recovery_trades:
                # Recovery phase after deep DD: 50% size
                position_size = 0.5
                recovery_count += 1
                stats["n_recovery_half"] += 1
            elif cold_mode:
                # Equity below EMA: reduce to cold_size
                position_size = cold_size
                stats["n_eq_filter"] += 1
            elif bb > 0.7:
                # V4 vol sizing
                position_size = 0.7
                stats["n_partial_size"] += 1
            else:
                position_size = 1.0

        # ══════════════════════════════════════════════════════════════
        # MIN HOLD FILTER (inherited from V4)
        # ══════════════════════════════════════════════════════════════
        if position == 1 and new_position == 0 and hold_days < 2 and exit_reason == "signal":
            cum_ret = (equity[i-1] * (1 + ret) - entry_equity) / entry_equity if entry_equity > 0 else 0
            if cum_ret > 0.01:
                new_position = 1
                stats["n_min_hold_skip"] += 1

        # ══════════════════════════════════════════════════════════════
        # ADAPTIVE EXIT (V4 logic)
        # ══════════════════════════════════════════════════════════════
        if position == 1 and exit_reason not in ("circuit_breaker_exit",):
            projected = equity[i-1] * (1 + ret * position_size)
            max_equity_in_trade = max(max_equity_in_trade, projected)
            cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
            max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

            rs = get_feat("rsi_slope_5d", i)
            hl = get_feat("higher_lows_count", i)
            bs = get_feat("breakout_setup_score", i)
            in_uptrend = rs > 0 and hl >= 2

            # ATR-based stop loss
            if atr14 is not None and close is not None and not np.isnan(atr14[i]):
                atr_stop = 2.5 * atr14[i] / close[i]
                atr_stop = max(0.03, min(atr_stop, 0.08))
            else:
                atr_stop = 0.05

            if cum_ret <= -atr_stop:
                new_position = 0
                exit_reason = "stop_loss"
                stats["n_sl"] += 1
            elif max_profit > 0.05 and new_position == 1:
                if max_profit > 0.20:
                    trail_pct = 0.35 if not in_uptrend else 0.50
                elif max_profit > 0.12:
                    trail_pct = 0.45 if not in_uptrend else 0.60
                elif max_profit > 0.05:
                    trail_pct = 0.60 if not in_uptrend else 0.75
                else:
                    trail_pct = 0.80

                giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                if giveback >= trail_pct:
                    new_position = 0
                    exit_reason = "trailing_stop"
                    stats["n_trail"] += 1

            # Trend continuation
            if new_position == 0 and exit_reason == "signal":
                if cum_ret > 0 and bs >= 3 and hl >= 3 and rs > 0:
                    new_position = 1
                    stats["n_trend_hold"] += 1

        # ══════════════════════════════════════════════════════════════
        # EXECUTE TRADE
        # ══════════════════════════════════════════════════════════════
        cost = 0
        if new_position != position:
            if new_position == 1:
                deploy = equity[i-1] * position_size
                cost = deploy * commission
                entry_equity = deploy - cost
                max_equity_in_trade = entry_equity
                current_entry_day = i
                hold_days = 0
            else:
                cost = equity[i-1] * position_size * (commission + tax)
                exit_eq = equity[i-1] - cost
                pnl = exit_eq - entry_equity if entry_equity > 0 else 0
                trades.append({
                    "entry_day": current_entry_day, "exit_day": i,
                    "holding_days": i - current_entry_day,
                    "pnl": pnl, "pnl_pct": (pnl / entry_equity * 100) if entry_equity > 0 else 0,
                    "is_win": pnl > 0, "exit_reason": exit_reason,
                    "position_size": position_size,
                })
                total_commission += cost

                # V5: Track consecutive losses
                if pnl <= 0:
                    consecutive_losses += 1
                    if consecutive_losses >= consec_loss_pause:
                        pause_until = i + pause_bars
                        consecutive_losses = 0
                else:
                    consecutive_losses = 0

                entry_equity = 0
                max_equity_in_trade = 0
                position_size = 1.0

        if position == 1:
            equity[i] = equity[i-1] * (1 + ret * position_size) - cost
            hold_days += 1
        else:
            equity[i] = equity[i-1] - cost

        position = new_position

    # Close open trade
    if position == 1 and entry_equity > 0:
        pnl = equity[-1] - entry_equity
        trades.append({
            "entry_day": current_entry_day, "exit_day": n-1,
            "holding_days": n-1-current_entry_day,
            "pnl": pnl, "pnl_pct": (pnl/entry_equity*100) if entry_equity > 0 else 0,
            "is_win": pnl > 0, "exit_reason": "end", "position_size": position_size,
        })

    # ── COMPUTE METRICS ──
    total_ret = (equity[-1] / initial_capital - 1) * 100
    years = n / 252
    ann_ret = ((equity[-1] / initial_capital) ** (1/max(years, 0.01)) - 1) * 100
    daily_rets = np.diff(equity) / equity[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    sharpe = (np.sqrt(252) * daily_rets.mean() / daily_rets.std()) if len(daily_rets) > 0 and daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min() * 100

    n_trades = len(trades)
    wins = [t for t in trades if t["is_win"]]
    losses = [t for t in trades if not t["is_win"]]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    max_loss = min([t["pnl_pct"] for t in losses]) if losses else 0
    max_win = max([t["pnl_pct"] for t in wins]) if wins else 0
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = gross_w / gross_l if gross_l > 0 else float('inf')
    avg_hold = np.mean([t["holding_days"] for t in trades]) if trades else 0
    expectancy = (win_rate/100 * avg_win + (1-win_rate/100) * avg_loss) if n_trades > 0 else 0

    return {
        "total_return_pct": round(total_ret, 2),
        "ann_return_pct": round(ann_ret, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "win_rate_pct": round(win_rate, 1),
        "total_trades": n_trades,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "max_win_pct": round(max_win, 2),
        "max_loss_pct": round(max_loss, 2),
        "avg_hold_days": round(avg_hold, 1),
        "expectancy_pct": round(expectancy, 2),
        "final_equity": round(equity[-1]),
        "equity_curve": equity,
        "trades": trades,
        **stats,
    }


def print_v5_window(window_label, bt, mode):
    icon = {"v4": "🚀", "v5": "🛡️"}[mode]
    filt = f"Filt:{bt['n_filtered']:>2d} Reg:{bt['n_regime_filtered']:>2d}"
    v5_extra = ""
    if mode == "v5":
        v5_extra = (f" EqF:{bt['n_eq_filter']:>2d} CB:{bt['n_circuit_breaker']:>2d}"
                    f" Rec:{bt['n_recovery_half']:>2d} Pau:{bt['n_consec_pause']:>2d}")
    print(f"   {icon} {window_label:30s} | "
          f"Ret:{bt['total_return_pct']:>+7.2f}% | "
          f"WR:{bt['win_rate_pct']:>5.1f}% | "
          f"PF:{bt['profit_factor']:>5.2f} | "
          f"Trades:{bt['total_trades']:>3d} | "
          f"DD:{bt['max_dd_pct']:>+6.1f}% | "
          f"AvgW:{bt['avg_win_pct']:>+5.1f}% | "
          f"SL:{bt['n_sl']:>2d} TS:{bt['n_trail']:>2d} {filt}{v5_extra}")


def print_v5_aggregate(model_name, mode, bt):
    print(f"\n   📊 AGGREGATE: {model_name} ({mode})")
    print(f"   {'─' * 70}")
    keys = ["total_return_pct", "ann_return_pct", "sharpe", "max_dd_pct",
            "profit_factor", "win_rate_pct", "total_trades",
            "avg_win_pct", "avg_loss_pct", "max_win_pct", "max_loss_pct",
            "avg_hold_days", "expectancy_pct"]
    for k in keys:
        print(f"   {k:25s}: {bt[k]}")
    extras = ["n_filtered", "n_regime_filtered", "n_sl", "n_trail",
              "n_trend_hold", "n_min_hold_skip", "n_partial_size",
              "n_eq_filter", "n_circuit_breaker", "n_consec_pause",
              "n_recovery_half", "circuit_breaker_days"]
    for k in extras:
        print(f"   {k:25s}: {bt.get(k, 0)}")


def main():
    parser = argparse.ArgumentParser(description="V5 Equity Curve Filter Analysis")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--capital", type=float, default=100_000_000)
    # V5 tuning params
    parser.add_argument("--eq-ema", type=int, default=30, help="EMA span for equity curve")
    parser.add_argument("--dd-cb", type=float, default=-0.30, help="DD threshold for deep DD mode")
    parser.add_argument("--cold-size", type=float, default=0.5, help="Position size in cold mode")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 10, "long_window": 40, "classes": 3},
    }

    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)

    max_sym = None if args.full else args.symbols
    symbols = loader.symbols[:max_sym] if max_sym else None
    model_names = args.models or ["random_forest", "xgboost", "lightgbm"]
    available = get_available_models()
    model_names = [m for m in model_names if m in available]

    feat_set = "leading"

    print("=" * 120)
    print("🛡️  V5 EQUITY CURVE FILTER — Reduce MaxDD while preserving returns")
    print(f"   Features: {feat_set} | Models: {model_names} | Symbols: {max_sym or 'all'}")
    print(f"   Vốn: {args.capital:,.0f} VND")
    print(f"   V5 params: EMA={args.eq_ema}, DD_CB={args.dd_cb}, Cold_Size={args.cold_size}")
    print("=" * 120)

    print("\n📦 Loading data...")
    raw_df = loader.load_all(symbols=symbols)
    print(f"   {len(raw_df)} rows, {raw_df['symbol'].nunique()} symbols")

    engine = FeatureEngine(feature_set=feat_set)
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_results = []
    per_window_results = []
    modes = ["v4", "v5"]

    for model_name in model_names:
        for mode in modes:
            print(f"\n{'─' * 100}")
            print(f"🤖 {model_name} | Mode: {mode.upper()}")

            agg_y_test, agg_y_pred, agg_returns = [], [], []
            agg_df_parts = []

            for window, train_df, test_df in splitter.split(df):
                try:
                    model = build_model(model_name)
                    X_train = np.nan_to_num(train_df[feature_cols].values)
                    y_train = train_df["target"].values.astype(int)
                    X_test = np.nan_to_num(test_df[feature_cols].values)
                    y_test = test_df["target"].values.astype(int)

                    offset = 0
                    if model_name == "xgboost" and y_train.min() < 0:
                        offset = abs(y_train.min())
                        y_train = y_train + offset

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    if offset > 0:
                        y_pred = y_pred - offset

                    rets = test_df["return_1d"].values if "return_1d" in test_df.columns else np.zeros(len(test_df))

                    if mode == "v4":
                        bt = backtest_v4(y_pred, rets, test_df.reset_index(drop=True),
                                         feature_cols, args.capital, mode="v4")
                    else:
                        bt = backtest_v5(y_pred, rets, test_df.reset_index(drop=True),
                                         feature_cols, args.capital, mode="v5",
                                         eq_ema_span=args.eq_ema,
                                         dd_circuit_breaker=args.dd_cb,
                                         cold_size=args.cold_size)

                    print_v5_window(window.label, bt, mode)

                    per_window_results.append({
                        "model": model_name, "mode": mode, "window": window.label,
                        "return_pct": bt["total_return_pct"], "win_rate": bt["win_rate_pct"],
                        "pf": bt["profit_factor"], "trades": bt["total_trades"],
                        "max_dd": bt["max_dd_pct"],
                        "avg_win": bt["avg_win_pct"], "avg_loss": bt["avg_loss_pct"],
                        "max_win": bt["max_win_pct"],
                    })

                    agg_y_test.extend(y_test.tolist())
                    agg_y_pred.extend(y_pred.tolist())
                    agg_returns.extend(rets.tolist())
                    agg_df_parts.append(test_df.reset_index(drop=True))
                except Exception as e:
                    print(f"   ❌ {window.label}: {e}")

            if agg_returns:
                agg_rets = np.array(agg_returns)
                agg_preds = np.array(agg_y_pred)
                agg_test_df = pd.concat(agg_df_parts, ignore_index=True)

                if mode == "v4":
                    agg_bt = backtest_v4(agg_preds, agg_rets, agg_test_df,
                                         feature_cols, args.capital, mode="v4")
                else:
                    agg_bt = backtest_v5(agg_preds, agg_rets, agg_test_df,
                                         feature_cols, args.capital, mode="v5",
                                         eq_ema_span=args.eq_ema,
                                         dd_circuit_breaker=args.dd_cb,
                                         cold_size=args.cold_size)

                agg_metrics = compute_metrics(agg_y_test, agg_y_pred)
                print_v5_aggregate(model_name, mode, agg_bt)

                all_results.append({
                    "model": model_name, "mode": mode,
                    "total_return_pct": agg_bt["total_return_pct"],
                    "ann_return_pct": agg_bt["ann_return_pct"],
                    "sharpe": agg_bt["sharpe"],
                    "max_dd_pct": agg_bt["max_dd_pct"],
                    "profit_factor": agg_bt["profit_factor"],
                    "win_rate_pct": agg_bt["win_rate_pct"],
                    "trades": agg_bt["total_trades"],
                    "avg_win_pct": agg_bt["avg_win_pct"],
                    "avg_loss_pct": agg_bt["avg_loss_pct"],
                    "max_win_pct": agg_bt["max_win_pct"],
                    "max_loss_pct": agg_bt["max_loss_pct"],
                    "avg_hold": agg_bt["avg_hold_days"],
                    "expectancy": agg_bt["expectancy_pct"],
                    "final_equity": agg_bt["final_equity"],
                    "f1": agg_metrics["f1_macro"],
                    "n_filtered": agg_bt.get("n_filtered", 0),
                    "n_regime": agg_bt.get("n_regime_filtered", 0),
                    "n_sl": agg_bt.get("n_sl", 0),
                    "n_trail": agg_bt.get("n_trail", 0),
                    "n_eq_filter": agg_bt.get("n_eq_filter", 0),
                    "n_circuit_breaker": agg_bt.get("n_circuit_breaker", 0),
                    "cb_days": agg_bt.get("circuit_breaker_days", 0),
                })

    # ══════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════════
    if all_results:
        print("\n" + "=" * 130)
        print("🏆 V4 vs V5 COMPARISON")
        print("=" * 130)
        res_df = pd.DataFrame(all_results).sort_values(["model", "mode"])
        print(res_df.to_string(index=False, float_format="%.2f"))

        # Improvement analysis
        print("\n" + "=" * 130)
        print("📊 V5 vs V4 — KEY CHANGES")
        print("=" * 130)

        compare_cols = ["total_return_pct", "sharpe", "max_dd_pct", "win_rate_pct",
                        "profit_factor", "avg_win_pct", "trades", "expectancy"]

        for mn in model_names:
            print(f"\n  📈 {mn.upper()}")
            v4_row = res_df[(res_df["model"] == mn) & (res_df["mode"] == "v4")]
            v5_row = res_df[(res_df["model"] == mn) & (res_df["mode"] == "v5")]
            if len(v4_row) == 0 or len(v5_row) == 0:
                continue
            v4 = v4_row.iloc[0]
            v5 = v5_row.iloc[0]

            print(f"  {'Metric':25s} | {'V4':>12s} | {'V5':>12s} | {'Delta':>12s}")
            print(f"  {'─' * 70}")
            for col in compare_cols:
                v4v = v4[col]
                v5v = v5[col]
                delta = v5v - v4v
                # DD improvement means less negative = better
                if col == "max_dd_pct":
                    icon = "🟢" if delta > 0 else "🔴"
                elif col == "trades":
                    icon = "⚪"
                else:
                    icon = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
                print(f"  {col:25s} | {v4v:>12.2f} | {v5v:>12.2f} | {icon}{delta:>+11.2f}")

            # V5-specific stats
            print(f"\n  V5 Protection Stats:")
            print(f"    Equity filter blocks : {v5.get('n_eq_filter', 0)}")
            print(f"    Circuit breaker acts  : {v5.get('n_circuit_breaker', 0)}")
            print(f"    Circuit breaker days  : {v5.get('cb_days', 0)}")

        # Per-window DD comparison
        print("\n" + "=" * 130)
        print("📅 PER-WINDOW MaxDD COMPARISON")
        print("=" * 130)
        pw_df = pd.DataFrame(per_window_results)
        for mn in model_names:
            print(f"\n  {mn}:")
            mn_pw = pw_df[pw_df["model"] == mn]
            for window in mn_pw["window"].unique():
                w = mn_pw[mn_pw["window"] == window]
                v4w = w[w["mode"] == "v4"]
                v5w = w[w["mode"] == "v5"]
                v4_dd = v4w["max_dd"].values[0] if len(v4w) > 0 else 0
                v5_dd = v5w["max_dd"].values[0] if len(v5w) > 0 else 0
                v4_ret = v4w["return_pct"].values[0] if len(v4w) > 0 else 0
                v5_ret = v5w["return_pct"].values[0] if len(v5w) > 0 else 0
                dd_icon = "🟢" if v5_dd > v4_dd else "🔴"
                print(f"    {window:35s} | V4: Ret{v4_ret:>+7.1f}% DD{v4_dd:>+6.1f}% | "
                      f"V5: Ret{v5_ret:>+7.1f}% DD{v5_dd:>+6.1f}% | {dd_icon} DD Δ{v5_dd-v4_dd:>+5.1f}%")

        # Save
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_df.to_csv(os.path.join(out_dir, f"v5_analysis_{ts}.csv"), index=False)
        pw_df.to_csv(os.path.join(out_dir, f"v5_per_window_{ts}.csv"), index=False)
        print(f"\n💾 Saved to results/v5_analysis_{ts}.csv")
        print(f"💾 Saved to results/v5_per_window_{ts}.csv")


if __name__ == "__main__":
    main()
