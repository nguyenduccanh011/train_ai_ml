"""
V4 DEEP ANALYSIS & OPTIMIZATION
================================
Phân tích chi tiết: V3 cải thiện gì, giảm gì, nguyên nhân gốc, và giải pháp V4.

V3 IMPROVEMENTS:
  🟢 Win rate: +15-16% (22% → 38%)
  🟢 Profit factor: +1.7x (1.2 → 3.0)
  🟢 Max loss/trade: cải thiện 7-9% (-14% → -5%)
  🟢 Avg loss/trade: cải thiện 0.5% (-2.2% → -1.6%)

V3 DECLINES:
  🔴 Total return: giảm mạnh (RF: 429→279, XGB: 65→-19, LGB: 101→-17)
  🔴 Avg win: giảm 2-3% (12% → 8-10%) — trailing stop cắt winner quá sớm
  🔴 Max drawdown: tệ hơn cho XGB/LGB — stop loss tạo realized losses
  🔴 Sharpe: giảm nhẹ

ROOT CAUSES OF DECLINE:
  1. Trailing stop cắt big winners — avg_win giảm 25% → mất lợi nhuận bull market
  2. Stop loss -5% lock in losses — trades có thể recover bị cắt sớm
  3. Serial compounding — bear market losses compound, trailing stop limit upside recovery
  4. Entry filter quá ít (chỉ 13-23 filtered) — chưa lọc đủ bad trades

V4 FIXES:
  A. REGIME FILTER: Không trade khi thị trường bear (price < SMA50, RSI < 40)
  B. DYNAMIC TRAILING: Dùng ATR-based trail thay vì % cố định, activate ở 5% thay vì 3%
  C. BETTER STOP LOSS: Trailing stop loss thay vì hard -5%, cho phép recovery
  D. STRONGER ENTRY: Score >= 3 (thay vì 2), thêm trend alignment check
  E. POSITION SIZING: Giảm size khi volatility cao (BB width > 80th percentile)
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


def backtest_v4(y_pred, returns, df_test, feature_cols, initial_capital=100_000_000,
                commission=0.0015, tax=0.001, mode="v4"):
    """
    V4 backtest: regime filter + dynamic trailing + better entries + position sizing.
    mode: 'original' = no filters, 'v3' = V3 logic, 'v4' = all V4 optimizations
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
    position_size = 1.0  # fraction of equity to deploy

    stats = {
        "n_filtered": 0, "n_regime_filtered": 0, "n_sl": 0,
        "n_trail": 0, "n_trend_hold": 0, "n_min_hold_skip": 0,
        "n_partial_size": 0,
    }

    # Extract features
    has_context = mode in ("v3", "v4") and len(df_test) == n
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

    # Compute simple moving averages for regime detection (V4)
    if mode == "v4" and "close" in df_test.columns:
        close = df_test["close"].values
        # SMA20 for short-term regime
        sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
        # SMA50 for medium-term regime 
        sma50 = pd.Series(close).rolling(50, min_periods=10).mean().values
        # ATR for dynamic stops
        if "high" in df_test.columns and "low" in df_test.columns:
            high = df_test["high"].values
            low = df_test["low"].values
            tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
            tr[0] = high[0] - low[0]
            atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values
        else:
            atr14 = np.full(n, close.mean() * 0.02)
    else:
        close = None
        sma20 = None
        sma50 = None
        atr14 = None

    def get_feat(name, idx):
        return feat_arrays[name][idx] if name in feat_arrays and idx < n else 0

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        new_position = 1 if pred == 1 else 0
        exit_reason = "signal"

        if mode == "original":
            # No filters at all
            pass

        elif mode in ("v3", "v4"):
            wp = get_feat("range_position_20d", i)
            dp = get_feat("dist_to_resistance", i)
            rs = get_feat("rsi_slope_5d", i)
            vs = get_feat("vol_surge_ratio", i)
            bs = get_feat("breakout_setup_score", i)
            hl = get_feat("higher_lows_count", i)
            od = get_feat("obv_price_divergence", i)
            bb = get_feat("bb_width_percentile", i)

            # ══════════════════════════════════════════════════════════════
            # V4 REGIME FILTER — Skip entries in bear market
            # ══════════════════════════════════════════════════════════════
            if mode == "v4" and new_position == 1 and position == 0:
                if close is not None and sma50 is not None:
                    # Bear regime: price below SMA50 AND RSI slope negative AND no bullish structure
                    price_below_sma50 = close[i] < sma50[i] if not np.isnan(sma50[i]) else False
                    price_below_sma20 = close[i] < sma20[i] if not np.isnan(sma20[i]) else False
                    
                    if price_below_sma50 and price_below_sma20 and rs <= 0:
                        # Strong bear — skip unless breakout setup is very strong
                        if bs < 3:
                            new_position = 0
                            stats["n_regime_filtered"] += 1

            # ══════════════════════════════════════════════════════════════
            # ENTRY QUALITY FILTER
            # ══════════════════════════════════════════════════════════════
            if new_position == 1 and position == 0:
                entry_score = 0
                if wp < 0.75: entry_score += 1      # Không mua đỉnh
                if dp > 0.02: entry_score += 1      # Không sát resistance
                if rs > 0: entry_score += 1          # Momentum tăng
                if vs > 1.1: entry_score += 1        # Volume xác nhận
                if hl >= 2: entry_score += 1         # Cấu trúc tăng

                # V4: Stricter threshold (3 vs 2 for V3)
                min_score = 3 if mode == "v4" else 2
                if entry_score < min_score:
                    new_position = 0
                    stats["n_filtered"] += 1

                # Strong reject: mua đỉnh không momentum
                if wp > 0.9 and rs <= 0 and bs < 2:
                    new_position = 0
                    stats["n_filtered"] += 1

                # V4: Additional reject — very high volatility without strong setup
                if mode == "v4" and bb > 0.85 and bs < 2 and entry_score < 4:
                    new_position = 0
                    stats["n_filtered"] += 1

            # ══════════════════════════════════════════════════════════════
            # V4 POSITION SIZING — Reduce size in high volatility
            # ══════════════════════════════════════════════════════════════
            if mode == "v4" and new_position == 1 and position == 0:
                if bb > 0.7:
                    position_size = 0.7  # 70% position in high vol
                    stats["n_partial_size"] += 1
                else:
                    position_size = 1.0

            # ══════════════════════════════════════════════════════════════
            # MIN HOLD FILTER
            # ══════════════════════════════════════════════════════════════
            if position == 1 and new_position == 0 and hold_days < 2:
                cum_ret = (equity[i-1] * (1 + ret) - entry_equity) / entry_equity if entry_equity > 0 else 0
                if cum_ret > 0.01:
                    new_position = 1
                    stats["n_min_hold_skip"] += 1

            # ══════════════════════════════════════════════════════════════
            # ADAPTIVE EXIT
            # ══════════════════════════════════════════════════════════════
            if position == 1:
                projected = equity[i-1] * (1 + ret * position_size)
                max_equity_in_trade = max(max_equity_in_trade, projected)
                cum_ret = (projected - entry_equity) / entry_equity if entry_equity > 0 else 0
                max_profit = (max_equity_in_trade - entry_equity) / entry_equity if entry_equity > 0 else 0

                in_uptrend = rs > 0 and hl >= 2

                if mode == "v4":
                    # ── V4 DYNAMIC STOP LOSS (ATR-based) ──
                    # Instead of hard -5%, use 2x ATR from entry
                    if atr14 is not None and close is not None and not np.isnan(atr14[i]):
                        atr_stop = 2.5 * atr14[i] / close[i]  # as percentage
                        atr_stop = max(0.03, min(atr_stop, 0.08))  # clamp 3%-8%
                    else:
                        atr_stop = 0.05

                    if cum_ret <= -atr_stop:
                        new_position = 0
                        exit_reason = "stop_loss"
                        stats["n_sl"] += 1

                    # ── V4 TRAILING: Activate later (5%), wider trail ──
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

                else:
                    # V3 logic (unchanged)
                    if cum_ret <= -0.05:
                        new_position = 0
                        exit_reason = "stop_loss"
                        stats["n_sl"] += 1
                    elif max_profit > 0.03 and new_position == 1:
                        if max_profit > 0.15:
                            trail_pct = 0.40 if not in_uptrend else 0.55
                        elif max_profit > 0.08:
                            trail_pct = 0.50 if not in_uptrend else 0.65
                        elif max_profit > 0.03:
                            trail_pct = 0.70
                        else:
                            trail_pct = 0.80
                        giveback = 1 - (cum_ret / max_profit) if max_profit > 0 else 0
                        if giveback >= trail_pct:
                            new_position = 0
                            exit_reason = "trailing_stop"
                            stats["n_trail"] += 1

                # ── TREND CONTINUATION: Don't sell if strong trend ──
                if new_position == 0 and exit_reason == "signal":
                    if cum_ret > 0 and bs >= 3 and hl >= 3 and rs > 0:
                        new_position = 1
                        stats["n_trend_hold"] += 1

        # ══════════════════════════════════════════════════════════════
        # EXECUTE TRADE
        # ══════════════════════════════════════════════════════════════
        cost = 0
        effective_ret = ret * (position_size if position == 1 else 1.0)

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
                # Adjust for position sizing — PnL is on deployed portion
                trades.append({
                    "entry_day": current_entry_day, "exit_day": i,
                    "holding_days": i - current_entry_day,
                    "pnl": pnl, "pnl_pct": (pnl / entry_equity * 100) if entry_equity > 0 else 0,
                    "is_win": pnl > 0, "exit_reason": exit_reason,
                    "position_size": position_size,
                })
                total_commission += cost
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

    # Expectancy per trade
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


def print_window_result(window_label, bt, mode, tag=""):
    filt_str = ""
    if mode == "v4":
        filt_str = f"Filt:{bt['n_filtered']:>2d} Reg:{bt['n_regime_filtered']:>2d}"
    elif mode == "v3":
        filt_str = f"Filt:{bt['n_filtered']:>2d}"
    
    icon = {"original": "📊", "v3": "🛡️", "v4": "🚀"}[mode]
    print(f"   {icon} {window_label:30s} | "
          f"Ret:{bt['total_return_pct']:>+7.2f}% | "
          f"WR:{bt['win_rate_pct']:>5.1f}% | "
          f"PF:{bt['profit_factor']:>5.2f} | "
          f"Trades:{bt['total_trades']:>3d} | "
          f"AvgW:{bt['avg_win_pct']:>+5.1f}% | "
          f"SL:{bt['n_sl']:>2d} TS:{bt['n_trail']:>2d} {filt_str}")


def print_aggregate(model_name, mode, bt):
    print(f"\n   📊 AGGREGATE: {model_name} ({mode})")
    print(f"   {'─' * 70}")
    keys = ["total_return_pct", "ann_return_pct", "sharpe", "max_dd_pct",
            "profit_factor", "win_rate_pct", "total_trades",
            "avg_win_pct", "avg_loss_pct", "max_win_pct", "max_loss_pct",
            "avg_hold_days", "expectancy_pct"]
    for k in keys:
        print(f"   {k:25s}: {bt[k]}")
    if mode in ("v3", "v4"):
        extras = ["n_filtered", "n_sl", "n_trail", "n_trend_hold", "n_min_hold_skip"]
        if mode == "v4":
            extras = ["n_filtered", "n_regime_filtered", "n_sl", "n_trail",
                       "n_trend_hold", "n_min_hold_skip", "n_partial_size"]
        for k in extras:
            print(f"   {k:25s}: {bt.get(k, 0)}")


def main():
    parser = argparse.ArgumentParser(description="V4 Analysis & Backtest")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--symbols", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--modes", nargs="+", default=["original", "v3", "v4"])
    parser.add_argument("--capital", type=float, default=100_000_000)
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

    print("=" * 110)
    print("🔬 V4 DEEP ANALYSIS — Original vs V3 vs V4")
    print(f"   Features: {feat_set} | Models: {model_names} | Symbols: {max_sym or 'all'}")
    print(f"   Vốn: {args.capital:,.0f} VND | Modes: {args.modes}")
    print("=" * 110)

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

    for model_name in model_names:
        for mode in args.modes:
            print(f"\n{'─' * 90}")
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
                    bt = backtest_v4(y_pred, rets, test_df.reset_index(drop=True),
                                     feature_cols, args.capital, mode=mode)

                    print_window_result(window.label, bt, mode)

                    per_window_results.append({
                        "model": model_name, "mode": mode, "window": window.label,
                        "return_pct": bt["total_return_pct"], "win_rate": bt["win_rate_pct"],
                        "pf": bt["profit_factor"], "trades": bt["total_trades"],
                        "avg_win": bt["avg_win_pct"], "avg_loss": bt["avg_loss_pct"],
                        "max_loss": bt["max_loss_pct"], "max_win": bt["max_win_pct"],
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
                agg_bt = backtest_v4(agg_preds, agg_rets, agg_test_df,
                                     feature_cols, args.capital, mode=mode)
                agg_metrics = compute_metrics(agg_y_test, agg_y_pred)

                print_aggregate(model_name, mode, agg_bt)

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
                    "n_partial": agg_bt.get("n_partial_size", 0),
                })

    # ══════════════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════
    if all_results:
        print("\n" + "=" * 130)
        print("🏆 FINAL COMPARISON — Original vs V3 vs V4")
        print("=" * 130)
        res_df = pd.DataFrame(all_results).sort_values(["model", "mode"])
        print(res_df.to_string(index=False, float_format="%.2f"))

        # ── IMPROVEMENT ANALYSIS ──
        print("\n" + "=" * 130)
        print("📊 CHI TIẾT CẢI THIỆN / SUY GIẢM")
        print("=" * 130)

        compare_cols = ["total_return_pct", "sharpe", "win_rate_pct", "profit_factor",
                        "max_dd_pct", "avg_win_pct", "avg_loss_pct", "max_loss_pct",
                        "trades", "expectancy", "avg_hold"]

        for mn in model_names:
            print(f"\n{'━' * 100}")
            print(f"  📈 {mn.upper()}")
            print(f"{'━' * 100}")

            rows = {}
            for mode in args.modes:
                r = res_df[(res_df["model"] == mn) & (res_df["mode"] == mode)]
                if len(r) > 0:
                    rows[mode] = r.iloc[0]

            if len(rows) < 2:
                continue

            # Header
            mode_labels = list(rows.keys())
            header = f"  {'Metric':25s}"
            for m in mode_labels:
                header += f" | {m:>12s}"
            if len(mode_labels) >= 2:
                header += f" | {'Δ(last-first)':>14s}"
            print(header)
            print(f"  {'─' * (25 + 15 * len(mode_labels) + 16)}")

            for col in compare_cols:
                vals = [rows[m][col] if col in rows[m] else 0 for m in mode_labels]
                line = f"  {col:25s}"
                for v in vals:
                    line += f" | {v:>12.2f}"
                delta = vals[-1] - vals[0]
                # Determine if improvement
                worse_is_lower = col in ("max_dd_pct", "avg_loss_pct", "max_loss_pct")
                if worse_is_lower:
                    icon = "🟢" if delta > 0 else "🔴"
                else:
                    icon = "🟢" if delta > 0 else "🔴"
                    if col == "trades":
                        icon = "⚪"  # neutral
                line += f" | {icon}{delta:>+12.2f}"
                print(line)

        # ── PER-WINDOW COMPARISON ──
        print("\n" + "=" * 130)
        print("📅 SO SÁNH THEO TỪNG WINDOW (Random Forest)")
        print("=" * 130)
        pw_df = pd.DataFrame(per_window_results)
        rf_pw = pw_df[pw_df["model"] == "random_forest"]
        if len(rf_pw) > 0:
            for window in rf_pw["window"].unique():
                w_data = rf_pw[rf_pw["window"] == window]
                print(f"\n  {window}:")
                for _, row in w_data.iterrows():
                    icon = {"original": "📊", "v3": "🛡️", "v4": "🚀"}.get(row["mode"], "")
                    print(f"    {icon} {row['mode']:10s} | Ret:{row['return_pct']:>+7.2f}% | "
                          f"WR:{row['win_rate']:>5.1f}% | PF:{row['pf']:>5.2f} | "
                          f"Trades:{row['trades']:>3.0f} | AvgW:{row['avg_win']:>+5.1f}% | MaxW:{row['max_win']:>+5.1f}%")

        # Save
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        res_df.to_csv(os.path.join(out_dir, f"v4_analysis_{ts}.csv"), index=False)
        pw_df.to_csv(os.path.join(out_dir, f"v4_per_window_{ts}.csv"), index=False)
        print(f"\n💾 Saved to results/v4_analysis_{ts}.csv")
        print(f"💾 Saved to results/v4_per_window_{ts}.csv")


if __name__ == "__main__":
    main()
