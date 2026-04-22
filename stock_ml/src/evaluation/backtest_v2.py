"""
Backtest V2 - Smart Exit with ATR Trailing Stop & 4-Level Exit System.

Strategy:
- Entry: Model predicts UPTREND (1)
- Exit: Smart 4-level system based on ATR, volume, trend alignment
- Re-entry: After exit, wait for model to predict UPTREND again + confirmation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TradeV2:
    """Enhanced trade tracking."""
    entry_day: int
    entry_price: float
    entry_equity: float
    exit_day: int = 0
    exit_price: float = 0
    exit_equity: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    holding_days: int = 0
    is_win: bool = False
    exit_reason: str = ""  # "model_signal", "trailing_stop", "partial_exit", "full_exit", "end_of_data"
    max_profit_pct: float = 0  # peak unrealized profit during trade
    max_drawdown_from_peak_pct: float = 0  # max dd from peak during trade


def backtest_smart_exit(
    y_pred: np.ndarray,
    returns: np.ndarray,
    prices: np.ndarray,
    atr: np.ndarray,
    volume_ratio: np.ndarray,
    rsi: np.ndarray,
    trend_short: np.ndarray,
    trend_medium: np.ndarray,
    initial_capital: float = 100_000_000,
    commission: float = 0.0015,
    tax: float = 0.001,
    # Smart exit params
    atr_multiplier_stop: float = 2.0,
    atr_multiplier_alert: float = 1.0,
    atr_multiplier_partial: float = 1.5,
    partial_exit_pct: float = 0.5,
    volume_spike_threshold: float = 2.0,
    min_hold_days: int = 2,
    cooldown_days: int = 1,
) -> Dict[str, Any]:
    """
    Smart exit backtest with ATR trailing stop and 4-level exit system.
    
    Exit Levels:
      L0 (HOLD): dd_from_peak < atr_multiplier_alert * ATR
      L1 (ALERT): dd_from_peak >= alert threshold, tighten stop
      L2 (PARTIAL): dd_from_peak >= partial threshold + volume spike or trend_short bearish
      L3 (FULL EXIT): dd_from_peak >= stop threshold, or trend_short+medium bearish
    """
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    
    position = 0.0  # 0=cash, 0.5=partial, 1.0=full
    trades_list: List[TradeV2] = []
    current_trade: Optional[TradeV2] = None
    total_commission = 0.0
    
    peak_equity_in_trade = 0.0
    days_since_exit = 999  # cooldown counter
    
    for i in range(1, n):
        pred = int(y_pred[i - 1]) if i - 1 < len(y_pred) else 0
        ret = returns[i] if not np.isnan(returns[i]) else 0.0
        current_atr_pct = atr[i] / prices[i] if prices[i] > 0 and not np.isnan(atr[i]) else 0.02
        vol_ratio = volume_ratio[i] if not np.isnan(volume_ratio[i]) else 1.0
        cur_rsi = rsi[i] if not np.isnan(rsi[i]) else 50.0
        t_short = int(trend_short[i]) if not np.isnan(trend_short[i]) else 1
        t_medium = int(trend_medium[i]) if not np.isnan(trend_medium[i]) else 1
        
        cost = 0.0
        exit_reason = ""
        
        if position > 0:
            # Update equity with current return
            pnl = equity[i - 1] * position * ret
            equity[i] = equity[i - 1] + pnl
            
            # Track peak
            if equity[i] > peak_equity_in_trade:
                peak_equity_in_trade = equity[i]
            
            # Drawdown from peak (within trade)
            dd_from_peak = (peak_equity_in_trade - equity[i]) / peak_equity_in_trade if peak_equity_in_trade > 0 else 0
            
            # Update trade tracking
            if current_trade is not None:
                unrealized_pct = (equity[i] - current_trade.entry_equity) / current_trade.entry_equity
                current_trade.max_profit_pct = max(current_trade.max_profit_pct, unrealized_pct * 100)
                current_trade.max_drawdown_from_peak_pct = max(current_trade.max_drawdown_from_peak_pct, dd_from_peak * 100)
            
            holding = (i - current_trade.entry_day) if current_trade else 0
            
            # ── SMART EXIT LOGIC ──
            
            # LEVEL 3: FULL EXIT - major reversal
            if (dd_from_peak >= atr_multiplier_stop * current_atr_pct and holding >= min_hold_days):
                exit_reason = "L3_full_stop"
            elif (t_short == 0 and t_medium == 0 and dd_from_peak >= atr_multiplier_alert * current_atr_pct and holding >= min_hold_days):
                exit_reason = "L3_trend_reversal"
            elif (vol_ratio >= volume_spike_threshold and dd_from_peak >= atr_multiplier_partial * current_atr_pct and holding >= min_hold_days):
                exit_reason = "L3_volume_spike"
            
            # LEVEL 2: PARTIAL EXIT
            elif (position == 1.0 and dd_from_peak >= atr_multiplier_partial * current_atr_pct 
                  and (t_short == 0 or vol_ratio >= volume_spike_threshold * 0.8)
                  and holding >= min_hold_days):
                # Reduce to partial position
                sell_fraction = partial_exit_pct
                sell_cost = equity[i] * sell_fraction * (commission + tax)
                total_commission += sell_cost
                equity[i] -= sell_cost
                position = 1.0 - sell_fraction
                # Don't close trade yet, just reduce
                continue
            
            # Model says exit (not uptrend)
            elif pred != 1 and holding >= min_hold_days:
                exit_reason = "model_signal"
            
            # Execute exit
            if exit_reason:
                sell_cost = equity[i] * position * (commission + tax)
                total_commission += sell_cost
                equity[i] -= sell_cost
                
                if current_trade is not None:
                    current_trade.exit_day = i
                    current_trade.exit_price = prices[i] if i < len(prices) else 0
                    current_trade.exit_equity = equity[i]
                    current_trade.holding_days = i - current_trade.entry_day
                    current_trade.pnl = current_trade.exit_equity - current_trade.entry_equity
                    current_trade.pnl_pct = (current_trade.pnl / current_trade.entry_equity * 100) if current_trade.entry_equity > 0 else 0
                    current_trade.is_win = current_trade.pnl > 0
                    current_trade.exit_reason = exit_reason
                    trades_list.append(current_trade)
                    current_trade = None
                
                position = 0.0
                days_since_exit = 0
                peak_equity_in_trade = 0
        
        else:
            # Not in position
            equity[i] = equity[i - 1]
            days_since_exit += 1
            
            # ── ENTRY LOGIC ──
            if pred == 1 and days_since_exit >= cooldown_days:
                # Confirmation: check trend + RSI
                enter = True
                
                # Don't enter if RSI > 80 (overbought)
                if cur_rsi > 80:
                    enter = False
                
                if enter:
                    buy_cost = equity[i] * commission
                    total_commission += buy_cost
                    equity[i] -= buy_cost
                    position = 1.0
                    peak_equity_in_trade = equity[i]
                    current_trade = TradeV2(
                        entry_day=i,
                        entry_price=prices[i] if i < len(prices) else 0,
                        entry_equity=equity[i],
                    )
    
    # Close any open trade at end
    if current_trade is not None:
        current_trade.exit_day = n - 1
        current_trade.exit_price = prices[-1] if len(prices) > 0 else 0
        current_trade.exit_equity = equity[-1]
        current_trade.holding_days = (n - 1) - current_trade.entry_day
        current_trade.pnl = current_trade.exit_equity - current_trade.entry_equity
        current_trade.pnl_pct = (current_trade.pnl / current_trade.entry_equity * 100) if current_trade.entry_equity > 0 else 0
        current_trade.is_win = current_trade.pnl > 0
        current_trade.exit_reason = "end_of_data"
        trades_list.append(current_trade)
    
    # ── Compute metrics ──
    total_return = (equity[-1] / initial_capital - 1) * 100
    trading_days = n
    years = trading_days / 252
    
    # Buy & Hold
    bnh_equity = initial_capital * np.cumprod(1 + np.nan_to_num(returns))
    bnh_return = (bnh_equity[-1] / initial_capital - 1) * 100
    
    ann_return = ((equity[-1] / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100
    
    daily_returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1)
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    sharpe = (np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
    
    peak = np.maximum.accumulate(equity)
    drawdown = np.where(peak > 0, (equity - peak) / peak, 0)
    max_dd = drawdown.min() * 100
    
    time_in_market = np.mean([1 if p > 0 else 0 for p in [position]]) * 100  # approximate
    
    # Trade stats
    n_trades = len(trades_list)
    winning_trades = [t for t in trades_list if t.is_win]
    losing_trades = [t for t in trades_list if not t.is_win]
    n_wins = len(winning_trades)
    n_losses = len(losing_trades)
    
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
    avg_pnl_pct = np.mean([t.pnl_pct for t in trades_list]) if n_trades > 0 else 0
    avg_hold = np.mean([t.holding_days for t in trades_list]) if n_trades > 0 else 0
    
    avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if n_wins > 0 else 0
    avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) if n_losses > 0 else 0
    max_win_pct = max([t.pnl_pct for t in winning_trades]) if n_wins > 0 else 0
    max_loss_pct = min([t.pnl_pct for t in losing_trades]) if n_losses > 0 else 0
    
    gross_wins = sum(t.pnl for t in winning_trades)
    gross_losses = abs(sum(t.pnl for t in losing_trades))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # Exit reason breakdown
    exit_reasons = {}
    for t in trades_list:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    # Calmar ratio
    calmar = abs(ann_return / max_dd) if max_dd != 0 else 0
    
    return {
        "initial_capital": initial_capital,
        "final_equity": round(equity[-1]),
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "buy_hold_return_pct": round(bnh_return, 2),
        "excess_return_pct": round(total_return - bnh_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar_ratio": round(calmar, 3),
        "profit_factor": round(profit_factor, 2),
        "total_trades": n_trades,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate_pct": round(win_rate, 1),
        "avg_pnl_pct_per_trade": round(avg_pnl_pct, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "max_win_pct": round(max_win_pct, 2),
        "max_loss_pct": round(max_loss_pct, 2),
        "avg_holding_days": round(avg_hold, 1),
        "total_commission": round(total_commission),
        "exit_reasons": exit_reasons,
        "equity_curve": equity,
    }


def backtest_per_symbol(
    symbol_data: Dict[str, Dict],
    initial_capital: float = 100_000_000,
    strategy: str = "smart_exit",
    **strategy_params,
) -> Dict[str, Any]:
    """
    Proper portfolio backtest: allocate capital equally across symbols,
    backtest each independently, then aggregate.
    """
    n_symbols = len(symbol_data)
    if n_symbols == 0:
        return {}
    
    capital_per_symbol = initial_capital / n_symbols
    
    all_symbol_results = {}
    total_final_equity = 0
    
    for symbol, data in symbol_data.items():
        y_pred = data["y_pred"]
        returns = data["returns"]
        
        if strategy == "smart_exit":
            result = backtest_smart_exit(
                y_pred=y_pred,
                returns=returns,
                prices=data.get("prices", np.ones(len(returns))),
                atr=data.get("atr", np.full(len(returns), np.nan)),
                volume_ratio=data.get("volume_ratio", np.ones(len(returns))),
                rsi=data.get("rsi", np.full(len(returns), 50.0)),
                trend_short=data.get("trend_short", np.ones(len(returns))),
                trend_medium=data.get("trend_medium", np.ones(len(returns))),
                initial_capital=capital_per_symbol,
                **strategy_params,
            )
        else:
            # Original simple backtest
            from .backtest import backtest_predictions
            result = backtest_predictions(
                y_pred=y_pred,
                returns=returns,
                initial_capital=capital_per_symbol,
            )
        
        all_symbol_results[symbol] = result
        total_final_equity += result["final_equity"]
    
    # Aggregate portfolio metrics
    portfolio_return = (total_final_equity / initial_capital - 1) * 100
    avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_symbol_results.values()])
    avg_max_dd = np.mean([r["max_drawdown_pct"] for r in all_symbol_results.values()])
    worst_dd = min([r["max_drawdown_pct"] for r in all_symbol_results.values()])
    total_trades = sum([r["total_trades"] for r in all_symbol_results.values()])
    total_wins = sum([r["winning_trades"] for r in all_symbol_results.values()])
    avg_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    # Buy & Hold portfolio
    avg_bnh = np.mean([r["buy_hold_return_pct"] for r in all_symbol_results.values()])
    
    years = np.mean([r.get("avg_holding_days", 252) for r in all_symbol_results.values()]) * total_trades / 252 / n_symbols
    years = max(years, 0.5)
    ann_return = ((total_final_equity / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else portfolio_return
    
    return {
        "portfolio": {
            "initial_capital": initial_capital,
            "final_equity": round(total_final_equity),
            "total_return_pct": round(portfolio_return, 2),
            "buy_hold_return_pct": round(avg_bnh, 2),
            "excess_return_pct": round(portfolio_return - avg_bnh, 2),
            "avg_sharpe": round(avg_sharpe, 3),
            "avg_max_dd_pct": round(avg_max_dd, 2),
            "worst_dd_pct": round(worst_dd, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(avg_win_rate, 1),
            "n_symbols": n_symbols,
        },
        "per_symbol": all_symbol_results,
    }


def format_smart_report(results: Dict[str, Any], symbol: str = "") -> str:
    """Format smart exit backtest results."""
    r = results
    cap = r["initial_capital"]
    profit = r["final_equity"] - cap
    
    def fmt(v):
        if abs(v) >= 1e9: return f"{v/1e9:.2f}B"
        elif abs(v) >= 1e6: return f"{v/1e6:.1f}M"
        else: return f"{v:,.0f}"
    
    lines = [
        f"{'─'*55}",
        f"📈 {symbol} | Return: {r['total_return_pct']:+.2f}% | B&H: {r['buy_hold_return_pct']:+.2f}% | Excess: {r['excess_return_pct']:+.2f}%",
        f"   Sharpe: {r['sharpe_ratio']:.3f} | MaxDD: {r['max_drawdown_pct']:.2f}% | Calmar: {r.get('calmar_ratio', 0):.3f}",
        f"   Trades: {r['total_trades']} | WinRate: {r['win_rate_pct']:.1f}% | PF: {r['profit_factor']:.2f}",
        f"   AvgWin: {r['avg_win_pct']:+.2f}% | AvgLoss: {r['avg_loss_pct']:+.2f}% | AvgHold: {r['avg_holding_days']:.1f}d",
        f"   Profit: {fmt(profit)} VND | Commission: {fmt(r['total_commission'])} VND",
    ]
    
    if "exit_reasons" in r and r["exit_reasons"]:
        reasons_str = ", ".join([f"{k}:{v}" for k, v in sorted(r["exit_reasons"].items())])
        lines.append(f"   Exit reasons: {reasons_str}")
    
    return "\n".join(lines)
