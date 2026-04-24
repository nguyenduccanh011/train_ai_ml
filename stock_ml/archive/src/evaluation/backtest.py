"""
Backtest simulator with smart exit strategies.
Supports: stop-loss, trailing stop, and combo mode.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Represents a single round-trip trade (buy → sell)."""
    entry_day: int
    entry_equity: float
    exit_day: int = 0
    exit_equity: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    holding_days: int = 0
    is_win: bool = False
    exit_reason: str = "signal"  # signal, stop_loss, trailing_stop


@dataclass
class SmartExitConfig:
    """Configuration for smart exit strategies."""
    stop_loss_pct: float = -0.07       # -7% stop-loss
    trailing_stop_pct: float = 0.50    # sell when price drops 50% from max profit
    enable_stop_loss: bool = True
    enable_trailing_stop: bool = True


def backtest_predictions(
    y_pred: np.ndarray,
    returns: np.ndarray,
    initial_capital: float = 100_000_000,
    commission: float = 0.0015,
    tax: float = 0.001,
    smart_exit: Optional[SmartExitConfig] = None,
) -> Dict[str, Any]:
    """
    Backtest with full trade-level tracking and smart exit strategies.
    
    Smart exit logic:
    - Stop-loss: Force sell when cumulative return from entry < stop_loss_pct
    - Trailing stop: Force sell when price drops trailing_stop_pct from max profit
    """
    n = len(y_pred)
    equity = np.zeros(n)
    equity[0] = initial_capital
    position = 0  # 0=cash, 1=invested
    trades_list: List[Trade] = []
    current_trade: Trade = None
    total_commission = 0
    daily_pnl = np.zeros(n)

    # Smart exit tracking
    trade_entry_equity = 0
    trade_max_equity = 0
    smart_exit_triggered = False
    exit_reason = "signal"
    
    # Stats for smart exit
    n_stop_loss = 0
    n_trailing_stop = 0

    for i in range(1, n):
        pred = int(y_pred[i - 1])
        ret = returns[i] if not np.isnan(returns[i]) else 0
        
        # Default: follow model signal
        new_position = 1 if pred == 1 else 0
        exit_reason = "signal"

        # ── Smart exit check (only when in position) ──
        if position == 1 and smart_exit is not None:
            # Calculate current equity before today's return
            current_eq = equity[i - 1]
            # Simulate today's return
            projected_eq = current_eq * (1 + ret)
            
            # Track max equity during trade
            trade_max_equity = max(trade_max_equity, projected_eq)
            
            # 1) Stop-loss check
            if smart_exit.enable_stop_loss:
                cum_return = (projected_eq - trade_entry_equity) / trade_entry_equity
                if cum_return <= smart_exit.stop_loss_pct:
                    new_position = 0
                    exit_reason = "stop_loss"
                    n_stop_loss += 1
            
            # 2) Trailing stop check (only if we had some profit)
            if smart_exit.enable_trailing_stop and new_position == 1:
                max_profit = (trade_max_equity - trade_entry_equity) / trade_entry_equity
                current_profit = (projected_eq - trade_entry_equity) / trade_entry_equity
                if max_profit > 0.02:  # Only activate after 2% profit
                    giveback_ratio = 1 - (current_profit / max_profit) if max_profit > 0 else 0
                    if giveback_ratio >= smart_exit.trailing_stop_pct:
                        new_position = 0
                        exit_reason = "trailing_stop"
                        n_trailing_stop += 1

        cost = 0
        if new_position != position:
            if new_position == 1:
                # BUY - open trade
                cost = equity[i - 1] * commission
                entry_eq = equity[i - 1] - cost
                current_trade = Trade(entry_day=i, entry_equity=entry_eq)
                trade_entry_equity = entry_eq
                trade_max_equity = entry_eq
            else:
                # SELL - close trade
                cost = equity[i - 1] * (commission + tax)
                if current_trade is not None:
                    current_trade.exit_day = i
                    current_trade.exit_equity = equity[i - 1] - cost
                    current_trade.holding_days = i - current_trade.entry_day
                    current_trade.pnl = current_trade.exit_equity - current_trade.entry_equity
                    current_trade.pnl_pct = (current_trade.pnl / current_trade.entry_equity * 100) if current_trade.entry_equity > 0 else 0
                    current_trade.is_win = current_trade.pnl > 0
                    current_trade.exit_reason = exit_reason
                    trades_list.append(current_trade)
                    current_trade = None
                    trade_entry_equity = 0
                    trade_max_equity = 0
                total_commission += cost

        # PnL
        if position == 1:
            pnl = equity[i - 1] * ret
        else:
            pnl = 0

        daily_pnl[i] = pnl
        equity[i] = equity[i - 1] + pnl - cost
        position = new_position

    # Close any open trade at end
    if current_trade is not None:
        current_trade.exit_day = n - 1
        current_trade.exit_equity = equity[-1]
        current_trade.holding_days = (n - 1) - current_trade.entry_day
        current_trade.pnl = current_trade.exit_equity - current_trade.entry_equity
        current_trade.pnl_pct = (current_trade.pnl / current_trade.entry_equity * 100) if current_trade.entry_equity > 0 else 0
        current_trade.is_win = current_trade.pnl > 0
        current_trade.exit_reason = "end_of_data"
        trades_list.append(current_trade)

    # ── Basic metrics ──
    total_return = (equity[-1] / initial_capital - 1) * 100
    trading_days = n
    years = trading_days / 252

    bnh_equity = initial_capital * (1 + returns).cumprod()
    bnh_equity = np.nan_to_num(bnh_equity, nan=initial_capital)
    bnh_return = (bnh_equity[-1] / initial_capital - 1) * 100

    ann_return = ((equity[-1] / initial_capital) ** (1 / max(years, 0.01)) - 1) * 100

    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    sharpe = (np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100

    time_in_market = (np.array(y_pred) == 1).mean() * 100

    # ── Trade-level metrics ──
    n_trades = len(trades_list)
    winning_trades = [t for t in trades_list if t.is_win]
    losing_trades = [t for t in trades_list if not t.is_win]
    n_wins = len(winning_trades)
    n_losses = len(losing_trades)

    if n_trades > 0:
        win_rate = n_wins / n_trades * 100
        avg_holding_days = np.mean([t.holding_days for t in trades_list])
        avg_pnl_per_trade = np.mean([t.pnl for t in trades_list])
        avg_pnl_pct_per_trade = np.mean([t.pnl_pct for t in trades_list])
        median_holding_days = np.median([t.holding_days for t in trades_list])
    else:
        win_rate = 0
        avg_holding_days = 0
        avg_pnl_per_trade = 0
        avg_pnl_pct_per_trade = 0
        median_holding_days = 0

    if n_wins > 0:
        avg_win_pnl = np.mean([t.pnl for t in winning_trades])
        avg_win_pct = np.mean([t.pnl_pct for t in winning_trades])
        max_win_pnl = max(t.pnl for t in winning_trades)
        max_win_pct = max(t.pnl_pct for t in winning_trades)
        avg_win_hold = np.mean([t.holding_days for t in winning_trades])
    else:
        avg_win_pnl = avg_win_pct = max_win_pnl = max_win_pct = avg_win_hold = 0

    if n_losses > 0:
        avg_loss_pnl = np.mean([t.pnl for t in losing_trades])
        avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades])
        max_loss_pnl = min(t.pnl for t in losing_trades)
        max_loss_pct = min(t.pnl_pct for t in losing_trades)
        avg_loss_hold = np.mean([t.holding_days for t in losing_trades])
    else:
        avg_loss_pnl = avg_loss_pct = max_loss_pnl = max_loss_pct = avg_loss_hold = 0

    # Profit Factor
    gross_wins = sum(t.pnl for t in winning_trades)
    gross_losses = abs(sum(t.pnl for t in losing_trades))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Expectancy
    loss_rate = n_losses / n_trades if n_trades > 0 else 0
    expectancy = (avg_win_pnl * (n_wins / n_trades) + avg_loss_pnl * loss_rate) if n_trades > 0 else 0

    # Consecutive wins/losses
    max_consec_wins = _max_consecutive(trades_list, True)
    max_consec_losses = _max_consecutive(trades_list, False)

    # Exit reason stats
    exit_reasons = {}
    for t in trades_list:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        # ── Capital ──
        "initial_capital": initial_capital,
        "final_equity": round(equity[-1]),
        "total_return_pct": round(total_return, 2),
        "annualized_return_pct": round(ann_return, 2),
        "buy_hold_return_pct": round(bnh_return, 2),
        "excess_return_pct": round(total_return - bnh_return, 2),
        # ── Risk ──
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(profit_factor, 2),
        # ── Trade counts ──
        "total_trades": n_trades,
        "winning_trades": n_wins,
        "losing_trades": n_losses,
        "win_rate_pct": round(win_rate, 1),
        # ── Avg per trade ──
        "avg_pnl_per_trade": round(avg_pnl_per_trade),
        "avg_pnl_pct_per_trade": round(avg_pnl_pct_per_trade, 2),
        # ── Winning trades ──
        "avg_win_pnl": round(avg_win_pnl),
        "avg_win_pct": round(avg_win_pct, 2),
        "max_win_pnl": round(max_win_pnl),
        "max_win_pct": round(max_win_pct, 2),
        "avg_win_holding_days": round(avg_win_hold, 1),
        # ── Losing trades ──
        "avg_loss_pnl": round(avg_loss_pnl),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "max_loss_pnl": round(max_loss_pnl),
        "max_loss_pct": round(max_loss_pct, 2),
        "avg_loss_holding_days": round(avg_loss_hold, 1),
        # ── Holding & timing ──
        "avg_holding_days": round(avg_holding_days, 1),
        "median_holding_days": round(median_holding_days, 1),
        "time_in_market_pct": round(time_in_market, 1),
        "trading_days_total": trading_days,
        # ── Streaks ──
        "max_consecutive_wins": max_consec_wins,
        "max_consecutive_losses": max_consec_losses,
        # ── Advanced ──
        "expectancy": round(expectancy),
        "total_commission": round(total_commission),
        # ── Smart exit stats ──
        "exit_reasons": exit_reasons,
        "n_stop_loss": n_stop_loss,
        "n_trailing_stop": n_trailing_stop,
        # ── Curves ──
        "equity_curve": equity,
    }


def _max_consecutive(trades: List[Trade], win: bool) -> int:
    """Count max consecutive wins or losses."""
    max_streak = 0
    current = 0
    for t in trades:
        if t.is_win == win:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def format_backtest_report(results: Dict[str, Any]) -> str:
    """Format backtest results as a detailed Vietnamese report."""
    r = results
    cap = r["initial_capital"]

    def fmt_money(v):
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B"
        elif abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        else:
            return f"{v:,.0f}"

    profit = r["final_equity"] - cap
    w = 55  # width

    lines = [
        "┌" + "─" * w + "┐",
        "│" + "📈 BACKTEST REPORT CHI TIẾT".center(w) + "│",
        "├" + "─" * w + "┤",
        "│" + " 💰 VỐN & LỢI NHUẬN".ljust(w) + "│",
        f"│  Vốn ban đầu:         {fmt_money(cap):>25s} VND  │",
        f"│  Vốn cuối kỳ:         {fmt_money(r['final_equity']):>25s} VND  │",
        f"│  Lợi nhuận:           {fmt_money(profit):>25s} VND  │",
        f"│  Tổng return:          {r['total_return_pct']:>+23.2f}%  │",
        f"│  Return/năm:           {r['annualized_return_pct']:>+23.2f}%  │",
        f"│  Buy&Hold return:      {r['buy_hold_return_pct']:>+23.2f}%  │",
        f"│  Vượt benchmark:       {r['excess_return_pct']:>+23.2f}%  │",
        "├" + "─" * w + "┤",
        "│" + " 📊 CHỈ SỐ RỦI RO".ljust(w) + "│",
        f"│  Sharpe Ratio:         {r['sharpe_ratio']:>25.3f}  │",
        f"│  Max Drawdown:         {r['max_drawdown_pct']:>24.2f}%  │",
        f"│  Profit Factor:        {r['profit_factor']:>25.2f}  │",
        f"│  Expectancy/lệnh:     {fmt_money(r['expectancy']):>25s} VND  │",
        "├" + "─" * w + "┤",
        "│" + " 🔢 THỐNG KÊ LỆNH".ljust(w) + "│",
        f"│  Tổng số lệnh:        {r['total_trades']:>25d}  │",
        f"│  Lệnh thắng:          {r['winning_trades']:>25d}  │",
        f"│  Lệnh thua:           {r['losing_trades']:>25d}  │",
        f"│  Tỷ lệ thắng:         {r['win_rate_pct']:>24.1f}%  │",
        f"│  Chuỗi thắng dài nhất:{r['max_consecutive_wins']:>25d}  │",
        f"│  Chuỗi thua dài nhất: {r['max_consecutive_losses']:>25d}  │",
    ]

    # Smart exit stats
    if r.get("n_stop_loss", 0) > 0 or r.get("n_trailing_stop", 0) > 0:
        lines += [
            "├" + "─" * w + "┤",
            "│" + " 🛡️  SMART EXIT".ljust(w) + "│",
            f"│  Stop-loss triggered:  {r.get('n_stop_loss', 0):>25d}  │",
            f"│  Trailing stop triggered:{r.get('n_trailing_stop', 0):>23d}  │",
            f"│  Signal exit:          {r.get('exit_reasons', {}).get('signal', 0):>25d}  │",
        ]

    lines += [
        "├" + "─" * w + "┤",
        "│" + " ✅ LỆNH THẮNG".ljust(w) + "│",
        f"│  LN trung bình:       {fmt_money(r['avg_win_pnl']):>25s} VND  │",
        f"│  LN trung bình (%):   {r['avg_win_pct']:>+24.2f}%  │",
        f"│  LN lớn nhất:         {fmt_money(r['max_win_pnl']):>25s} VND  │",
        f"│  LN lớn nhất (%):     {r['max_win_pct']:>+24.2f}%  │",
        f"│  Giữ TB (ngày):       {r['avg_win_holding_days']:>25.1f}  │",
        "├" + "─" * w + "┤",
        "│" + " ❌ LỆNH THUA".ljust(w) + "│",
        f"│  Lỗ trung bình:       {fmt_money(r['avg_loss_pnl']):>25s} VND  │",
        f"│  Lỗ trung bình (%):   {r['avg_loss_pct']:>+24.2f}%  │",
        f"│  Lỗ lớn nhất:         {fmt_money(r['max_loss_pnl']):>25s} VND  │",
        f"│  Lỗ lớn nhất (%):     {r['max_loss_pct']:>+24.2f}%  │",
        f"│  Giữ TB (ngày):       {r['avg_loss_holding_days']:>25.1f}  │",
        "├" + "─" * w + "┤",
        "│" + " ⏱️  THỜI GIAN".ljust(w) + "│",
        f"│  Giữ TB/lệnh (ngày):  {r['avg_holding_days']:>25.1f}  │",
        f"│  Giữ median (ngày):   {r['median_holding_days']:>25.1f}  │",
        f"│  % thời gian đầu tư:  {r['time_in_market_pct']:>24.1f}%  │",
        f"│  Tổng ngày GD:        {r['trading_days_total']:>25d}  │",
        f"│  Phí GD tổng:         {fmt_money(r['total_commission']):>25s} VND  │",
        "└" + "─" * w + "┘",
    ]
    return "\n".join(lines)
