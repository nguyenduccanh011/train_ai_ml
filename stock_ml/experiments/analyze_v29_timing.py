"""
Analyze V29 trades for entry/exit timing weaknesses:
  - Losing trades that bought near peaks (price 5/10/20 days earlier was lower)
  - Winning/losing trades that sold too early (price 5/10 days later was higher)
  - Trades that sold too late (price 5/10 days before exit was higher)
Output a structured report for V30 design.
"""
import os, sys, glob, json
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.normpath(os.path.join(ROOT, "..", "portable_data", "vn_stock_ai_dataset_cleaned", "all_symbols"))


def load_prices(symbol):
    path = os.path.join(DATA_DIR, f"symbol={symbol}", "timeframe=1D", "data.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]


def nearest_idx(df, date):
    date = pd.to_datetime(date)
    if getattr(date, "tzinfo", None) is not None:
        date = date.tz_localize(None) if hasattr(date, "tz_localize") else date.replace(tzinfo=None)
    pos = df["timestamp"].searchsorted(date)
    if pos >= len(df): pos = len(df) - 1
    return pos


def analyze(trade_csv, label):
    print(f"\n{'='*100}\n  {label}: {trade_csv}\n{'='*100}")
    df = pd.read_csv(trade_csv, parse_dates=["entry_date", "exit_date"])
    print(f"  Total trades: {len(df)}")
    print(f"  Win rate    : {(df['pnl_pct']>0).mean()*100:.2f}%")
    print(f"  Avg PnL     : {df['pnl_pct'].mean():+.3f}%")
    print(f"  Total PnL   : {df['pnl_pct'].sum():+.1f}%")
    print(f"  Profit factor: {df.loc[df['pnl_pct']>0,'pnl_pct'].sum() / -df.loc[df['pnl_pct']<0,'pnl_pct'].sum():.2f}")
    print(f"  Avg hold    : {df['holding_days'].mean():.2f}d")
    print(f"  Max loss    : {df['pnl_pct'].min():+.2f}%")
    print(f"  Loss trades : {(df['pnl_pct']<0).sum()} ({(df['pnl_pct']<0).mean()*100:.1f}%)")

    # cache prices
    cache = {}
    for sym in df["symbol"].unique():
        p = load_prices(sym)
        if p is not None: cache[sym] = p

    rows = []
    for _, r in df.iterrows():
        sym = r["symbol"]
        if sym not in cache: continue
        p = cache[sym]
        ei = nearest_idx(p, r["entry_date"])
        xi = nearest_idx(p, r["exit_date"])
        entry_close = p.iloc[ei]["close"]
        exit_close = p.iloc[xi]["close"]

        # entry timing: had we bought 5/10/20 days earlier, what price would we have?
        e5  = p.iloc[max(ei-5, 0)]["close"]
        e10 = p.iloc[max(ei-10,0)]["close"]
        e20 = p.iloc[max(ei-20,0)]["close"]
        # window high/low last 20d
        win = p.iloc[max(ei-20,0):ei+1]
        max_hi_20 = win["high"].max() if len(win) else entry_close
        min_lo_20 = win["low"].min() if len(win) else entry_close
        # distance from local high (proxy for buying near peak)
        dist_from_high20 = (entry_close - max_hi_20) / max_hi_20 * 100  # negative means below high
        # rally before entry
        rally_5d  = (entry_close - e5)  / e5  * 100
        rally_10d = (entry_close - e10) / e10 * 100
        rally_20d = (entry_close - e20) / e20 * 100

        # exit timing: had we sold 5/10 days earlier OR 5/10 later
        x_minus5  = p.iloc[max(xi-5, 0)]["close"]
        x_minus10 = p.iloc[max(xi-10,0)]["close"]
        x_plus5   = p.iloc[min(xi+5, len(p)-1)]["close"]
        x_plus10  = p.iloc[min(xi+10,len(p)-1)]["close"]
        x_plus20  = p.iloc[min(xi+20,len(p)-1)]["close"]
        # forward window high after exit
        fwd = p.iloc[xi+1:min(xi+21,len(p))]
        fwd_max = fwd["high"].max() if len(fwd) else exit_close
        fwd_max_pct = (fwd_max - exit_close) / exit_close * 100  # >0 means we sold too early

        rows.append({
            "symbol": sym, "entry_date": r["entry_date"], "exit_date": r["exit_date"],
            "pnl_pct": r["pnl_pct"], "holding_days": r["holding_days"], "exit_reason": r.get("exit_reason",""),
            "entry_close": entry_close, "exit_close": exit_close,
            "rally_5d": rally_5d, "rally_10d": rally_10d, "rally_20d": rally_20d,
            "dist_from_high20": dist_from_high20,
            "buy_5d_earlier_pct": (entry_close - e5)/e5*100,
            "buy_10d_earlier_pct": (entry_close - e10)/e10*100,
            "buy_20d_earlier_pct": (entry_close - e20)/e20*100,
            "sell_5d_earlier_pct": (x_minus5 - exit_close)/exit_close*100,   # >0 means earlier price higher
            "sell_10d_earlier_pct": (x_minus10 - exit_close)/exit_close*100,
            "sell_5d_later_pct": (x_plus5 - exit_close)/exit_close*100,      # >0 means later price higher
            "sell_10d_later_pct": (x_plus10- exit_close)/exit_close*100,
            "sell_20d_later_pct": (x_plus20- exit_close)/exit_close*100,
            "fwd20_max_pct": fwd_max_pct,
        })
    a = pd.DataFrame(rows)

    # ============= ENTRY TIMING =============
    losers = a[a["pnl_pct"] < 0].copy()
    winners = a[a["pnl_pct"] > 0].copy()
    print(f"\n  ── ENTRY TIMING (loser cohort, n={len(losers)}) ──")
    print(f"    Mean rally_5d  : {losers['rally_5d'].mean():+.2f}%   (winners: {winners['rally_5d'].mean():+.2f}%)")
    print(f"    Mean rally_10d : {losers['rally_10d'].mean():+.2f}%  (winners: {winners['rally_10d'].mean():+.2f}%)")
    print(f"    Mean rally_20d : {losers['rally_20d'].mean():+.2f}%  (winners: {winners['rally_20d'].mean():+.2f}%)")
    print(f"    Mean dist_from_20d_high: {losers['dist_from_high20'].mean():+.2f}%  (winners: {winners['dist_from_high20'].mean():+.2f}%)")

    # near-peak entries: bought within 2% of 20d high AND rally_10d > 8%
    near_peak = losers[(losers["dist_from_high20"] >= -2.0) & (losers["rally_10d"] > 8)]
    rallied_strong = losers[losers["rally_10d"] > 12]
    bought_after_drop = losers[losers["rally_10d"] < -5]
    print(f"\n    Loser BUYS NEAR PEAK (≤2% of 20d high & rally10d>8%): {len(near_peak)} ({len(near_peak)/max(len(losers),1)*100:.1f}% of losers), "
          f"avg PnL {near_peak['pnl_pct'].mean():+.2f}%")
    print(f"    Loser BUYS AFTER STRONG RALLY (rally_10d>12%): {len(rallied_strong)} ({len(rallied_strong)/max(len(losers),1)*100:.1f}%), "
          f"avg PnL {rallied_strong['pnl_pct'].mean():+.2f}%")
    # if had bought 5d earlier, were prices lower? buy_5d_earlier_pct >0 means current entry MORE expensive than 5d ago
    losers_better_5d  = losers[losers["buy_5d_earlier_pct"] > 3]
    losers_better_10d = losers[losers["buy_10d_earlier_pct"] > 5]
    losers_better_20d = losers[losers["buy_20d_earlier_pct"] > 8]
    print(f"\n    Losers where 5d-earlier price was ≥3% LOWER  : {len(losers_better_5d)}/{len(losers)} ({len(losers_better_5d)/max(len(losers),1)*100:.1f}%) "
          f"avg cheaper by {losers_better_5d['buy_5d_earlier_pct'].mean():.2f}%")
    print(f"    Losers where 10d-earlier price was ≥5% LOWER : {len(losers_better_10d)}/{len(losers)} ({len(losers_better_10d)/max(len(losers),1)*100:.1f}%) "
          f"avg cheaper by {losers_better_10d['buy_10d_earlier_pct'].mean():.2f}%")
    print(f"    Losers where 20d-earlier price was ≥8% LOWER : {len(losers_better_20d)}/{len(losers)} ({len(losers_better_20d)/max(len(losers),1)*100:.1f}%) "
          f"avg cheaper by {losers_better_20d['buy_20d_earlier_pct'].mean():.2f}%")

    # ============= EXIT TIMING =============
    print(f"\n  ── EXIT TIMING (all trades, n={len(a)}) ──")
    sold_too_early = a[a["fwd20_max_pct"] > 5]
    sold_way_too_early = a[a["fwd20_max_pct"] > 10]
    print(f"    Trades where price went +5% within 20d AFTER exit (sold too early): "
          f"{len(sold_too_early)}/{len(a)} ({len(sold_too_early)/len(a)*100:.1f}%), avg lost upside {sold_too_early['fwd20_max_pct'].mean():.2f}%")
    print(f"    Trades where price went +10% within 20d AFTER exit             : "
          f"{len(sold_way_too_early)}/{len(a)} ({len(sold_way_too_early)/len(a)*100:.1f}%), avg lost upside {sold_way_too_early['fwd20_max_pct'].mean():.2f}%")
    print(f"\n    Mean sell_5d_later  : {a['sell_5d_later_pct'].mean():+.2f}%  (positive = price kept rising after exit)")
    print(f"    Mean sell_10d_later : {a['sell_10d_later_pct'].mean():+.2f}%")
    print(f"    Mean sell_20d_later : {a['sell_20d_later_pct'].mean():+.2f}%")
    print(f"    Mean sell_5d_earlier: {a['sell_5d_earlier_pct'].mean():+.2f}%  (positive = SHOULD have sold earlier)")
    print(f"    Mean sell_10d_earlier:{a['sell_10d_earlier_pct'].mean():+.2f}%")

    # break down by exit reason
    if "exit_reason" in a.columns:
        print(f"\n  ── EXIT REASON BREAKDOWN ──")
        g = a.groupby("exit_reason").agg(
            n=("pnl_pct","size"),
            avg_pnl=("pnl_pct","mean"),
            wr=("pnl_pct", lambda x: (x>0).mean()*100),
            sold_too_early_5plus=("fwd20_max_pct", lambda x: (x>5).mean()*100),
            avg_lost_upside=("fwd20_max_pct","mean"),
            avg_should_sell_earlier_5d=("sell_5d_earlier_pct","mean"),
        ).sort_values("n", ascending=False)
        print(g.round(2).to_string())

    # near-peak entry losers, deepest losers
    print(f"\n  ── TOP-15 WORST LOSERS (entry context) ──")
    worst = losers.nsmallest(15, "pnl_pct")[["symbol","entry_date","pnl_pct","holding_days",
                                              "rally_10d","dist_from_high20",
                                              "buy_10d_earlier_pct","exit_reason"]]
    print(worst.to_string(index=False))

    # winners that sold too early (biggest missed upside)
    print(f"\n  ── TOP-15 'SOLD TOO EARLY' TRADES (biggest forward upside missed) ──")
    early = a.nlargest(15, "fwd20_max_pct")[["symbol","entry_date","exit_date","pnl_pct","holding_days",
                                              "exit_reason","fwd20_max_pct","sell_10d_later_pct"]]
    print(early.to_string(index=False))

    return a


if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "results")
    a29 = analyze(os.path.join(out_dir, "trades_v29.csv"), "V29")
    a28 = analyze(os.path.join(out_dir, "trades_v28.csv"), "V28")
    a29.to_csv(os.path.join(out_dir, "v29_timing_audit.csv"), index=False)
    a28.to_csv(os.path.join(out_dir, "v28_timing_audit.csv"), index=False)

    print("\n\n" + "="*100)
    print("  V29 vs V28 KEY DELTAS")
    print("="*100)
    for col in ["rally_5d","rally_10d","rally_20d","dist_from_high20",
                "fwd20_max_pct","sell_5d_later_pct","sell_10d_later_pct","sell_20d_later_pct",
                "sell_5d_earlier_pct","sell_10d_earlier_pct"]:
        print(f"    {col:<26} V29={a29[col].mean():+.2f}%  V28={a28[col].mean():+.2f}%   Δ={a29[col].mean()-a28[col].mean():+.2f}")
