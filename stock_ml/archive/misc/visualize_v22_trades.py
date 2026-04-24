"""
V22 Trade Visualization Dashboard
===================================
Interactive matplotlib charts showing:
1. Equity curves: V19.1 vs V19.3 vs V22 vs Rule
2. Per-symbol PnL comparison bar chart
3. Trade scatter: PnL vs holding days
4. Exit reason breakdown
5. Per-symbol price chart with trade markers (selectable)
6. Monthly/yearly performance heatmap
7. Drawdown chart
8. Win/Loss distribution
"""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from src.data.splitter import WalkForwardSplitter
from src.data.target import TargetGenerator
from src.features.engine import FeatureEngine
from src.models.registry import build_model
from run_v19_1_compare import run_test as run_test_base, run_rule_test, calc_metrics
from run_v19_1_compare import backtest_v19_1
from run_v19_3_compare import backtest_v19_3
from run_v22_final import backtest_v22

SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"


def run_with_equity(backtest_fn, symbols_str=SYMBOLS):
    """Run backtest and return trades + per-symbol equity curves."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    config = {
        "data": {"data_dir": data_dir},
        "split": {"method": "walk_forward", "train_years": 4, "test_years": 1,
                  "gap_days": 0, "first_test_year": 2020, "last_test_year": 2025},
        "target": {"type": "trend_regime", "trend_method": "dual_ma",
                   "short_window": 5, "long_window": 20, "classes": 3},
    }
    pick = [s.strip() for s in symbols_str.split(",")]
    loader = DataLoader(data_dir)
    splitter = WalkForwardSplitter.from_config(config)
    target_gen = TargetGenerator.from_config(config)
    raw_df = loader.load_all(symbols=pick)
    engine = FeatureEngine(feature_set="leading")
    df = engine.compute_for_all_symbols(raw_df)
    df = target_gen.generate_for_all_symbols(df)
    feature_cols = engine.get_feature_columns(df)
    df = df.dropna(subset=feature_cols + ["target"])

    all_trades = []
    equity_data = {}

    for window, train_df, test_df in splitter.split(df):
        model = build_model("lightgbm")
        X_train = np.nan_to_num(train_df[feature_cols].values)
        y_train = train_df["target"].values.astype(int)
        model.fit(X_train, y_train)

        for sym in test_df["symbol"].unique():
            if sym not in pick:
                continue
            sym_test = test_df[test_df["symbol"] == sym].reset_index(drop=True)
            if len(sym_test) < 10:
                continue
            X_sym = np.nan_to_num(sym_test[feature_cols].values)
            y_pred = model.predict(X_sym)
            rets = sym_test["return_1d"].values

            r = backtest_fn(y_pred, rets, sym_test, feature_cols,
                            mod_a=True, mod_b=True, mod_c=False, mod_d=False,
                            mod_e=True, mod_f=True, mod_g=True, mod_h=True,
                            mod_i=True, mod_j=True)

            date_col = "date" if "date" in sym_test.columns else "timestamp"
            dates = pd.to_datetime(sym_test[date_col].values)
            closes = sym_test["close"].values

            for t in r["trades"]:
                t["symbol"] = sym
                entry_idx = t["entry_day"]
                exit_idx = t["exit_day"]
                if entry_idx < len(dates):
                    t["entry_date_dt"] = dates[entry_idx]
                if exit_idx < len(dates):
                    t["exit_date_dt"] = dates[exit_idx]
                if entry_idx < len(closes):
                    t["entry_price"] = closes[entry_idx]
                if exit_idx < len(closes):
                    t["exit_price"] = closes[exit_idx]
            all_trades.extend(r["trades"])

            key = (sym, window)
            equity_data[key] = {
                "dates": dates,
                "closes": closes,
                "equity": r["equity_curve"],
            }

    return all_trades, equity_data


def make_v22_fn():
    def bt_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v22(y_pred, returns, df_test, feature_cols,
                                  fast_exit_threshold_std=-0.06, **kwargs)
    return bt_fn


def plot_dashboard(trades_v191, trades_v193, trades_v22, trades_rule):
    """Main dashboard with 8 panels."""
    df_v191 = pd.DataFrame(trades_v191)
    df_v193 = pd.DataFrame(trades_v193)
    df_v22 = pd.DataFrame(trades_v22)
    df_rule = pd.DataFrame(trades_rule)

    for df in [df_v191, df_v193, df_v22, df_rule]:
        if "entry_date" in df.columns:
            df["entry_dt"] = pd.to_datetime(df["entry_date"].astype(str).str[:10], errors="coerce")
            df["exit_dt"] = pd.to_datetime(df["exit_date"].astype(str).str[:10], errors="coerce")
            df["year"] = df["entry_dt"].dt.year

    fig = plt.figure(figsize=(24, 18), facecolor="#1a1a2e")
    fig.suptitle("V22 TRADING DASHBOARD — Full Backtest Analysis (2020-2025)",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.97, top=0.94, bottom=0.04)

    colors = {"V19.1": "#4fc3f7", "V19.3": "#ffb74d", "V22": "#66bb6a", "Rule": "#ef5350"}
    bg_color = "#16213e"
    text_color = "#e0e0e0"
    grid_color = "#2a3a5c"

    def style_ax(ax, title=""):
        ax.set_facecolor(bg_color)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(grid_color)
        ax.spines["left"].set_color(grid_color)
        ax.grid(True, alpha=0.15, color=grid_color)

    # ═══ 1. Cumulative PnL over time ═══
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Cumulative PnL Over Time")

    for label, df, color in [("V19.1", df_v191, colors["V19.1"]),
                              ("V19.3", df_v193, colors["V19.3"]),
                              ("V22", df_v22, colors["V22"]),
                              ("Rule", df_rule, colors["Rule"])]:
        if "exit_dt" not in df.columns or df["exit_dt"].isna().all():
            continue
        sorted_df = df.dropna(subset=["exit_dt"]).sort_values("exit_dt")
        cum_pnl = sorted_df["pnl_pct"].cumsum()
        ax1.plot(sorted_df["exit_dt"], cum_pnl, label=label, color=color, linewidth=1.8, alpha=0.9)

    ax1.legend(fontsize=9, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax1.set_ylabel("Cumulative PnL (%)", color=text_color, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ═══ 2. Per-Symbol PnL Bar Chart ═══
    ax2 = fig.add_subplot(gs[0, 2:])
    style_ax(ax2, "Per-Symbol Total PnL Comparison")

    syms = sorted(set(df_v22["symbol"].unique()))
    x = np.arange(len(syms))
    w = 0.2

    for offset, (label, df, color) in enumerate([
        ("V19.1", df_v191, colors["V19.1"]),
        ("V19.3", df_v193, colors["V19.3"]),
        ("V22", df_v22, colors["V22"]),
        ("Rule", df_rule, colors["Rule"]),
    ]):
        vals = []
        for sym in syms:
            st = df[df["symbol"] == sym]
            vals.append(st["pnl_pct"].sum() if len(st) > 0 else 0)
        ax2.bar(x + (offset - 1.5) * w, vals, w, label=label, color=color, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels(syms, fontsize=8, color=text_color, rotation=45)
    ax2.legend(fontsize=8, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax2.set_ylabel("Total PnL (%)", color=text_color, fontsize=9)
    ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)

    # ═══ 3. V22 Trade Scatter: PnL vs Hold Days ═══
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3, "V22 Trades: PnL vs Holding Days (color=trend)")

    trend_colors = {"strong": "#66bb6a", "moderate": "#ffb74d", "weak": "#ef5350"}
    if "entry_trend" in df_v22.columns:
        for trend, color in trend_colors.items():
            mask = df_v22["entry_trend"] == trend
            ax3.scatter(df_v22.loc[mask, "holding_days"],
                       df_v22.loc[mask, "pnl_pct"],
                       c=color, alpha=0.6, s=25, label=trend, edgecolors="white", linewidth=0.3)
    ax3.axhline(y=0, color="white", linewidth=0.8, alpha=0.4)
    ax3.set_xlabel("Holding Days", color=text_color, fontsize=9)
    ax3.set_ylabel("PnL (%)", color=text_color, fontsize=9)
    ax3.legend(fontsize=8, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)

    # ═══ 4. Exit Reason Breakdown ═══
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, "V22 Exit Reason — Total PnL")

    if "exit_reason" in df_v22.columns:
        reason_pnl = df_v22.groupby("exit_reason")["pnl_pct"].sum().sort_values()
        bar_colors = ["#ef5350" if v < 0 else "#66bb6a" for v in reason_pnl.values]
        ax4.barh(reason_pnl.index, reason_pnl.values, color=bar_colors, alpha=0.85)
        ax4.set_xlabel("Total PnL (%)", color=text_color, fontsize=9)
        for i, (idx, val) in enumerate(reason_pnl.items()):
            ax4.text(val + (5 if val >= 0 else -5), i, f"{val:+.0f}%",
                    va="center", ha="left" if val >= 0 else "right",
                    color=text_color, fontsize=7)

    # ═══ 5. Exit Reason Count ═══
    ax5 = fig.add_subplot(gs[1, 3])
    style_ax(ax5, "V22 Exit Reason — Count & WR")

    if "exit_reason" in df_v22.columns:
        reason_stats = []
        for reason, grp in df_v22.groupby("exit_reason"):
            wins = len(grp[grp["pnl_pct"] > 0])
            wr = wins / len(grp) * 100 if len(grp) > 0 else 0
            reason_stats.append((reason, len(grp), wr))
        reason_stats.sort(key=lambda x: x[1], reverse=True)
        reasons = [r[0] for r in reason_stats]
        counts = [r[1] for r in reason_stats]
        wrs = [r[2] for r in reason_stats]

        y_pos = np.arange(len(reasons))
        ax5.barh(y_pos, counts, color="#4fc3f7", alpha=0.7)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(reasons, fontsize=8)
        ax5.set_xlabel("Count", color=text_color, fontsize=9)
        for i, (cnt, wr) in enumerate(zip(counts, wrs)):
            ax5.text(cnt + 2, i, f"WR={wr:.0f}%", va="center", color=text_color, fontsize=7)

    # ═══ 6. Win/Loss Distribution ═══
    ax6 = fig.add_subplot(gs[2, 0])
    style_ax(ax6, "V22 PnL Distribution")

    pnl_vals = df_v22["pnl_pct"].values
    wins = pnl_vals[pnl_vals > 0]
    losses = pnl_vals[pnl_vals <= 0]
    bins = np.arange(-25, 50, 2)
    ax6.hist(wins, bins=bins, color="#66bb6a", alpha=0.7, label=f"Wins ({len(wins)})")
    ax6.hist(losses, bins=bins, color="#ef5350", alpha=0.7, label=f"Losses ({len(losses)})")
    ax6.axvline(x=0, color="white", linewidth=1, alpha=0.5)
    ax6.axvline(x=np.mean(pnl_vals), color="#ffb74d", linewidth=1.5, linestyle="--",
               label=f"Mean={np.mean(pnl_vals):+.1f}%")
    ax6.set_xlabel("PnL (%)", color=text_color, fontsize=9)
    ax6.set_ylabel("Count", color=text_color, fontsize=9)
    ax6.legend(fontsize=8, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)

    # ═══ 7. Drawdown from peak ═══
    ax7 = fig.add_subplot(gs[2, 1])
    style_ax(ax7, "Cumulative PnL Drawdown")

    for label, df, color in [("V19.1", df_v191, colors["V19.1"]),
                              ("V22", df_v22, colors["V22"]),
                              ("Rule", df_rule, colors["Rule"])]:
        if "exit_dt" not in df.columns or df["exit_dt"].isna().all():
            continue
        sorted_df = df.dropna(subset=["exit_dt"]).sort_values("exit_dt")
        cum_pnl = sorted_df["pnl_pct"].cumsum().values
        peak = np.maximum.accumulate(cum_pnl)
        dd = cum_pnl - peak
        ax7.fill_between(sorted_df["exit_dt"], dd, 0, alpha=0.3, color=color, label=label)
        ax7.plot(sorted_df["exit_dt"], dd, color=color, linewidth=1, alpha=0.8)

    ax7.set_ylabel("Drawdown (%)", color=text_color, fontsize=9)
    ax7.legend(fontsize=8, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # ═══ 8. Yearly Performance Comparison ═══
    ax8 = fig.add_subplot(gs[2, 2:])
    style_ax(ax8, "Yearly PnL Comparison")

    years = sorted(df_v22["year"].dropna().unique())
    x = np.arange(len(years))
    w = 0.2
    for offset, (label, df, color) in enumerate([
        ("V19.1", df_v191, colors["V19.1"]),
        ("V19.3", df_v193, colors["V19.3"]),
        ("V22", df_v22, colors["V22"]),
        ("Rule", df_rule, colors["Rule"]),
    ]):
        vals = []
        for yr in years:
            yt = df[df["year"] == yr] if "year" in df.columns else pd.DataFrame()
            vals.append(yt["pnl_pct"].sum() if len(yt) > 0 else 0)
        ax8.bar(x + (offset - 1.5) * w, vals, w, label=label, color=color, alpha=0.85)
    ax8.set_xticks(x)
    ax8.set_xticklabels([str(int(y)) for y in years], fontsize=9, color=text_color)
    ax8.legend(fontsize=8, facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)
    ax8.set_ylabel("Total PnL (%)", color=text_color, fontsize=9)
    ax8.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)

    # ═══ 9. Position Size vs PnL ═══
    ax9 = fig.add_subplot(gs[3, 0])
    style_ax(ax9, "V22: Position Size vs PnL")

    if "position_size" in df_v22.columns:
        c = ["#66bb6a" if p > 0 else "#ef5350" for p in df_v22["pnl_pct"]]
        ax9.scatter(df_v22["position_size"], df_v22["pnl_pct"],
                   c=c, alpha=0.5, s=20, edgecolors="white", linewidth=0.2)
        ax9.axhline(y=0, color="white", linewidth=0.8, alpha=0.4)
        ax9.set_xlabel("Position Size", color=text_color, fontsize=9)
        ax9.set_ylabel("PnL (%)", color=text_color, fontsize=9)

    # ═══ 10. V22 Monthly PnL Heatmap ═══
    ax10 = fig.add_subplot(gs[3, 1:3])
    style_ax(ax10, "V22 Monthly PnL Heatmap")

    if "entry_dt" in df_v22.columns:
        df_v22c = df_v22.dropna(subset=["entry_dt"]).copy()
        df_v22c["month"] = df_v22c["entry_dt"].dt.month
        df_v22c["year_m"] = df_v22c["entry_dt"].dt.year
        pivot = df_v22c.pivot_table(values="pnl_pct", index="year_m", columns="month",
                                     aggfunc="sum", fill_value=0)
        im = ax10.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                         vmin=-50, vmax=100)
        ax10.set_yticks(range(len(pivot.index)))
        ax10.set_yticklabels([str(int(y)) for y in pivot.index], fontsize=8)
        ax10.set_xticks(range(len(pivot.columns)))
        ax10.set_xticklabels([f"M{int(m)}" for m in pivot.columns], fontsize=8)
        for yi in range(len(pivot.index)):
            for xi in range(len(pivot.columns)):
                val = pivot.values[yi, xi]
                if val != 0:
                    ax10.text(xi, yi, f"{val:+.0f}", ha="center", va="center",
                             fontsize=6, color="black" if abs(val) < 30 else "white",
                             fontweight="bold")
        plt.colorbar(im, ax=ax10, shrink=0.8, label="PnL %")

    # ═══ 11. Summary Stats Table ═══
    ax11 = fig.add_subplot(gs[3, 3])
    ax11.set_facecolor(bg_color)
    ax11.axis("off")
    ax11.set_title("Summary Metrics", color="white", fontsize=11, fontweight="bold", pad=8)

    metrics = []
    for label, df in [("V19.1", df_v191), ("V19.3", df_v193), ("V22", df_v22), ("Rule", df_rule)]:
        m = calc_metrics(df.to_dict("records"))
        metrics.append([label, f"{m['trades']}", f"{m['wr']:.1f}%",
                       f"{m['avg_pnl']:+.2f}%", f"{m['total_pnl']:+.0f}%",
                       f"{m['pf']:.2f}", f"{m['max_loss']:+.1f}%"])

    col_labels = ["Model", "#", "WR", "AvgPnL", "TotPnL", "PF", "MaxLoss"]
    table = ax11.table(cellText=metrics, colLabels=col_labels,
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for key, cell in table.get_celld().items():
        cell.set_edgecolor(grid_color)
        if key[0] == 0:
            cell.set_facecolor("#1a3a5c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(bg_color)
            cell.set_text_props(color=text_color)
            if key[1] == 0:
                lbl = metrics[key[0] - 1][0]
                cell.set_text_props(color=colors.get(lbl, text_color), fontweight="bold")

    plt.savefig(os.path.join(os.path.dirname(__file__), "results", "v22_dashboard.png"),
                dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("  Dashboard saved to results/v22_dashboard.png")
    plt.close(fig)


def plot_symbol_trades(trades_v22, symbol="VND"):
    """Detailed per-symbol trade chart with price + entry/exit markers."""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    loader = DataLoader(data_dir)
    raw = loader.load_all(symbols=[symbol])
    sym_data = raw[raw["symbol"] == symbol].copy()
    date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
    sym_data[date_col] = pd.to_datetime(sym_data[date_col], utc=True).dt.tz_localize(None)
    sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
    sym_data = sym_data[sym_data[date_col] >= "2020-01-01"]

    df_t = pd.DataFrame(trades_v22)
    sym_trades = df_t[df_t["symbol"] == symbol].copy()

    if "entry_date" in sym_trades.columns:
        sym_trades["entry_dt"] = pd.to_datetime(sym_trades["entry_date"].astype(str).str[:10], errors="coerce")
        sym_trades["exit_dt"] = pd.to_datetime(sym_trades["exit_date"].astype(str).str[:10], errors="coerce")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), facecolor="#1a1a2e",
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    for ax in [ax1, ax2]:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2a3a5c")
        ax.spines["left"].set_color("#2a3a5c")
        ax.grid(True, alpha=0.15, color="#2a3a5c")

    # Price chart
    ax1.set_title(f"V22 Trades — {symbol} (2020-2025)", color="white", fontsize=14, fontweight="bold")
    ax1.plot(sym_data[date_col], sym_data["close"], color="#78909c", linewidth=0.8, alpha=0.9, label="Close")

    # SMA overlays
    sym_data["sma20"] = sym_data["close"].rolling(20).mean()
    sym_data["sma50"] = sym_data["close"].rolling(50).mean()
    ax1.plot(sym_data[date_col], sym_data["sma20"], color="#4fc3f7", linewidth=0.6, alpha=0.5, label="SMA20")
    ax1.plot(sym_data[date_col], sym_data["sma50"], color="#ffb74d", linewidth=0.6, alpha=0.5, label="SMA50")

    # Trade markers
    for _, t in sym_trades.iterrows():
        pnl = t["pnl_pct"]
        entry_dt = t.get("entry_dt", None)
        exit_dt = t.get("exit_dt", None)
        if pd.isna(entry_dt) or pd.isna(exit_dt):
            continue

        # Find price at entry/exit dates
        entry_row = sym_data[sym_data[date_col] >= entry_dt].head(1)
        exit_row = sym_data[sym_data[date_col] >= exit_dt].head(1)
        if len(entry_row) == 0 or len(exit_row) == 0:
            continue

        ep = entry_row["close"].values[0]
        xp = exit_row["close"].values[0]
        ed = entry_row[date_col].values[0]
        xd = exit_row[date_col].values[0]

        color = "#66bb6a" if pnl > 0 else "#ef5350"
        alpha = min(0.8, 0.2 + abs(pnl) / 30)

        # Shade the trade period
        ax1.axvspan(ed, xd, alpha=0.08, color=color)

        # Entry arrow (up triangle)
        ax1.scatter(ed, ep, marker="^", c="#4fc3f7", s=60, zorder=5, edgecolors="white", linewidth=0.5)
        # Exit arrow (down triangle)
        ax1.scatter(xd, xp, marker="v", c=color, s=60, zorder=5, edgecolors="white", linewidth=0.5)

        # PnL label
        mid_date = ed + (xd - ed) / 2
        label_y = max(ep, xp) * 1.02
        ax1.annotate(f"{pnl:+.1f}%", xy=(mid_date, label_y),
                    fontsize=6, color=color, ha="center", fontweight="bold",
                    alpha=alpha)

    ax1.set_ylabel("Price (VND)", color="#e0e0e0", fontsize=10)
    ax1.legend(fontsize=8, facecolor="#16213e", edgecolor="#2a3a5c", labelcolor="#e0e0e0", loc="upper left")

    # PnL bar chart below
    ax2.set_title("Per-Trade PnL", color="white", fontsize=10, fontweight="bold")
    for _, t in sym_trades.iterrows():
        exit_dt = t.get("exit_dt", None)
        if pd.isna(exit_dt):
            continue
        pnl = t["pnl_pct"]
        color = "#66bb6a" if pnl > 0 else "#ef5350"
        ax2.bar(exit_dt, pnl, width=5, color=color, alpha=0.7)

    ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax2.set_ylabel("PnL (%)", color="#e0e0e0", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.tight_layout()
    safe_sym = symbol.replace("/", "_")
    plt.savefig(os.path.join(os.path.dirname(__file__), "results", f"v22_trades_{safe_sym}.png"),
                dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"  Trade chart saved to results/v22_trades_{safe_sym}.png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None,
                        help="Show per-symbol trade chart (e.g. VND, HPG)")
    parser.add_argument("--all-symbols", action="store_true",
                        help="Generate trade charts for all 14 symbols")
    args = parser.parse_args()

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

    print("=" * 80)
    print("V22 VISUALIZATION DASHBOARD")
    print("=" * 80)

    # Run all models
    print("\n  Running V19.1...")
    trades_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                                backtest_fn=backtest_v19_1)
    print("  Running V19.3...")
    trades_v193 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                                backtest_fn=backtest_v19_3)
    print("  Running V22...")
    trades_v22 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                               backtest_fn=make_v22_fn())
    print("  Running Rule...")
    trades_rule = run_rule_test(SYMBOLS)

    # Main dashboard
    print("\n  Generating dashboard...")
    plot_dashboard(trades_v191, trades_v193, trades_v22, trades_rule)

    # Per-symbol charts
    if args.symbol:
        print(f"\n  Generating {args.symbol} trade chart...")
        plot_symbol_trades(trades_v22, args.symbol)
    elif args.all_symbols:
        for sym in SYMBOLS.split(","):
            sym = sym.strip()
            print(f"\n  Generating {sym} trade chart...")
            plot_symbol_trades(trades_v22, sym)
    else:
        # Default: show top 4 most interesting symbols
        for sym in ["VND", "TCB", "HPG", "AAV"]:
            print(f"\n  Generating {sym} trade chart...")
            plot_symbol_trades(trades_v22, sym)

    print("\n" + "=" * 80)
    print("DONE - All charts saved to stock_ml/results/")
    print("=" * 80)
