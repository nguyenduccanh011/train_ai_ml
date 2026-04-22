"""
V23 Trade Visualization — Multi-model comparison on price charts.

For each symbol: price chart with BUY/SELL markers from V19.1, V19.3, V22, V23, Rule
overlaid so you can visually compare entry/exit quality across models.

Also generates a summary dashboard comparing all models.
"""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import DataLoader
from run_v19_1_compare import run_test as run_test_base, run_rule_test, calc_metrics
from run_v19_1_compare import backtest_v19_1
from run_v19_3_compare import backtest_v19_3
from run_v22_final import backtest_v22_final
from run_v23_optimal import backtest_v23

SYMBOLS = "ACB,FPT,HPG,SSI,VND,MBB,TCB,VNM,DGC,AAS,AAV,REE,BID,VIC"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

MODEL_COLORS = {
    "V19.1": "#4fc3f7",
    "V19.3": "#ffb74d",
    "V22":   "#ab47bc",
    "V23":   "#66bb6a",
    "Rule":  "#ef5350",
}

BG_COLOR = "#16213e"
FIG_BG = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a3a5c"


def style_ax(ax, title=""):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(GRID_COLOR)
    ax.grid(True, alpha=0.15, color=GRID_COLOR)


def load_price_data(symbol):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "portable_data", "vn_stock_ai_dataset_cleaned")
    loader = DataLoader(data_dir)
    raw = loader.load_all(symbols=[symbol])
    sym_data = raw[raw["symbol"] == symbol].copy()
    date_col = "timestamp" if "timestamp" in sym_data.columns else "date"
    sym_data[date_col] = pd.to_datetime(sym_data[date_col], utc=True).dt.tz_localize(None)
    sym_data = sym_data.sort_values(date_col).reset_index(drop=True)
    sym_data = sym_data[sym_data[date_col] >= "2020-01-01"].copy()
    sym_data["sma20"] = sym_data["close"].rolling(20).mean()
    sym_data["sma50"] = sym_data["close"].rolling(50).mean()
    return sym_data, date_col


def prepare_trades(all_trades, symbol):
    df = pd.DataFrame(all_trades)
    st = df[df["symbol"] == symbol].copy()
    if "entry_date" in st.columns:
        st["entry_dt"] = pd.to_datetime(st["entry_date"].astype(str).str[:10], errors="coerce")
        st["exit_dt"] = pd.to_datetime(st["exit_date"].astype(str).str[:10], errors="coerce")
    return st


def plot_multi_model_symbol(sym_data, date_col, trades_dict, symbol, save_path):
    """
    Price chart with BUY/SELL markers for each model in different colors.
    Below: per-model PnL bars side by side.
    """
    n_models = len(trades_dict)
    fig = plt.figure(figsize=(24, 14), facecolor=FIG_BG)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 1.2, 1.2], hspace=0.12,
                  left=0.04, right=0.97, top=0.93, bottom=0.04)

    # --- Panel 1: Price + trade markers ---
    ax_price = fig.add_subplot(gs[0])
    style_ax(ax_price, f"{symbol} — Multi-Model Trade Comparison (2020-2025)")

    ax_price.plot(sym_data[date_col], sym_data["close"],
                  color="#78909c", linewidth=0.9, alpha=0.9, label="Close", zorder=1)
    ax_price.plot(sym_data[date_col], sym_data["sma20"],
                  color="#4fc3f7", linewidth=0.5, alpha=0.35, label="SMA20")
    ax_price.plot(sym_data[date_col], sym_data["sma50"],
                  color="#ffb74d", linewidth=0.5, alpha=0.35, label="SMA50")

    y_offsets = {"V19.1": -0.04, "V19.3": -0.02, "V22": 0.00, "V23": 0.02, "Rule": 0.04}
    marker_sizes = {"V19.1": 40, "V19.3": 40, "V22": 45, "V23": 70, "Rule": 35}

    for model_name, sym_trades in trades_dict.items():
        color = MODEL_COLORS[model_name]
        y_off = y_offsets.get(model_name, 0)
        ms = marker_sizes.get(model_name, 50)

        for _, t in sym_trades.iterrows():
            entry_dt = t.get("entry_dt")
            exit_dt = t.get("exit_dt")
            pnl = t["pnl_pct"]
            if pd.isna(entry_dt) or pd.isna(exit_dt):
                continue

            entry_row = sym_data[sym_data[date_col] >= entry_dt].head(1)
            exit_row = sym_data[sym_data[date_col] >= exit_dt].head(1)
            if len(entry_row) == 0 or len(exit_row) == 0:
                continue

            ep = entry_row["close"].values[0] * (1 + y_off)
            xp = exit_row["close"].values[0] * (1 + y_off)
            ed = entry_row[date_col].values[0]
            xd = exit_row[date_col].values[0]

            if model_name == "V23":
                ax_price.axvspan(ed, xd, alpha=0.06,
                                 color="#66bb6a" if pnl > 0 else "#ef5350", zorder=0)

            ax_price.scatter(ed, ep, marker="^", c=color, s=ms, zorder=5,
                             edgecolors="white", linewidth=0.4, alpha=0.85)
            ax_price.scatter(xd, xp, marker="v", c=color, s=ms, zorder=5,
                             edgecolors="white", linewidth=0.4, alpha=0.85)

            if model_name == "V23":
                mid_date = ed + (xd - ed) / 2
                label_y = max(entry_row["close"].values[0], exit_row["close"].values[0]) * 1.04
                ax_price.annotate(f"{pnl:+.1f}%", xy=(mid_date, label_y),
                                  fontsize=6, color=color, ha="center", fontweight="bold",
                                  alpha=0.9, zorder=6)

    legend_handles = []
    for mn, mc in MODEL_COLORS.items():
        if mn in trades_dict:
            legend_handles.append(plt.scatter([], [], marker="^", c=mc, s=40, label=f"{mn} entry"))
            legend_handles.append(plt.scatter([], [], marker="v", c=mc, s=40, label=f"{mn} exit"))
    ax_price.legend(handles=legend_handles, fontsize=7, facecolor=BG_COLOR,
                    edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc="upper left", ncol=5)
    ax_price.set_ylabel("Price", color=TEXT_COLOR, fontsize=10)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # --- Panel 2: Per-trade PnL bars for each model ---
    ax_pnl = fig.add_subplot(gs[1], sharex=ax_price)
    style_ax(ax_pnl, "Per-Trade PnL by Model")

    bar_width_days = {2: 2, 3: 1.5, 4: 1.2, 5: 1}
    bw = bar_width_days.get(n_models, 1.5)
    offsets_d = np.linspace(-bw * n_models / 2, bw * n_models / 2, n_models)

    for idx, (model_name, sym_trades) in enumerate(trades_dict.items()):
        color = MODEL_COLORS[model_name]
        for _, t in sym_trades.iterrows():
            exit_dt = t.get("exit_dt")
            if pd.isna(exit_dt):
                continue
            pnl = t["pnl_pct"]
            dt_offset = pd.Timedelta(days=offsets_d[idx])
            bar_color = color if pnl > 0 else "#555555"
            ax_pnl.bar(exit_dt + dt_offset, pnl, width=bw, color=bar_color, alpha=0.7,
                       edgecolor=color, linewidth=0.3)

    ax_pnl.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax_pnl.set_ylabel("PnL (%)", color=TEXT_COLOR, fontsize=9)

    # --- Panel 3: Cumulative PnL per model ---
    ax_cum = fig.add_subplot(gs[2], sharex=ax_price)
    style_ax(ax_cum, "Cumulative PnL per Model")

    for model_name, sym_trades in trades_dict.items():
        color = MODEL_COLORS[model_name]
        if len(sym_trades) == 0:
            continue
        sorted_t = sym_trades.dropna(subset=["exit_dt"]).sort_values("exit_dt")
        if len(sorted_t) == 0:
            continue
        cum = sorted_t["pnl_pct"].cumsum()
        lw = 2.5 if model_name == "V23" else 1.2
        ax_cum.plot(sorted_t["exit_dt"], cum, color=color, linewidth=lw,
                    alpha=0.9, label=model_name)

    ax_cum.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)
    ax_cum.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, loc="upper left")
    ax_cum.set_ylabel("Cum PnL (%)", color=TEXT_COLOR, fontsize=9)
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(all_model_trades):
    """Summary dashboard: all models compared."""
    fig = plt.figure(figsize=(26, 20), facecolor=FIG_BG)
    fig.suptitle("V23 OPTIMAL — Full Backtest Dashboard (2020-2025)",
                 fontsize=18, fontweight="bold", color="white", y=0.98)
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.97, top=0.94, bottom=0.04)

    dfs = {}
    for name, trades in all_model_trades.items():
        df = pd.DataFrame(trades)
        if "entry_date" in df.columns:
            df["entry_dt"] = pd.to_datetime(df["entry_date"].astype(str).str[:10], errors="coerce")
            df["exit_dt"] = pd.to_datetime(df["exit_date"].astype(str).str[:10], errors="coerce")
            df["year"] = df["entry_dt"].dt.year
        dfs[name] = df

    # 1. Cumulative PnL
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, "Cumulative PnL Over Time")
    for name, df in dfs.items():
        if "exit_dt" not in df.columns:
            continue
        s = df.dropna(subset=["exit_dt"]).sort_values("exit_dt")
        lw = 2.5 if name == "V23" else 1.3
        ax1.plot(s["exit_dt"], s["pnl_pct"].cumsum(), label=name,
                 color=MODEL_COLORS.get(name, "#aaa"), linewidth=lw, alpha=0.9)
    ax1.legend(fontsize=9, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax1.set_ylabel("Cumulative PnL (%)", color=TEXT_COLOR, fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 2. Per-symbol bar
    ax2 = fig.add_subplot(gs[0, 2:])
    style_ax(ax2, "Per-Symbol Total PnL")
    syms = sorted(set().union(*(df["symbol"].unique() for df in dfs.values() if "symbol" in df.columns)))
    x = np.arange(len(syms))
    w = 0.15
    for idx, (name, df) in enumerate(dfs.items()):
        vals = [df[df["symbol"] == s]["pnl_pct"].sum() if s in df["symbol"].values else 0 for s in syms]
        ax2.bar(x + (idx - 2) * w, vals, w, label=name,
                color=MODEL_COLORS.get(name, "#aaa"), alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(syms, fontsize=7, color=TEXT_COLOR, rotation=45)
    ax2.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax2.set_ylabel("Total PnL (%)", color=TEXT_COLOR, fontsize=9)
    ax2.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)

    # 3. V23 scatter
    ax3 = fig.add_subplot(gs[1, :2])
    style_ax(ax3, "V23 Trades: PnL vs Holding Days (color=trend)")
    df23 = dfs.get("V23", pd.DataFrame())
    trend_colors = {"strong": "#66bb6a", "moderate": "#ffb74d", "weak": "#ef5350"}
    if "entry_trend" in df23.columns:
        for trend, tc in trend_colors.items():
            mask = df23["entry_trend"] == trend
            ax3.scatter(df23.loc[mask, "holding_days"], df23.loc[mask, "pnl_pct"],
                        c=tc, alpha=0.6, s=25, label=trend, edgecolors="white", linewidth=0.3)
    ax3.axhline(y=0, color="white", linewidth=0.8, alpha=0.4)
    ax3.set_xlabel("Holding Days", color=TEXT_COLOR, fontsize=9)
    ax3.set_ylabel("PnL (%)", color=TEXT_COLOR, fontsize=9)
    ax3.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # 4. Exit reason PnL comparison
    ax4 = fig.add_subplot(gs[1, 2:])
    style_ax(ax4, "Exit Reason Total PnL — All Models")
    all_reasons = sorted(set().union(*(df["exit_reason"].unique() for df in dfs.values() if "exit_reason" in df.columns)))
    y_pos = np.arange(len(all_reasons))
    h = 0.15
    for idx, (name, df) in enumerate(dfs.items()):
        if "exit_reason" not in df.columns:
            continue
        vals = [df[df["exit_reason"] == r]["pnl_pct"].sum() if r in df["exit_reason"].values else 0 for r in all_reasons]
        ax4.barh(y_pos + (idx - 2) * h, vals, h, label=name,
                 color=MODEL_COLORS.get(name, "#aaa"), alpha=0.85)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(all_reasons, fontsize=7)
    ax4.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax4.set_xlabel("Total PnL (%)", color=TEXT_COLOR, fontsize=9)

    # 5. Drawdown
    ax5 = fig.add_subplot(gs[2, :2])
    style_ax(ax5, "Cumulative PnL Drawdown")
    for name, df in dfs.items():
        if "exit_dt" not in df.columns:
            continue
        s = df.dropna(subset=["exit_dt"]).sort_values("exit_dt")
        cum = s["pnl_pct"].cumsum().values
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        lw = 2.0 if name == "V23" else 1.0
        ax5.fill_between(s["exit_dt"], dd, 0, alpha=0.2, color=MODEL_COLORS.get(name, "#aaa"))
        ax5.plot(s["exit_dt"], dd, color=MODEL_COLORS.get(name, "#aaa"), linewidth=lw,
                 alpha=0.8, label=name)
    ax5.set_ylabel("Drawdown (%)", color=TEXT_COLOR, fontsize=9)
    ax5.legend(fontsize=8, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 6. Yearly bars
    ax6 = fig.add_subplot(gs[2, 2:])
    style_ax(ax6, "Yearly PnL Comparison")
    all_years = sorted(set().union(*(df["year"].dropna().unique() for df in dfs.values() if "year" in df.columns)))
    x = np.arange(len(all_years))
    w = 0.15
    for idx, (name, df) in enumerate(dfs.items()):
        if "year" not in df.columns:
            continue
        vals = [df[df["year"] == yr]["pnl_pct"].sum() for yr in all_years]
        ax6.bar(x + (idx - 2) * w, vals, w, label=name,
                color=MODEL_COLORS.get(name, "#aaa"), alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels([str(int(y)) for y in all_years], fontsize=9, color=TEXT_COLOR)
    ax6.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    ax6.set_ylabel("Total PnL (%)", color=TEXT_COLOR, fontsize=9)
    ax6.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)

    # 7. V23 PnL distribution
    ax7 = fig.add_subplot(gs[3, 0])
    style_ax(ax7, "V23 PnL Distribution")
    if len(df23) > 0:
        pnl = df23["pnl_pct"].values
        bins = np.arange(-30, 60, 2)
        ax7.hist(pnl[pnl > 0], bins=bins, color="#66bb6a", alpha=0.7, label=f"Wins ({(pnl>0).sum()})")
        ax7.hist(pnl[pnl <= 0], bins=bins, color="#ef5350", alpha=0.7, label=f"Losses ({(pnl<=0).sum()})")
        ax7.axvline(x=0, color="white", linewidth=1, alpha=0.5)
        ax7.axvline(x=np.mean(pnl), color="#ffb74d", linewidth=1.5, linestyle="--",
                    label=f"Mean={np.mean(pnl):+.1f}%")
        ax7.set_xlabel("PnL (%)", color=TEXT_COLOR, fontsize=9)
        ax7.legend(fontsize=7, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # 8. Fast exit & Peak protect comparison
    ax8 = fig.add_subplot(gs[3, 1])
    style_ax(ax8, "Key Exit Reason Comparison")
    key_reasons = ["fast_exit_loss", "peak_protect_dist", "peak_protect_ema", "signal_hard_cap"]
    y_pos = np.arange(len(key_reasons))
    h = 0.15
    for idx, (name, df) in enumerate(dfs.items()):
        if "exit_reason" not in df.columns:
            continue
        vals = [df[df["exit_reason"] == r]["pnl_pct"].sum() if r in df["exit_reason"].values else 0
                for r in key_reasons]
        ax8.barh(y_pos + (idx - 2) * h, vals, h, label=name,
                 color=MODEL_COLORS.get(name, "#aaa"), alpha=0.85)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(key_reasons, fontsize=7)
    ax8.legend(fontsize=6, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # 9. V23 Monthly heatmap
    ax9 = fig.add_subplot(gs[3, 2])
    style_ax(ax9, "V23 Monthly PnL Heatmap")
    if "entry_dt" in df23.columns and len(df23) > 0:
        dc = df23.dropna(subset=["entry_dt"]).copy()
        dc["month"] = dc["entry_dt"].dt.month
        dc["year_m"] = dc["entry_dt"].dt.year
        pivot = dc.pivot_table(values="pnl_pct", index="year_m", columns="month",
                               aggfunc="sum", fill_value=0)
        im = ax9.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-50, vmax=100)
        ax9.set_yticks(range(len(pivot.index)))
        ax9.set_yticklabels([str(int(y)) for y in pivot.index], fontsize=7)
        ax9.set_xticks(range(len(pivot.columns)))
        ax9.set_xticklabels([f"M{int(m)}" for m in pivot.columns], fontsize=7)
        for yi in range(len(pivot.index)):
            for xi in range(len(pivot.columns)):
                val = pivot.values[yi, xi]
                if val != 0:
                    ax9.text(xi, yi, f"{val:+.0f}", ha="center", va="center",
                             fontsize=5, color="black" if abs(val) < 30 else "white",
                             fontweight="bold")

    # 10. Summary table
    ax10 = fig.add_subplot(gs[3, 3])
    ax10.set_facecolor(BG_COLOR)
    ax10.axis("off")
    ax10.set_title("Summary Metrics", color="white", fontsize=11, fontweight="bold", pad=8)
    rows = []
    for name, df in dfs.items():
        m = calc_metrics(df.to_dict("records"))
        rows.append([name, f"{m['trades']}", f"{m['wr']:.1f}%",
                     f"{m['avg_pnl']:+.2f}%", f"{m['total_pnl']:+.0f}%",
                     f"{m['pf']:.2f}", f"{m['max_loss']:+.1f}%"])
    cols = ["Model", "#", "WR", "AvgPnL", "TotPnL", "PF", "MaxLoss"]
    table = ax10.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if key[0] == 0:
            cell.set_facecolor("#1a3a5c")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(BG_COLOR)
            cell.set_text_props(color=TEXT_COLOR)
            if key[1] == 0:
                lbl = rows[key[0] - 1][0]
                cell.set_text_props(color=MODEL_COLORS.get(lbl, TEXT_COLOR), fontweight="bold")

    save_path = os.path.join(RESULTS_DIR, "v23_dashboard.png")
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="VND,HPG,SSI,TCB,AAV,FPT",
                        help="Comma-separated symbols for per-symbol charts")
    parser.add_argument("--all", action="store_true", help="Generate charts for all 14 symbols")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 100)
    print("V23 VISUALIZATION — Multi-Model Trade Comparison")
    print("=" * 100)

    def v22_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v22_final(y_pred, returns, df_test, feature_cols, **kwargs)

    def v23_fn(y_pred, returns, df_test, feature_cols, **kwargs):
        return backtest_v23(y_pred, returns, df_test, feature_cols,
                            peak_protect_strong_threshold=0.12, **kwargs)

    print("\n  Running V19.1...")
    t_v191 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                           backtest_fn=backtest_v19_1)
    print("  Running V19.3...")
    t_v193 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                           backtest_fn=backtest_v19_3)
    print("  Running V22...")
    t_v22 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=v22_fn)
    print("  Running V23...")
    t_v23 = run_test_base(SYMBOLS, True, True, False, False, True, True, True, True, True, True,
                          backtest_fn=v23_fn)
    print("  Running Rule...")
    t_rule = run_rule_test(SYMBOLS)

    all_model_trades = {
        "V19.1": t_v191,
        "V19.3": t_v193,
        "V22": t_v22,
        "V23": t_v23,
        "Rule": t_rule,
    }

    # Print summary
    print("\n  SUMMARY:")
    for name, trades in all_model_trades.items():
        m = calc_metrics(trades)
        print(f"    {name:<8}: #{m['trades']:>4} WR={m['wr']:>5.1f}% AvgPnL={m['avg_pnl']:>+6.2f}% "
              f"TotPnL={m['total_pnl']:>+9.1f}% PF={m['pf']:>5.2f} MaxLoss={m['max_loss']:>+6.1f}%")

    # Dashboard
    print("\n  Generating dashboard...")
    dash_path = plot_dashboard(all_model_trades)
    print(f"  Saved: {dash_path}")

    # Per-symbol charts
    chart_symbols = SYMBOLS.split(",") if args.all else [s.strip() for s in args.symbols.split(",")]

    for sym in chart_symbols:
        sym = sym.strip()
        print(f"\n  Generating {sym} chart...")
        sym_data, date_col = load_price_data(sym)

        trades_dict = {}
        for name, trades in all_model_trades.items():
            trades_dict[name] = prepare_trades(trades, sym)

        save_path = os.path.join(RESULTS_DIR, f"v23_trades_{sym}.png")
        plot_multi_model_symbol(sym_data, date_col, trades_dict, sym, save_path)
        print(f"  Saved: {save_path}")

    print("\n" + "=" * 100)
    print(f"DONE — All charts saved to {RESULTS_DIR}/")
    print("=" * 100)
