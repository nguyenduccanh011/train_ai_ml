import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df_base = pd.read_csv("results/trades_v42_base.csv")
df_a    = pd.read_csv("results/trades_v42_a.csv")

for df in [df_base, df_a]:
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df["pnl_pct"]    = df["pnl_pct"].astype(float)

def cum_pnl_series(df):
    df2 = df.dropna(subset=["exit_date"]).sort_values("exit_date")
    return df2["exit_date"].values, df2["pnl_pct"].cumsum().values

d_base, p_base = cum_pnl_series(df_base)
d_a,    p_a    = cum_pnl_series(df_a)

def yearly(df):
    df2 = df.dropna(subset=["exit_date"]).copy()
    df2["year"] = pd.to_datetime(df2["exit_date"]).dt.year
    g = df2.groupby("year")
    years = sorted(df2["year"].unique())
    tot = [g.get_group(y)["pnl_pct"].sum() for y in years]
    wr  = [(g.get_group(y)["pnl_pct"] > 0).mean()*100 for y in years]
    return years, tot, wr

ry_b, tp_b, wr_b = yearly(df_base)
ry_a, tp_a, wr_a = yearly(df_a)

def exit_stats(df):
    d = {}
    for r, g in df.groupby("exit_reason"):
        pnl = g["pnl_pct"]
        d[r] = {"n": len(g), "avg": pnl.mean(), "tot": pnl.sum()}
    return d

eb_b = exit_stats(df_base)
eb_a = exit_stats(df_a)

hold_b = df_base["holding_days"].dropna()
hold_a = df_a["holding_days"].dropna()

COL_BASE = "#5b9bd5"
COL_A    = "#70c17a"
COL_GRID = "#2a2a3a"
COL_TEXT = "#cccccc"
BG_AX    = "#14141f"

fig = plt.figure(figsize=(20, 16), facecolor="#0f0f1a")
fig.suptitle(
    "V42 Backtest  |  V37a  vs  V37a + Exit Model B\n(ACB, VCB, HPG, MWG, VNM  |  2020-2025)",
    fontsize=15, color="white", fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

def style_ax(ax, title):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=COL_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(COL_GRID)
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
    ax.grid(True, color=COL_GRID, linewidth=0.5, linestyle="--", alpha=0.7)

# 1. Cumulative PnL
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(d_base, p_base, color=COL_BASE, linewidth=2,
         label=f"V42_base  n={len(df_base)}  avg={df_base['pnl_pct'].mean():+.2f}%  hold={hold_b.mean():.1f}d")
ax1.plot(d_a,    p_a,    color=COL_A,    linewidth=2,
         label=f"V42_a +ModelB  n={len(df_a)}  avg={df_a['pnl_pct'].mean():+.2f}%  hold={hold_a.mean():.1f}d")
ax1.axhline(0, color="#555", linewidth=0.7, linestyle=":")
ax1.fill_between(d_a,    0, p_a,    alpha=0.10, color=COL_A)
ax1.fill_between(d_base, 0, p_base, alpha=0.10, color=COL_BASE)
style_ax(ax1, "Cumulative PnL (%)")
ax1.set_ylabel("Cumulative PnL %", color=COL_TEXT, fontsize=8)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.3, labelcolor="white")

# 2. Yearly Total PnL
ax2 = fig.add_subplot(gs[1, 0])
all_years = sorted(set(ry_b) | set(ry_a))
x = np.arange(len(all_years))
w = 0.35
tp_b_arr = [tp_b[ry_b.index(y)] if y in ry_b else 0 for y in all_years]
tp_a_arr = [tp_a[ry_a.index(y)] if y in ry_a else 0 for y in all_years]
ax2.bar(x - w/2, tp_b_arr, w, color=COL_BASE, alpha=0.85, label="base")
ax2.bar(x + w/2, tp_a_arr, w, color=COL_A,   alpha=0.85, label="+ModelB")
ax2.axhline(0, color="#555", linewidth=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels([str(y) for y in all_years], fontsize=7)
style_ax(ax2, "Yearly Total PnL (%)")
ax2.legend(fontsize=7, framealpha=0.3, labelcolor="white")
ax2.set_ylabel("Total PnL %", color=COL_TEXT, fontsize=7)

# 3. Yearly Win Rate
ax3 = fig.add_subplot(gs[1, 1])
wr_b_arr = [wr_b[ry_b.index(y)] if y in ry_b else 0 for y in all_years]
wr_a_arr = [wr_a[ry_a.index(y)] if y in ry_a else 0 for y in all_years]
ax3.bar(x - w/2, wr_b_arr, w, color=COL_BASE, alpha=0.85, label="base")
ax3.bar(x + w/2, wr_a_arr, w, color=COL_A,   alpha=0.85, label="+ModelB")
ax3.axhline(50, color="#888", linewidth=0.7, linestyle="--")
ax3.set_xticks(x)
ax3.set_xticklabels([str(y) for y in all_years], fontsize=7)
ax3.set_ylim(0, 100)
style_ax(ax3, "Yearly Win Rate (%)")
ax3.legend(fontsize=7, framealpha=0.3, labelcolor="white")
ax3.set_ylabel("Win Rate %", color=COL_TEXT, fontsize=7)

# 4. Hold Distribution
ax4 = fig.add_subplot(gs[1, 2])
max_hold = int(max(hold_b.max(), hold_a.max())) + 5
bins = np.arange(0, max_hold, 5)
ax4.hist(hold_b, bins=bins, color=COL_BASE, alpha=0.6, label=f"base avg={hold_b.mean():.1f}d")
ax4.hist(hold_a, bins=bins, color=COL_A,   alpha=0.6, label=f"+ModelB avg={hold_a.mean():.1f}d")
style_ax(ax4, "Hold Duration Distribution")
ax4.set_xlabel("Hold (days)", color=COL_TEXT, fontsize=8)
ax4.set_ylabel("Count", color=COL_TEXT, fontsize=8)
ax4.legend(fontsize=7, framealpha=0.3, labelcolor="white")

# 5. Exit Reason Base
ax5 = fig.add_subplot(gs[2, 0])
reasons_b = sorted(eb_b.keys(), key=lambda r: eb_b[r]["avg"])
avgs_b = [eb_b[r]["avg"] for r in reasons_b]
colors_b = [COL_A if v >= 0 else "#e05c5c" for v in avgs_b]
ax5.barh(range(len(reasons_b)), avgs_b, color=colors_b, alpha=0.85)
ax5.set_yticks(range(len(reasons_b)))
ax5.set_yticklabels([f"{r}  n={eb_b[r]['n']}" for r in reasons_b], fontsize=7)
ax5.axvline(0, color="#888", linewidth=0.7)
style_ax(ax5, "Base: Avg PnL by Exit Reason")
ax5.set_xlabel("Avg PnL %", color=COL_TEXT, fontsize=7)

# 6. Exit Reason Model B
ax6 = fig.add_subplot(gs[2, 1])
reasons_a = sorted(eb_a.keys(), key=lambda r: eb_a[r]["avg"])
avgs_a = [eb_a[r]["avg"] for r in reasons_a]
colors_a = [COL_A if v >= 0 else "#e05c5c" for v in avgs_a]
ax6.barh(range(len(reasons_a)), avgs_a, color=colors_a, alpha=0.85)
ax6.set_yticks(range(len(reasons_a)))
ax6.set_yticklabels([f"{r}  n={eb_a[r]['n']}" for r in reasons_a], fontsize=7)
ax6.axvline(0, color="#888", linewidth=0.7)
style_ax(ax6, "+ModelB: Avg PnL by Exit Reason")
ax6.set_xlabel("Avg PnL %", color=COL_TEXT, fontsize=7)

# 7. Summary Table
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor(BG_AX)
ax7.axis("off")

def metrics(df):
    pnl = df["pnl_pct"]
    wr  = (pnl > 0).mean() * 100
    pos = pnl[pnl > 0].sum()
    neg = abs(pnl[pnl < 0].sum())
    pf  = pos / neg if neg > 0 else float("inf")
    return {
        "Trades":        str(len(df)),
        "Win Rate":      f"{wr:.1f}%",
        "Avg PnL":       f"{pnl.mean():+.2f}%",
        "Total PnL":     f"{pnl.sum():+.1f}%",
        "Profit Factor": f"{pf:.2f}",
        "Max Loss":      f"{pnl.min():+.2f}%",
        "Avg Hold":      f"{df['holding_days'].mean():.1f}d",
    }

mb = metrics(df_base)
ma = metrics(df_a)
rows = list(mb.keys())

y0 = 0.93
ax7.text(0.04, y0, "Metric",    color="white",   fontsize=8.5, fontweight="bold", transform=ax7.transAxes)
ax7.text(0.44, y0, "Base",      color=COL_BASE,  fontsize=8.5, fontweight="bold", transform=ax7.transAxes)
ax7.text(0.72, y0, "+Model B",  color=COL_A,     fontsize=8.5, fontweight="bold", transform=ax7.transAxes)
ax7.plot([0.02, 0.98], [y0 - 0.05, y0 - 0.05], color=COL_GRID, linewidth=0.8, transform=ax7.transAxes)

for i, row in enumerate(rows):
    yi = y0 - 0.11 - i * 0.10
    ax7.text(0.04, yi, row,       color=COL_TEXT, fontsize=8, transform=ax7.transAxes)
    ax7.text(0.44, yi, mb[row],   color=COL_BASE, fontsize=8, transform=ax7.transAxes)
    ax7.text(0.72, yi, ma[row],   color=COL_A,    fontsize=8, transform=ax7.transAxes)

ax7.set_title("Summary", color="white", fontsize=9, fontweight="bold", pad=6)

plt.savefig("results/v42_compare.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: results/v42_compare.png")
plt.show()
