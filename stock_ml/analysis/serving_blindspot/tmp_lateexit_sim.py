# Late-exit / giveback forensic on the serving champion trades.
# Simulates (overlay on actual exits, engine-parity mechanics):
#   A) breakeven_lock_mfe=0.12, offset=0.0  (engine.py:1799-1806 semantics, fill close_next)
#   B) simple close-based peak-trail 6% / 8% / 10% (trigger close<=peak_close*(1-x), fill close_next)
# Engine parity facts used (engine.py):
#   entry_fill = limit*(1+slip) [line 1699]; peak_high = max HIGH incl entry bar [1700,1758]
#   exit: reason on bar i -> exit_idx=i+1, fill = close[exit_idx]*(1-slip) [2241-2249]
#   net = exit_fill/entry_fill - 1 - (2*commission + tax) = ... - 0.004
#   min_hold_bars=2 gates all reason exits (hold_bars = i - entry_idx >= 2)
import sqlite3, json
import numpy as np
import pandas as pd

DIR = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
TR = DIR + r"/trades_metrics.csv"
DB = r"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
SLIP = 0.0015
RT_COST = 2 * 0.0015 + 0.001  # 0.004

tr = pd.read_csv(TR, parse_dates=["entry_date", "exit_date", "signal_date"])
assert len(tr) == 3781, len(tr)

con = sqlite3.connect(DB)
syms = sorted(tr.symbol.unique())
ph = ",".join("?" * len(syms))
bars = pd.read_sql(
    f"select symbol, date, open, high, low, close from ohlcv where symbol in ({ph}) order by symbol, date",
    con, params=syms, parse_dates=["date"])
con.close()

BD = {}
for s, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    BD[s] = {
        "dates": g["date"].values,
        "pos": {pd.Timestamp(d): i for i, d in enumerate(g["date"].values)},
        "o": g["open"].to_numpy(), "h": g["high"].to_numpy(),
        "l": g["low"].to_numpy(), "c": g["close"].to_numpy(),
    }

# ---------- parity check ----------
bad_exit_px, bad_pnl = 0, 0
rows = []
for t in tr.itertuples():
    b = BD[t.symbol]
    pe = b["pos"].get(t.entry_date)
    px = b["pos"].get(t.exit_date)
    assert pe is not None and px is not None, (t.symbol, t.entry_date)
    if abs(b["c"][px] * (1 - SLIP) - t.exit_price) > 1e-6 * t.exit_price:
        bad_exit_px += 1
    if abs((t.exit_price / t.entry_price - 1 - RT_COST) - t.pnl_pct) > 1e-9:
        bad_pnl += 1
    rows.append((pe, px))
tr["pe"] = [r[0] for r in rows]
tr["px"] = [r[1] for r in rows]
print("PARITY: exit_price!=close*(1-slip):", bad_exit_px, "| pnl formula mismatch:", bad_pnl)

# verify giveback_pct definition on a sample: exit_close/peak_close - 1 ?
samp = tr.head(200)
errs = []
for t in samp.itertuples():
    b = BD[t.symbol]
    pk = b["c"][t.pe:t.px + 1].max()
    errs.append(abs((b["c"][t.px] / pk - 1) - t.giveback_pct))
print("giveback_pct == exit_close/max_close-1 ? max abs err on 200:", max(errs))

# bars from peak to exit
tr["bars_peak_to_exit"] = tr["holding_days"] - tr["bars_to_peak"]

# ---------- simulations ----------
def simulate(t, mode, x=None):
    """Return (sim_pnl, trigger_relday) or None if overlay never fires earlier.
    mode='be' breakeven lock (mfe_thr=0.12, offset=0, engine highs/lows basis);
    mode='trail' close-based peak trail of x."""
    b = BD[t.symbol]
    pe, px = t.pe, t.px
    entry_fill = t.entry_price
    act_trigger = px - 1  # actual exit reason fired on the bar before the fill bar
    h, l, c = b["h"], b["l"], b["c"]
    peak_h = h[pe]
    peak_c = c[pe]
    for i in range(pe + 1, act_trigger):  # strictly earlier triggers only
        if h[i] > peak_h:
            peak_h = h[i]
        if c[i] > peak_c:
            peak_c = c[i]
        hold = i - pe
        if hold < 2:
            continue
        fired = False
        if mode == "be":
            if peak_h / entry_fill - 1.0 >= 0.12 and l[i] <= entry_fill:
                fired = True
        else:
            if c[i] <= peak_c * (1.0 - x):
                fired = True
        if fired:
            fill = c[i + 1] * (1 - SLIP)
            return fill / entry_fill - 1 - RT_COST, i - pe
    return None

def run_overlay(mode, x=None):
    sim = tr["pnl_pct"].to_numpy().copy()
    hit = np.zeros(len(tr), bool)
    for k, t in enumerate(tr.itertuples()):
        r = simulate(t, mode, x)
        if r is not None:
            sim[k] = r[0]
            hit[k] = True
    return sim, hit

def pf(p):
    return p[p > 0].sum() / abs(p[p <= 0].sum())

def summarize(name, sim, hit):
    d = sim - tr["pnl_pct"].to_numpy()
    out = {
        "name": name, "n_triggered": int(hit.sum()),
        "sum_delta_u": round(float(d.sum()), 2),
        "sum_pnl_new": round(float(sim.sum()), 2), "sum_pnl_old": round(float(tr.pnl_pct.sum()), 2),
        "pf_new": round(float(pf(sim)), 2), "pf_old": round(float(pf(tr.pnl_pct.to_numpy())), 2),
        "n_improved": int((d > 1e-12).sum()), "n_hurt": int((d < -1e-12).sum()),
        "gain_from_improved_u": round(float(d[d > 0].sum()), 2),
        "loss_from_hurt_u": round(float(d[d < 0].sum()), 2),
        "bigloss_old": int((tr.pnl_pct <= -0.15).sum()), "bigloss_new": int((sim <= -0.15).sum()),
        "worst_old": round(float(tr.pnl_pct.min()), 4), "worst_new": round(float(sim.min()), 4),
        "delta_excl_bad_adj": round(float(d[~tr.spans_bad_adjustment.to_numpy()].sum()), 2),
    }
    for col in ["bucket", "exit_reason"]:
        g = pd.DataFrame({col: tr[col], "d": d}).groupby(col)["d"].agg(["sum", "count"])
        out[f"delta_by_{col}"] = {i: [round(r["sum"], 2), int(r["count"])] for i, r in g.iterrows()}
    return out

results = {}
for nm, mode, x in [("breakeven_lock_012", "be", None),
                    ("trail_6", "trail", 0.06), ("trail_8", "trail", 0.08), ("trail_10", "trail", 0.10)]:
    sim, hit = run_overlay(mode, x)
    results[nm] = summarize(nm, sim, hit)
    tr[f"sim_{nm}"] = sim
    tr[f"hit_{nm}"] = hit

# ---------- descriptive stats ----------
def q(s):
    s = s.dropna()
    return {"n": int(len(s)), "mean": round(float(s.mean()), 4), "med": round(float(s.median()), 4),
            "p75": round(float(s.quantile(0.75)), 4), "p90": round(float(s.quantile(0.90)), 4),
            "min": round(float(s.min()), 4)}

desc = {}
desc["giveback_by_bucket"] = {k: q(g) for k, g in tr.groupby("bucket")["giveback_pct"]}
desc["giveback_by_exit_reason"] = {k: q(g) for k, g in tr.groupby("exit_reason")["giveback_pct"]}
desc["giveback_overall"] = q(tr["giveback_pct"])
desc["bars_peak_to_exit_by_bucket"] = {k: q(g) for k, g in tr.groupby("bucket")["bars_peak_to_exit"]}
desc["bars_peak_to_exit_overall"] = q(tr["bars_peak_to_exit"])

# round-trip winners: mfe>=10% but pnl<=3%
rt = tr[(tr.mfe_pct >= 0.10) & (tr.pnl_pct <= 0.03)]
desc["roundtrip"] = {
    "n": len(rt), "pct_of_closed": round(len(rt) / len(tr), 4),
    "sum_pnl_u": round(rt.pnl_pct.sum(), 2),
    "sum_mfe_minus_pnl_u": round((rt.mfe_pct - rt.pnl_pct).sum(), 2),
    "mean_mfe": round(rt.mfe_pct.mean(), 4), "mean_pnl": round(rt.pnl_pct.mean(), 4),
    "med_bars_peak_to_exit": float(rt.bars_peak_to_exit.median()),
    "mean_bars_peak_to_exit": round(rt.bars_peak_to_exit.mean(), 1),
    "med_hold": float(rt.holding_days.median()),
}

# winners-turned-loss: mfe>=12% & pnl<0
wtl = tr[(tr.mfe_pct >= 0.12) & (tr.pnl_pct < 0)]
be = tr["sim_breakeven_lock_012"]
desc["winners_turned_loss"] = {
    "n": len(wtl), "sum_pnl_u": round(wtl.pnl_pct.sum(), 2),
    "mean_pnl": round(wtl.pnl_pct.mean(), 4), "mean_mfe": round(wtl.mfe_pct.mean(), 4),
    "worst": round(wtl.pnl_pct.min(), 4),
    "n_be_triggered": int(tr.loc[wtl.index, "hit_breakeven_lock_012"].sum()),
    "cohort_pnl_after_be": round(be.loc[wtl.index].sum(), 2),
    "cohort_savings_u": round((be.loc[wtl.index] - wtl.pnl_pct).sum(), 2),
}
# breakeven side effect: winners (pnl>0) cut by the lock
cut = tr[(tr.pnl_pct > 0) & tr.hit_breakeven_lock_012 & (tr.sim_breakeven_lock_012 < tr.pnl_pct)]
desc["be_cut_winners"] = {"n": len(cut),
                          "lost_u": round((cut.sim_breakeven_lock_012 - cut.pnl_pct).sum(), 2),
                          "mean_old_pnl": round(cut.pnl_pct.mean(), 4)}
imp = tr[tr.hit_breakeven_lock_012 & (tr.sim_breakeven_lock_012 > tr.pnl_pct)]
desc["be_improved"] = {"n": len(imp), "gain_u": round((imp.sim_breakeven_lock_012 - imp.pnl_pct).sum(), 2)}

print(json.dumps({"sims": results, "desc": desc}, indent=1, default=str))

# ---------- examples ----------
tr["d_be"] = tr.sim_breakeven_lock_012 - tr.pnl_pct
cols = ["symbol", "entry_date", "exit_date", "pnl_pct", "mfe_pct", "bars_to_peak",
        "holding_days", "giveback_pct", "d_be", "sim_trail_8", "spans_bad_adjustment"]
print("\n== TOP 10 winners-turned-loss saved by breakeven_lock ==")
print(tr.loc[wtl.index].nlargest(10, "d_be")[cols].to_string(index=False))
print("\n== TOP 10 biggest giveback (mfe>=15%) ==")
print(tr[tr.mfe_pct >= 0.15].nsmallest(10, "giveback_pct")[cols].to_string(index=False))
print("\n== TOP 8 trades HURT most by trail_8 ==")
tr["d_t8"] = tr.sim_trail_8 - tr.pnl_pct
print(tr.nsmallest(8, "d_t8")[["symbol", "entry_date", "exit_date", "pnl_pct", "mfe_pct",
                               "sim_trail_8", "d_t8", "bucket"]].to_string(index=False))
print("\n== TOP 8 trades SAVED most by trail_8 ==")
print(tr.nlargest(8, "d_t8")[["symbol", "entry_date", "exit_date", "pnl_pct", "mfe_pct",
                              "sim_trail_8", "d_t8", "bucket"]].to_string(index=False))

tr.to_csv(DIR + r"/tmp_lateexit_sim_out.csv", index=False)
