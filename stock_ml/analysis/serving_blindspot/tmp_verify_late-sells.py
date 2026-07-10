# -*- coding: utf-8 -*-
# Adversarial re-verification of late-sell / giveback claims.
# Fully independent recompute from trades_metrics.csv + ohlcv.db (does NOT reuse tmp_lateexit_sim_out.csv).
import sqlite3, json
import numpy as np
import pandas as pd

DIR = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
DB = r"C:/Users/DUC CANH PC/Desktop/stock-serving/data/ohlcv.db"
SLIP = 0.0015
RT = 0.004  # 2*0.0015 commission + 0.001 tax

tr = pd.read_csv(DIR + "/trades_metrics.csv", parse_dates=["entry_date", "exit_date", "signal_date"])
N = len(tr)
print("n_closed_trades:", N)

con = sqlite3.connect(DB)
syms = sorted(tr.symbol.unique())
ph = ",".join("?" * len(syms))
bars = pd.read_sql(f"select symbol,date,open,high,low,close from ohlcv where symbol in ({ph}) order by symbol,date",
                   con, params=syms, parse_dates=["date"])
con.close()
B = {}
for s, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    B[s] = {"pos": {pd.Timestamp(d): i for i, d in enumerate(g["date"].values)},
            "dates": g["date"].values,
            "h": g["high"].to_numpy(float), "l": g["low"].to_numpy(float),
            "c": g["close"].to_numpy(float)}

pe = np.empty(N, int); px = np.empty(N, int)
for k, t in enumerate(tr.itertuples()):
    pe[k] = B[t.symbol]["pos"][t.entry_date]
    px[k] = B[t.symbol]["pos"][t.exit_date]
tr["pe"], tr["px"] = pe, px

# ---------- CLAIM 10: parity (all trades, not a sample) ----------
bad_px = bad_pnl = bad_hold = 0
gb_err = np.zeros(N); mfe_err = np.zeros(N); btp_bad = 0
for k, t in enumerate(tr.itertuples()):
    b = B[t.symbol]
    c_exit = b["c"][t.px]
    if abs(c_exit * (1 - SLIP) - t.exit_price) > 1e-6 * t.exit_price: bad_px += 1
    if abs((t.exit_price / t.entry_price - 1 - RT) - t.pnl_pct) > 1e-9: bad_pnl += 1
    if (t.px - t.pe) != t.holding_days: bad_hold += 1
    hc = b["c"][t.pe:t.px + 1]; hh = b["h"][t.pe:t.px + 1]
    gb_err[k] = abs(c_exit / hc.max() - 1 - t.giveback_pct)
    limit_raw = t.entry_price / (1 + SLIP)
    mfe_err[k] = abs(hh.max() / limit_raw - 1 - t.mfe_pct)
    if int(np.argmax(hh)) != t.bars_to_peak: btp_bad += 1
print(json.dumps({"parity": {"bad_exit_price": bad_px, "bad_pnl_formula": bad_pnl,
                             "bad_holding_days_vs_bars": bad_hold,
                             "giveback_def_max_abs_err_all": float(gb_err.max()),
                             "mfe_def_max_abs_err_all": float(mfe_err.max()),
                             "bars_to_peak_argmax_high_mismatch": btp_bad,
                             "n_spans_bad_adjustment": int(tr.spans_bad_adjustment.sum())}}))

# ---------- CLAIM 1: giveback distribution ----------
g = tr["giveback_pct"]
print(json.dumps({"claim1": {"mean": round(g.mean(), 5), "median": round(g.median(), 5),
                             "p75": round(g.quantile(0.75), 5),
                             "n_le_-10": int((g <= -0.10).sum()), "pct_le_-10": round((g <= -0.10).mean(), 4),
                             "n_le_-15": int((g <= -0.15).sum()), "pct_le_-15": round((g <= -0.15).mean(), 4)}}))

# ---------- CLAIM 2: giveback by bucket / exit_reason ----------
by_b = {k: {"n": len(v), "mean": round(v.mean(), 5), "med": round(v.median(), 5)}
        for k, v in tr.groupby("bucket")["giveback_pct"]}
by_r = {k: {"n": len(v), "mean": round(v.mean(), 5), "med": round(v.median(), 5)}
        for k, v in tr.groupby("exit_reason")["giveback_pct"]}
print(json.dumps({"claim2": {"by_bucket": by_b, "by_exit_reason": by_r}}))

# ---------- CLAIM 3: bars from (high) peak to exit ----------
tr["b2x"] = tr["holding_days"] - tr["bars_to_peak"]
def dsc(s):
    return {"n": len(s), "mean": round(s.mean(), 3), "med": float(s.median()),
            "p90": float(s.quantile(0.90))}
print(json.dumps({"claim3": {"overall": dsc(tr.b2x),
                             "big_win": dsc(tr[tr.bucket == "big_win"].b2x),
                             "big_loss": dsc(tr[tr.bucket == "big_loss"].b2x)}}))

# ---------- CLAIM 4: round-trip winners ----------
rt4 = tr[(tr.mfe_pct >= 0.10) & (tr.pnl_pct <= 0.03)]
print(json.dumps({"claim4": {"n": len(rt4), "pct_closed": round(len(rt4) / N, 4),
                             "sum_pnl_u": round(rt4.pnl_pct.sum(), 3),
                             "sum_mfe_minus_pnl_u": round((rt4.mfe_pct - rt4.pnl_pct).sum(), 3),
                             "mean_mfe": round(rt4.mfe_pct.mean(), 4), "mean_pnl": round(rt4.pnl_pct.mean(), 4),
                             "med_bars_peak_to_exit": float(rt4.b2x.median()),
                             "med_hold": float(rt4.holding_days.median())}}))

# ---------- CLAIM 5: winners-turned-loss ----------
wtl = tr[(tr.mfe_pct >= 0.12) & (tr.pnl_pct < 0)]
w = wtl.loc[wtl.pnl_pct.idxmin()]
print(json.dumps({"claim5": {"n": len(wtl), "sum_pnl_u": round(wtl.pnl_pct.sum(), 3),
                             "mean_mfe": round(wtl.mfe_pct.mean(), 4),
                             "worst": round(wtl.pnl_pct.min(), 4),
                             "worst_row": f"{w.symbol} {w.entry_date.date()}"}}, default=str))

# ---------- CLAIMS 6-8: independent overlay sims ----------
def sim_trade(t, mode, x=None):
    """Scan bars strictly before the actual trigger bar (px-1); fill close[i+1]*(1-slip).
    Returns (pnl, i) or None."""
    b = B[t.symbol]
    h, l, c = b["h"], b["l"], b["c"]
    peak_h = h[t.pe]; peak_c = c[t.pe]
    for i in range(t.pe + 1, t.px - 1):
        peak_h = max(peak_h, h[i]); peak_c = max(peak_c, c[i])
        if i - t.pe < 2:
            continue
        if mode == "be":
            fired = (peak_h / t.entry_price - 1 >= 0.12) and (l[i] <= t.entry_price)
        else:
            fired = c[i] <= peak_c * (1 - x)
        if fired:
            return c[i + 1] * (1 - SLIP) / t.entry_price - 1 - RT, i
    return None

def pf(p): return p[p > 0].sum() / abs(p[p <= 0].sum())

pnl0 = tr.pnl_pct.to_numpy()
out = {}
sims = {}
for nm, mode, x in [("be", "be", None), ("t6", "tr", 0.06), ("t8", "tr", 0.08), ("t10", "tr", 0.10)]:
    sim = pnl0.copy(); hit = np.zeros(N, bool); trig_i = np.full(N, -1)
    for k, t in enumerate(tr.itertuples()):
        r = sim_trade(t, mode, x)
        if r is not None:
            sim[k], hit[k] = r[0], True; trig_i[k] = r[1]
    d = sim - pnl0
    bb = pd.DataFrame({"b": tr.bucket, "d": d}).groupby("b")["d"].sum().round(3).to_dict()
    out[nm] = {"n_trig": int(hit.sum()), "sum_delta": round(d.sum(), 3),
               "old_sum": round(pnl0.sum(), 3), "new_sum": round(sim.sum(), 3),
               "pf_old": round(pf(pnl0), 3), "pf_new": round(pf(sim), 3),
               "n_improved": int((d > 1e-12).sum()), "gain_u": round(d[d > 0].sum(), 3),
               "n_hurt": int((d < -1e-12).sum()), "loss_u": round(d[d < 0].sum(), 3),
               "delta_by_bucket": bb,
               "delta_loss+bigloss+small": round(sum(bb.get(z, 0) for z in ["loss", "big_loss", "small"]), 3),
               "delta_excl_bad_adj": round(d[~tr.spans_bad_adjustment.to_numpy()].sum(), 3),
               "bigloss_old": int((pnl0 <= -0.15).sum()), "bigloss_new": int((sim <= -0.15).sum()),
               "worst_old": round(pnl0.min(), 4), "worst_new": round(sim.min(), 4)}
    sims[nm] = (sim, hit, trig_i)
print(json.dumps({"sims": out}))

# claim 6 extras: BE on wtl cohort + winners cut
sim_be, hit_be, _ = sims["be"]
m_wtl = tr.index.isin(wtl.index)
cut = (pnl0 > 0) & hit_be & (sim_be < pnl0)
print(json.dumps({"claim6_extra": {
    "wtl_n_triggered": int(hit_be[m_wtl].sum()),
    "wtl_savings_u": round((sim_be[m_wtl] - pnl0[m_wtl]).sum(), 3),
    "wtl_cohort_after": round(sim_be[m_wtl].sum(), 3),
    "winners_cut_n": int(cut.sum()),
    "winners_cut_lost_u": round((sim_be[cut] - pnl0[cut]).sum(), 3)}}))

# claim 8 extras: new/removed big_loss at trail_8, deepest new, worst trade identity
for nm in ["t8", "t10"]:
    sim, hit, trig_i = sims[nm]
    new_bl = (sim <= -0.15) & (pnl0 > -0.15)
    rem_bl = (sim > -0.15) & (pnl0 <= -0.15)
    sub = tr[new_bl].copy(); sub["sim"] = sim[new_bl]; sub["ti"] = trig_i[new_bl]
    sub = sub.sort_values("sim")
    rows = []
    for t in sub.head(8).itertuples():
        fill_date = str(pd.Timestamp(B[t.symbol]["dates"][int(t.ti) + 1]).date()) if t.ti >= 0 else None
        rows.append(f"{t.symbol} entry={t.entry_date.date()} simfill={fill_date} sim={t.sim:.4f} mae={t.mae_pct:.4f}")
    kworst = int(np.argmin(sim))
    tw = tr.iloc[kworst]
    wfill = str(pd.Timestamp(B[tw.symbol]["dates"][int(trig_i[kworst]) + 1]).date()) if trig_i[kworst] >= 0 else str(tw.exit_date.date())
    print(json.dumps({f"claim8_{nm}": {"n_new_bigloss": int(new_bl.sum()), "n_removed_bigloss": int(rem_bl.sum()),
                                       "deepest_new": rows,
                                       "worst_trade": f"{tw.symbol} entry={tw.entry_date.date()} simfill={wfill} sim={sim[kworst]:.4f} old={tw.pnl_pct:.4f} mae={tw.mae_pct:.4f}"}}))

# ---------- CLAIM 9: oracle sell-at-peak-close ----------
orac_f = (1 + tr.pnl_pct) / (1 + tr.giveback_pct) - 1          # claimed formula
orac_x = np.empty(N)                                            # exact from DB
for k, t in enumerate(tr.itertuples()):
    pkc = B[t.symbol]["c"][t.pe:t.px + 1].max()
    orac_x[k] = pkc * (1 - SLIP) / t.entry_price - 1 - RT
d_f = orac_f - tr.pnl_pct; d_x = orac_x - pnl0
bybf = pd.DataFrame({"b": tr.bucket, "d": d_f}).groupby("b")["d"].sum().round(2).to_dict()
bw = tr.bucket == "big_win"
print(json.dumps({"claim9": {"sum_delta_formula": round(d_f.sum(), 2), "sum_delta_exact": round(d_x.sum(), 2),
                             "new_total_formula": round(orac_f.sum(), 2),
                             "delta_by_bucket": bybf,
                             "bigwin_realized": round(pnl0[bw].sum(), 2),
                             "bigwin_at_peak": round(orac_f[bw].sum(), 2),
                             "pct_returned": round(d_f.sum() / orac_f.sum(), 3)}}))
