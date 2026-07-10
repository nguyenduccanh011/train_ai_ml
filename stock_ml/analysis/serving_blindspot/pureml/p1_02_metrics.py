# -*- coding: utf-8 -*-
"""P1-E1 buoc 2 — do luong quyet dinh kill/go cho tuyen ranking.

 (i)  Rank-IC OOS theo nam (Spearman per-date pred vs fwd20_dm) + null band
      (analytic + empirical shuffle-within-date cho model chinh)
 (ii) Spread top-20 vs bottom-20 va top-20 vs universe-EW, NET COST (roundtrip 0.7%
      leaderboard: comm 2x0.15% + slip 2x0.15% + tax 0.1%), rebalance tuan (5 bar),
      thuc thi lag-1-bar (vao close[t+1], ky close[t+1]->close[t+6])
 (iii) Turnover top-20/tuan + hysteresis (vao top-20, ra khi rank>30)
 (iv) Fill-rate: cohort ten MOI vao top-20 (hysteresis), limit close(t)*0.955 hieu luc
      40 bar nhu champion; % fill, ngay-den-fill, fwd20/fwd40 net tu GIA FILL,
      doi chieu voi vao thang close(t+1) khong limit (adverse-selection check)
 (v)  Overlap voi champion gb_x08 s42: % entry champion nam trong top-20 tai rebalance
      gan nhat; % (rebalance x ten) top-20 ma champion dang giu vi the
Out: p1_metrics.json + p1_ic_table.csv + p1_spread_table.csv + p1_fill_table.csv
"""
import json
import os

import numpy as np
import pandas as pd

OUT = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot\pureml"
TRADES = (r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
          r"\signalq\nicheloss\gb_x08_s42_trades.csv")
K, K_OUT = 20, 30
REB_EVERY = 5
LAG = 1                      # thuc thi: vao o close[t+LAG]
COST_RT = 2 * 0.0015 + 2 * 0.0015 + 0.001   # 0.007 roundtrip (leaderboard costs)
PB_DEPTH, PB_WIN = 0.045, 40  # pullback-limit champion
MIN_SYM = 40
N_SHUFFLE = 200
RNG = np.random.default_rng(42)
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

pred = pd.read_parquet(os.path.join(OUT, "p1_preds.parquet"))
ds = pd.read_parquet(os.path.join(OUT, "p1_dataset.parquet"),
                     columns=["symbol", "date", "open", "high", "low", "close"])

# ---- model cols: seed-mean = mean per-date pct-rank (scale-free, ranker co scale khac nhau)
def add_rank_mean(df, cols, name):
    r = [df.groupby("date")[c].rank(pct=True) for c in cols]
    df[name] = pd.concat(r, axis=1).mean(axis=1)

add_rank_mean(pred, ["pred_reg_s42", "pred_reg_s7", "pred_reg_s99"], "pred_reg_mean")
add_rank_mean(pred, ["pred_rank_s42", "pred_rank_s7", "pred_rank_s99"], "pred_rank_mean")

IC_MODELS = ["pred_reg_s42", "pred_reg_s7", "pred_reg_s99", "pred_reg_mean",
             "pred_rank_s42", "pred_rank_s7", "pred_rank_s99", "pred_rank_mean",
             "pred_reg_h10_s42", "pred_momo"]
PORT_MODELS = ["pred_reg_mean", "pred_rank_mean", "pred_momo"]

# ============================================================ (i) Rank-IC
def per_date_ic(df, xcol, ycol="fwd20_dm"):
    def f(g):
        m = g[xcol].notna() & g[ycol].notna()
        if m.sum() < MIN_SYM:
            return np.nan
        return g.loc[m, xcol].rank().corr(g.loc[m, ycol].rank())
    return df.groupby("date").apply(f).dropna()

def year_stats(ic):
    out = {}
    for y in YEARS:
        s = ic[ic.index.year == y]
        out[str(y)] = round(float(s.mean()), 4) if len(s) >= 30 else None
    p22 = ic[ic.index.year >= 2022]
    out["pooled_ge2022"] = round(float(p22.mean()), 4)
    out["t_cons_ge2022"] = (round(float(p22.mean() / (p22.std() / np.sqrt(len(p22) / 21.0))), 2)
                            if len(p22) > 42 else None)
    out["pooled_all"] = round(float(ic.mean()), 4)
    return out

def null_band(df, ycol="fwd20_dm", years_ge=2022, empirical=False):
    rows = []
    for dt, g in df.groupby("date"):
        if dt.year < years_ge:
            continue
        y = g[ycol].dropna()
        if len(y) < MIN_SYM:
            continue
        rows.append(y.rank().to_numpy(float))
    ns = np.array([len(r) for r in rows], float)
    sd = float(np.sqrt(np.mean(1.0 / (ns - 1.0))) / np.sqrt(len(rows)))
    out = {"null_sd": round(sd, 4), "null_lo": round(-1.96 * sd, 4),
           "null_hi": round(1.96 * sd, 4), "n_days": len(rows)}
    if empirical:
        means = []
        for _ in range(N_SHUFFLE):
            tot = 0.0
            for yv in rows:
                xp = RNG.permutation(yv)
                tot += np.corrcoef(xp, yv)[0, 1] if yv.std() > 0 else 0.0
            means.append(tot / len(rows))
        means = np.array(means)
        out["emp_lo"] = round(float(np.percentile(means, 2.5)), 4)
        out["emp_hi"] = round(float(np.percentile(means, 97.5)), 4)
    return out

print("=== (i) Rank-IC per-date Spearman vs fwd20_dm ===", flush=True)
ic_rows = []
for mcol in IC_MODELS:
    ycol = "fwd10_dm" if "h10" in mcol else "fwd20_dm"
    ic = per_date_ic(pred, mcol, ycol)
    st = year_stats(ic)
    ic_rows.append({"model": mcol, "target": ycol, **st})
    print(mcol, st, flush=True)
nb = null_band(pred, empirical=True)
print("null band (>=2022, shuffle-within-date):", nb, flush=True)
ic_table = pd.DataFrame(ic_rows)
ic_table.to_csv(os.path.join(OUT, "p1_ic_table.csv"), index=False)

# ============================================================ price grid
cal = pd.DatetimeIndex(sorted(pred.date.unique()))
ds = ds[ds.date >= cal[0]]
piv_c = ds.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
piv_o = ds.pivot_table(index="date", columns="symbol", values="open", aggfunc="last").sort_index()
piv_l = ds.pivot_table(index="date", columns="symbol", values="low", aggfunc="last").sort_index()
grid_dates = piv_c.index
gpos = {d: i for i, d in enumerate(grid_dates)}

pred_piv = {m: pred.pivot_table(index="date", columns="symbol", values=m, aggfunc="last")
            for m in PORT_MODELS}
reb_dates = [d for i, d in enumerate(cal) if i % REB_EVERY == 0]

def period_ret(members, t_reb, t_next):
    """EW return close[t_reb+LAG] -> close[t_next+LAG] cua members (NaN bo qua)."""
    i0, i1 = gpos[t_reb] + LAG, gpos[t_next] + LAG
    if i1 >= len(grid_dates):
        return np.nan
    c0 = piv_c.iloc[i0][members]
    c1 = piv_c.iloc[i1][members]
    r = (c1 / c0 - 1.0).dropna()
    return float(r.mean()) if len(r) else np.nan

def run_portfolio(mcol, scheme="plain"):
    """scheme plain: top-K moi rebalance. hysteresis: vao top-K, ra khi rank>K_OUT."""
    pp = pred_piv[mcol]
    rows, port = [], set()
    for j, t in enumerate(reb_dates[:-1]):
        t_next = reb_dates[j + 1]
        if t not in pp.index:
            continue
        s = pp.loc[t].dropna()
        if len(s) < MIN_SYM:
            continue
        rk = s.rank(ascending=False)          # 1 = tot nhat
        top = set(rk[rk <= K].index)
        bot = set(rk[rk > len(s) - K].index)
        if scheme == "plain":
            new_port = top
        else:
            stay = {x for x in port if x in rk.index and rk[x] <= K_OUT}
            new_port = stay | top
        n_prev = max(len(port), 1)
        exited = port - new_port
        entered = new_port - port
        turnover = (len(entered) + len(exited)) / 2.0 / max(len(new_port), 1)
        cost = (len(entered) + len(exited)) / 2.0 * COST_RT / max(len(new_port), 1)
        r_top = period_ret(sorted(new_port), t, t_next)
        r_bot = period_ret(sorted(bot), t, t_next)
        r_uni = period_ret(sorted(s.index), t, t_next)
        rows.append(dict(date=t, year=pd.Timestamp(t).year, n_port=len(new_port),
                         turnover=turnover, cost=cost, r_top=r_top, r_bot=r_bot,
                         r_uni=r_uni, entered=",".join(sorted(entered))))
        port = new_port
    return pd.DataFrame(rows)

def spread_table(pf, name):
    pf = pf.dropna(subset=["r_top", "r_uni"]).copy()
    pf["r_top_net"] = pf.r_top - pf.cost
    out = []
    for y, g in pf.groupby("year"):
        n = len(g)
        out.append(dict(model=name, year=y, n_reb=n,
                        turnover=round(float(g.turnover.mean()), 3),
                        top_gross_ann=round(float(g.r_top.mean() * 52), 4),
                        top_net_ann=round(float(g.r_top_net.mean() * 52), 4),
                        uni_ann=round(float(g.r_uni.mean() * 52), 4),
                        bot_ann=round(float(g.r_bot.mean() * 52), 4),
                        sprd_tb_net=round(float((g.r_top_net - g.r_bot).mean() * 52), 4),
                        sprd_tu_net=round(float((g.r_top_net - g.r_uni).mean() * 52), 4)))
    g = pf[pf.year >= 2022]
    out.append(dict(model=name, year="ge2022", n_reb=len(g),
                    turnover=round(float(g.turnover.mean()), 3),
                    top_gross_ann=round(float(g.r_top.mean() * 52), 4),
                    top_net_ann=round(float(g.r_top_net.mean() * 52), 4),
                    uni_ann=round(float(g.r_uni.mean() * 52), 4),
                    bot_ann=round(float(g.r_bot.mean() * 52), 4),
                    sprd_tb_net=round(float((g.r_top_net - g.r_bot).mean() * 52), 4),
                    sprd_tu_net=round(float((g.r_top_net - g.r_uni).mean() * 52), 4)))
    return out

print("\n=== (ii)+(iii) spread net-cost + turnover ===", flush=True)
sp_rows, pf_store = [], {}
for mcol in PORT_MODELS:
    for scheme in ("plain", "hyst"):
        pf = run_portfolio(mcol, scheme)
        pf_store[(mcol, scheme)] = pf
        rows = spread_table(pf, f"{mcol}|{scheme}")
        sp_rows += rows
        r = rows[-1]
        print(f"{mcol}|{scheme} >=2022: sprd_tb_net={r['sprd_tb_net']:+.3f} "
              f"sprd_tu_net={r['sprd_tu_net']:+.3f} top_net={r['top_net_ann']:+.3f} "
              f"uni={r['uni_ann']:+.3f} turnover={r['turnover']:.3f}", flush=True)
sp_table = pd.DataFrame(sp_rows)
sp_table.to_csv(os.path.join(OUT, "p1_spread_table.csv"), index=False)

# ============================================================ (iv) fill-rate
sym_grid = {s: i for i, s in enumerate(piv_c.columns)}
C, O, L = piv_c.to_numpy(), piv_o.to_numpy(), piv_l.to_numpy()

def fill_sim(pf):
    """Moi ten MOI vao portfolio tai rebalance t: limit = close(t)*(1-4.5%), 40 bar."""
    ev = []
    for r in pf.itertuples():
        if not r.entered:
            continue
        t = r.date
        it = gpos[t]
        for s in r.entered.split(","):
            js = sym_grid.get(s)
            if js is None or np.isnan(C[it, js]):
                continue
            limit = C[it, js] * (1.0 - PB_DEPTH)
            fill_i, fill_px = None, None
            for k in range(1, PB_WIN + 1):
                i2 = it + k
                if i2 >= len(grid_dates):
                    break
                lo, op = L[i2, js], O[i2, js]
                if np.isnan(lo):
                    continue
                if lo <= limit:
                    fill_i, fill_px = i2, min(op, limit) if not np.isnan(op) else limit
                    break
            def fwd(i0, px, h):
                i1 = i0 + h
                if i1 >= len(grid_dates) or np.isnan(C[i1, js]) or px is None:
                    return np.nan
                return C[i1, js] / px - 1.0 - COST_RT
            # doi chieu: vao thang close(t+1) khong limit
            i_alt = it + LAG
            alt_px = C[i_alt, js] if i_alt < len(grid_dates) else np.nan
            ev.append(dict(symbol=s, date=t, year=pd.Timestamp(t).year,
                           filled=fill_i is not None,
                           days_to_fill=(fill_i - it) if fill_i else np.nan,
                           f20=fwd(fill_i, fill_px, 20) if fill_i else np.nan,
                           f40=fwd(fill_i, fill_px, 40) if fill_i else np.nan,
                           a20=fwd(i_alt, alt_px, 20), a40=fwd(i_alt, alt_px, 40)))
    return pd.DataFrame(ev)

print("\n=== (iv) fill-rate pullback-limit 4.5%/40 tren cohort vao top-20 ===", flush=True)
fill_rows = []
for mcol in PORT_MODELS:
    fe = fill_sim(pf_store[(mcol, "hyst")])
    fe.to_parquet(os.path.join(OUT, f"p1_fill_events_{mcol}.parquet"), index=False)
    for grp, g in [(str(y), fe[fe.year == y]) for y in YEARS] + \
                  [("ge2022", fe[fe.year >= 2022]), ("all", fe)]:
        if not len(g):
            continue
        f = g[g.filled]
        fill_rows.append(dict(
            model=mcol, period=grp, n_orders=len(g),
            fill_rate=round(float(g.filled.mean()), 3),
            med_days=float(f.days_to_fill.median()) if len(f) else None,
            fill_f20=round(float(f.f20.mean()), 4) if f.f20.notna().any() else None,
            fill_f40=round(float(f.f40.mean()), 4) if f.f40.notna().any() else None,
            all_a20=round(float(g.a20.mean()), 4) if g.a20.notna().any() else None,
            all_a40=round(float(g.a40.mean()), 4) if g.a40.notna().any() else None,
            unfil_a20=round(float(g.loc[~g.filled, "a20"].mean()), 4)
            if (~g.filled).any() and g.loc[~g.filled, "a20"].notna().any() else None))
    r = [x for x in fill_rows if x["model"] == mcol and x["period"] == "ge2022"][0]
    print(f"{mcol} >=2022: fill_rate={r['fill_rate']:.3f} fill_f20={r['fill_f20']} "
          f"fill_f40={r['fill_f40']} vs all_a20={r['all_a20']} unfilled_a20={r['unfil_a20']}",
          flush=True)
fill_table = pd.DataFrame(fill_rows)
fill_table.to_csv(os.path.join(OUT, "p1_fill_table.csv"), index=False)

# ============================================================ (v) overlap champion
print("\n=== (v) overlap voi champion gb_x08 s42 ===", flush=True)
tr = pd.read_csv(TRADES, parse_dates=["entry_date", "exit_date"])
tr = tr[tr.entry_date >= "2021-01-01"]
overlap = {}
for mcol in PORT_MODELS:
    pf = pf_store[(mcol, "plain")].set_index("date")
    # membership top-K tai moi rebalance
    memb = {}
    pp = pred_piv[mcol]
    for t in pf.index:
        s = pp.loc[t].dropna()
        rk = s.rank(ascending=False)
        memb[t] = set(rk[rk <= K].index)
    reb_arr = pd.DatetimeIndex(sorted(memb.keys()))
    # (a) % entry champion co mat trong top-20 tai rebalance gan nhat truoc do
    hits, tot, per_year = 0, 0, {}
    for r in tr.itertuples():
        idx = reb_arr.searchsorted(r.entry_date, side="right") - 1
        if idx < 0:
            continue
        t = reb_arr[idx]
        tot += 1
        y = r.entry_date.year
        a, b = per_year.get(y, (0, 0))
        hit = r.symbol in memb[t]
        hits += hit
        per_year[y] = (a + hit, b + 1)
    # (b) % slot top-20 ma champion dang giu vi the (book overlap)
    slot_hit, slot_tot = 0, 0
    open_iv = list(zip(tr.symbol, tr.entry_date, tr.exit_date))
    for t, mm in memb.items():
        td = pd.Timestamp(t)
        holding = {s for s, e0, e1 in open_iv if e0 <= td <= e1}
        slot_hit += len(mm & holding)
        slot_tot += len(mm)
    overlap[mcol] = {
        "entry_in_top20_pct": round(hits / tot, 3) if tot else None,
        "entry_per_year": {str(y): f"{a}/{b}={a/b:.2f}" for y, (a, b) in sorted(per_year.items())},
        "book_slot_overlap_pct": round(slot_hit / slot_tot, 3) if slot_tot else None}
    print(mcol, overlap[mcol], flush=True)

with open(os.path.join(OUT, "p1_metrics.json"), "w") as fh:
    json.dump({"ic": ic_rows, "null_band_ge2022": nb, "spread": sp_rows,
               "fill": fill_rows, "overlap": overlap,
               "config": dict(K=K, K_OUT=K_OUT, reb_every=REB_EVERY, lag=LAG,
                              cost_rt=COST_RT, pb_depth=PB_DEPTH, pb_win=PB_WIN)},
              fh, indent=1, default=str)
print("\nsaved p1_metrics.json / p1_ic_table.csv / p1_spread_table.csv / p1_fill_table.csv")
print("P1_02_DONE")
