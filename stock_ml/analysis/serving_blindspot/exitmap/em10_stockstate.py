# -*- coding: utf-8 -*-
"""em10: PHAN RA 2 QUAN THE (CUU vs PHA) theo TRANG THAI CO PHIEU tai bar can thiep dau tien.

Co che 1 (mkt_drop suppress): cohort = pdr_forensic_scan.csv blocker=mkt_drop, bar = sw_date.
Co che 2 (struct-donch80 vs %-trail): cohort = pdr_structtrail_cf.csv, bar = cf_date (%-trail fire).
CUU = delta > 0 (co che giup), PHA = delta < 0 (co che pha).

Discriminator do tai bar can thiep (chi dung du lieu den bar do):
  ma20_rel, ma60_rel : close/MA − 1 cua MA
  blw20_run          : so bar lien tiep close < MA20 (tinh ca bar j)
  snr21              : mean(ret,21)/std(ret,21) cua MA
  sell_run           : so bar force-sell lien tiep den bar j (proxy dl12/nonbull/lowbreadth)
  rs5                : ret5 ma − ret5 VNINDEX
  idio5              : ret5 ma − beta120 × ret5 VNINDEX
  age                : so bar tu entry
  gain, giveback_e   : gain hien tai, (peak−close)/entry

Sau do grid-search ranh gioi don + kep: cat u-pha / mat u-cuu (>=2022), kem mega-guard.
Chay tu repo root: python stock_ml/analysis/serving_blindspot/exitmap/em10_stockstate.py
"""
import json
import sys

import duckdb
import numpy as np
import pandas as pd
import psycopg2

sys.path.insert(0, r"f:/PROJECTS/train_ai_ml/stock_ml")
from src.backtest.engine import _load_vnindex  # noqa: E402

EM = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot/exitmap"
pd.set_option("display.width", 250)

scan = pd.read_csv(EM + "/pdr_forensic_scan.csv",
                   parse_dates=["entry_date", "exit_date", "sw_date"])
cfst = pd.read_csv(EM + "/pdr_structtrail_cf.csv",
                   parse_dates=["entry_date", "exit_date", "cf_date"])
md = scan[scan.blocker == "mkt_drop"].copy()
print(f"cohort mkt_drop: {len(md)} (>=2022: {(md.year_entry >= 2022).sum()})")
print(f"cohort struct  : {len(cfst)} (>=2022: {(cfst.year_entry >= 2022).sum()})")

pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                      user="stockml", password="stockml_dev")
cur = pg.cursor()
cur.execute("SELECT symbols_json FROM universe_versions WHERE universe_id=8 AND version=2")
uni_syms = [d["symbol"] for d in json.loads(cur.fetchone()[0])]
pg.close()

duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, open, high, low, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in uni_syms))).df()
alls = duck.execute("SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' "
                    "ORDER BY symbol, date").df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
alls["date"] = pd.to_datetime(alls["date"])

# ---------- market series (em08 parity) cho force gates ----------
piv = bars.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
rets = piv.pct_change()
mret_c = rets.replace([np.inf, -np.inf], np.nan).clip(-0.5, 0.5).mean(axis=1)
lvl = (1.0 + mret_c.fillna(0.0)).cumprod()
ma35 = lvl.rolling(35, min_periods=35).mean()
nonbull = ((lvl < ma35).rolling(2, min_periods=2).sum() >= 2).where(ma35.notna(), True)
pall = alls.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
ind = (pall > pall.rolling(50, min_periods=50).mean())
breadth = ind.sum(axis=1) / ind.notna().sum(axis=1).clip(lower=1)
lowbreadth = breadth < 0.25

vni = _load_vnindex()  # Series date -> level
vni = vni.sort_index()
vni_ret = vni.pct_change()
vni_ret5 = vni.pct_change(5)


def date_flag(series, dt):
    v = series.asof(dt)
    return bool(v) if not pd.isna(v) else False


def causal_leg(close, pct):
    n = len(close)
    leg = np.zeros(n, dtype=np.int8)
    direction, ext = 0, close[0] if n else 0.0
    for i in range(1, n):
        p = close[i]
        if direction >= 0 and p > ext:
            ext = p; direction = 1
        elif direction <= 0 and p < ext:
            ext = p; direction = -1
        elif direction == 1 and p <= ext * (1.0 - pct):
            direction = -1; ext = p
        elif direction == -1 and p >= ext * (1.0 + pct):
            direction = 1; ext = p
        leg[i] = direction
    return leg


def run_len(flags):
    """so phan tu True lien tiep ket thuc tai i (vector)."""
    out = np.zeros(len(flags), dtype=int)
    r = 0
    for i, f in enumerate(flags):
        r = r + 1 if f else 0
        out[i] = r
    return out


SYM = {}
for sym, g in bars.groupby("symbol"):
    g = g.reset_index(drop=True)
    dates = pd.DatetimeIndex(g["date"])
    c = g["close"].to_numpy(float)
    h = g["high"].to_numpy(float)
    lo = g["low"].to_numpy(float)
    cs = pd.Series(c)
    r1 = cs.pct_change()
    ma20 = cs.rolling(20, min_periods=20).mean().to_numpy()
    ma60 = cs.rolling(60, min_periods=60).mean().to_numpy()
    snr21 = (r1.rolling(21).mean() / (r1.rolling(21).std() + 1e-12)).to_numpy()
    ret5 = cs.pct_change(5).to_numpy()
    # beta 120 bar vs VNI (chi dung qua khu)
    vr = pd.Series(dates.normalize().map(vni_ret).to_numpy(dtype=float))
    cov = r1.rolling(120, min_periods=60).cov(vr)
    var = vr.rolling(120, min_periods=60).var()
    beta = (cov / (var + 1e-12)).clip(-1, 4).to_numpy()
    vr5 = dates.normalize().map(vni_ret5).to_numpy(dtype=float)
    # force gates (em02/em08 parity)
    below20 = (c < ma20) & ~np.isnan(ma20)
    bma20p2 = pd.Series(below20).rolling(2, min_periods=2).sum().to_numpy() >= 2
    leg12 = causal_leg(c, 0.12)
    leg6 = causal_leg(c, 0.06)
    nb = np.array([date_flag(nonbull, d) for d in dates])
    lb = np.array([date_flag(lowbreadth, d) for d in dates])
    force = (leg12 == -1) | (bma20p2 & nb) | ((leg6 == -1) & lb)
    SYM[sym] = dict(dates=dates, c=c, h=h, lo=lo, ma20=ma20, ma60=ma60, snr21=snr21,
                    ret5=ret5, beta=beta, vr5=vr5,
                    blw20_run=run_len(below20), sell_run=run_len(force))
print("symbols precomputed:", len(SYM))


def feats(sym, entry_date, entry_price, iv_date):
    s = SYM[sym]
    di = s["dates"]
    ei = di.get_indexer([entry_date])[0]
    j = di.get_indexer([iv_date])[0]
    if ei < 0 or j < 0:
        return None
    peak = float(np.max(s["h"][ei:j + 1]))
    c = s["c"][j]
    rs5 = s["ret5"][j] - s["vr5"][j]
    idio5 = s["ret5"][j] - s["beta"][j] * s["vr5"][j]
    return dict(
        ma20_rel=c / s["ma20"][j] - 1 if not np.isnan(s["ma20"][j]) else np.nan,
        ma60_rel=c / s["ma60"][j] - 1 if not np.isnan(s["ma60"][j]) else np.nan,
        blw20_run=int(s["blw20_run"][j]), snr21=float(s["snr21"][j]),
        sell_run=int(s["sell_run"][j]), rs5=float(rs5), idio5=float(idio5),
        age=int(j - ei), gain=c / entry_price - 1.0,
        giveback_e=(peak - c) / entry_price)


# trades goc de lay entry_price
tr = pd.read_csv(EM + "/gbx08_s42_trades.csv", parse_dates=["entry_date", "exit_date"])
epmap = {(r.symbol, r.entry_date): float(r.entry_price) for r in tr.itertuples()}

VARS = ["ma20_rel", "ma60_rel", "blw20_run", "snr21", "sell_run", "rs5", "idio5",
        "age", "gain", "giveback_e"]


def build(df, bar_col, name):
    rows = []
    for r in df.itertuples():
        ep = epmap.get((r.symbol, r.entry_date))
        if ep is None or r.symbol not in SYM:
            continue
        f = feats(r.symbol, r.entry_date, ep, getattr(r, bar_col))
        if f is None:
            continue
        rows.append(dict(symbol=r.symbol, entry_date=r.entry_date.date(),
                         year_entry=r.year_entry, pnl=r.pnl, cf_pnl=r.cf_pnl,
                         delta=r.delta, **f))
    out = pd.DataFrame(rows)
    out.to_csv(EM + f"/stockstate_{name}.csv", index=False)
    return out


def anatomy(df, name):
    g = df[df.year_entry >= 2022].copy()
    g["grp"] = np.where(g.delta > 0, "CUU", np.where(g.delta < 0, "PHA", "ZERO"))
    print("\n" + "=" * 110)
    print(f"[{name}] >=2022: n={len(g)}  u_cuu={g.delta[g.delta > 0].sum():+.2f} "
          f"(n={(g.delta > 0).sum()})  u_pha={g.delta[g.delta < 0].sum():+.2f} (n={(g.delta < 0).sum()})")
    q = g[g.grp != "ZERO"].groupby("grp")[VARS].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(q.round(3).to_string())
    # trong so u: phan phoi cua delta theo tung bien (weighted median thi phuc tap, in mean theo bucket sau)
    return g


def boundary_scan(g, name, mega_thr=0.2):
    """flag = 'trang thai gay' -> khong suppress / chuyen %-trail. improvement = -sum(delta[flag])."""
    u_pha = -g.delta[g.delta < 0].sum()
    u_cuu = g.delta[g.delta > 0].sum()
    grids = {
        "ma20_rel": ("<", [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0]),
        "ma60_rel": ("<", [-0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0]),
        "blw20_run": (">=", [1, 2, 3, 4, 5, 7, 10]),
        "snr21": ("<", [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0]),
        "sell_run": (">=", [1, 2, 3, 4, 5, 6, 8]),
        "rs5": ("<", [-0.10, -0.08, -0.06, -0.04, -0.03, -0.02, -0.01, 0.0]),
        "idio5": ("<", [-0.10, -0.08, -0.06, -0.04, -0.03, -0.02, -0.01, 0.0]),
        "age": (">=", [20, 30, 40, 60, 80]),
        "gain": ("<", [0.0, 0.05, 0.10, 0.15]),
        "giveback_e": (">=", [0.05, 0.08, 0.10, 0.15, 0.20]),
    }

    def flag_of(var, op, thr):
        v = g[var]
        return (v < thr) if op == "<" else (v >= thr)

    res = []
    for var, (op, thrs) in grids.items():
        for thr in thrs:
            fl = flag_of(var, op, thr) & g[var].notna()
            cut = -g.delta[fl & (g.delta < 0)].sum()
            lost = g.delta[fl & (g.delta > 0)].sum()
            megas = g[fl & (g.delta > mega_thr)]
            res.append(dict(rule=f"{var}{op}{thr}", n=int(fl.sum()), cut=cut, lost=lost,
                            net=cut - lost, pha_cov=cut / max(u_pha, 1e-9),
                            cuu_loss=lost / max(u_cuu, 1e-9),
                            mega_hit=",".join(f"{r.symbol}{str(r.entry_date)[:7]}" for r in megas.itertuples())))
    R = pd.DataFrame(res).sort_values("net", ascending=False)
    print(f"\n[{name}] TOP 20 ranh gioi don (net = u_pha cat − u_cuu mat, >=2022; "
          f"muc tieu pha_cov>=0.70 & cuu_loss<=0.20):")
    print(R.head(20).round(3).to_string(index=False))
    # cap doi: AND cua 8 rule don tot nhat (net) x nhau
    top = R.head(8)
    pair_res = []
    rules = []
    for _, rr in top.iterrows():
        var = rr.rule.split("<")[0].split(">=")[0]
        op = "<" if "<" in rr.rule else ">="
        thr = float(rr.rule.replace(var + op, ""))
        rules.append((var, op, thr))
    for a in range(len(rules)):
        for b in range(a + 1, len(rules)):
            va, oa, ta = rules[a]
            vb, ob, tb = rules[b]
            if va == vb:
                continue
            fl = (flag_of(va, oa, ta) & g[va].notna()) & (flag_of(vb, ob, tb) & g[vb].notna())
            cut = -g.delta[fl & (g.delta < 0)].sum()
            lost = g.delta[fl & (g.delta > 0)].sum()
            megas = g[fl & (g.delta > mega_thr)]
            pair_res.append(dict(rule=f"{va}{oa}{ta} & {vb}{ob}{tb}", n=int(fl.sum()),
                                 cut=cut, lost=lost, net=cut - lost,
                                 pha_cov=cut / max(u_pha, 1e-9), cuu_loss=lost / max(u_cuu, 1e-9),
                                 mega_hit=",".join(f"{r.symbol}{str(r.entry_date)[:7]}" for r in megas.itertuples())))
    P = pd.DataFrame(pair_res).sort_values("net", ascending=False)
    print(f"\n[{name}] TOP 15 ranh gioi KEP (AND):")
    print(P.head(15).round(3).to_string(index=False))
    return R, P


MK = build(md, "sw_date", "mkt")
ST = build(cfst, "cf_date", "struct")
gm = anatomy(MK, "MKT_DROP suppress")
gs = anatomy(ST, "STRUCT-DONCH80 vs %-trail")
Rm, Pm = boundary_scan(gm, "MKT_DROP")
Rs, Ps = boundary_scan(gs, "STRUCT")
