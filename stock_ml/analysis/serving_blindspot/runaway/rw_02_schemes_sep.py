# -*- coding: utf-8 -*-
"""RUNAWAY AUTOPSY — per-scheme occupancy-consistent sim + separator ex-ante.

Moi exit scheme chay sequential sim RIENG (slot logic + blocked core theo dung
exit window cua scheme do). Separator: feature ex-ante tai bar tin hieu, quintile
net-after-occ + permutation null.
"""
import sqlite3

import numpy as np
import pandas as pd

BASE = r"f:\PROJECTS\train_ai_ml\stock_ml\analysis\serving_blindspot"
OUT = BASE + r"\runaway"
DB = r"C:\Users\DUC CANH PC\Desktop\stock-serving\data\ohlcv.db"
SLIP_IN, SLIP_OUT, FEE = 1.0015, 0.9985, 0.004

def net(e, x):
    return (x * SLIP_OUT) / (e * SLIP_IN) - 1.0 - FEE

sig = pd.read_csv(f"{BASE}/signals.csv")
uf = pd.read_csv(f"{BASE}/unfilled_signals.csv")
tr = pd.read_csv(f"{BASE}/trades_raw.csv")
tr["entry_signal_date"] = tr["entry_signal_date"].astype(str)
con = sqlite3.connect(DB)
px = pd.read_sql("select symbol,date,close,low from ohlcv order by symbol,date", con)
con.close()

A = {}
for s, g in px.groupby("symbol"):
    g = g.reset_index(drop=True)
    A[s] = dict(dates=g.date.to_numpy(), idx={d: i for i, d in enumerate(g.date)},
                c=g.close.to_numpy(float), l=g.low.to_numpy(float))

sell_idx = {}
for s, g in sig[sig.signal < 0].groupby("symbol"):
    a = A.get(s)
    if a is None:
        continue
    sell_idx[s] = np.array(sorted(a["idx"][d] for d in g.date if d in a["idx"]))

core_by_sym = {s: g.sort_values("entry_signal_date").reset_index(drop=True)
               for s, g in tr.groupby("symbol")}

u = uf[uf.drop_reason == "unfilled"].copy()
co = u[u.window_end_close > u.signal_close].sort_values(["symbol", "signal_date"])

def exit_bar(a, ei, scheme):
    n = len(a["c"])
    if scheme == "champ":
        ss = sell_idx.get_sym
    if scheme.startswith("ride"):
        return min(ei + int(scheme[4:]), n - 1), "horizon"
    if scheme == "trail10":
        peak = a["c"][ei]
        for t in range(ei + 1, n):
            peak = max(peak, a["c"][t])
            if a["c"][t] < peak * 0.90:
                return min(t + 1, n - 1), "trail"
        return n - 1, "end_of_data"
    raise ValueError(scheme)

def run_scheme(scheme):
    rows = []
    open_until = {}
    for r in co.itertuples():
        a = A[r.symbol]
        i = a["idx"][r.signal_date]
        n = len(a["c"])
        if i + 1 >= n:
            continue
        if open_until.get(r.symbol, -1) >= i:
            continue
        ei = i + 1
        ec = a["c"][ei]
        if scheme == "champ":
            ss = sell_idx.get(r.symbol, np.array([], int))
            nxt = ss[ss >= ei + 1]
            xi = min(int(nxt[0]) + 1, n - 1) if len(nxt) else n - 1
        else:
            xi, _ = exit_bar(a, ei, scheme)
        xc = a["c"][xi]
        cb = core_by_sym.get(r.symbol)
        blocked_pnl, blocked_n = 0.0, 0
        if cb is not None:
            x_date = a["dates"][xi]
            m = (cb.entry_signal_date > r.signal_date) & (cb.entry_signal_date <= x_date)
            blocked_n = int(m.sum()); blocked_pnl = float(cb.loc[m, "pnl_pct"].sum())
        open_until[r.symbol] = xi
        rows.append(dict(symbol=r.symbol, signal_date=r.signal_date, year=int(r.signal_date[:4]),
                         pnl=net(ec, xc), hold=xi - ei, blocked_n=blocked_n, blocked_pnl=blocked_pnl,
                         sig_bar=i))
    d = pd.DataFrame(rows)
    d["net"] = d.pnl - d.blocked_pnl
    return d

print("=" * 70)
print("PER-SCHEME (slot + occupancy nhat quan theo scheme)")
print(f"{'scheme':<9}{'taken':>6}{'pnl_u':>9}{'blk_n':>7}{'blk_u':>9}{'NET_u':>9}{'holdM':>7}{'WR':>6}")
schemes = {}
for sc in ("champ", "ride21", "ride40", "trail10"):
    d = run_scheme(sc)
    schemes[sc] = d
    print(f"{sc:<9}{len(d):>6}{d.pnl.sum():>9.1f}{int(d.blocked_n.sum()):>7}{d.blocked_pnl.sum():>9.1f}"
          f"{d.net.sum():>9.1f}{d.hold.median():>7.0f}{(d.pnl>0).mean():>6.2f}")

print("\nper-year NET (u):")
pv = pd.DataFrame({sc: d.groupby("year").net.sum() for sc, d in schemes.items()})
pv["n_champ"] = schemes["champ"].groupby("year").size()
print(pv.round(1).to_string())

# ---------- SEPARATOR ex-ante ----------
print("\n" + "=" * 70)
print("SEPARATOR EX-ANTE (scheme champ, per-trade net = pnl - blocked)")
d = schemes["champ"].copy()

# scores tai bar tin hieu
sb = sig[sig.signal > 0][["symbol", "date", "score", "score2", "score3", "score4", "score5",
                          "exit_score", "entry_csr"]]
d = d.merge(sb, left_on=["symbol", "signal_date"], right_on=["symbol", "date"], how="left")

# feature gia: dist MA20, ret5, ret21, vol20, runup tu low20
feat_rows = []
for r in d.itertuples():
    a = A[r.symbol]
    i = r.sig_bar
    c = a["c"]
    w = c[max(0, i - 19):i + 1]
    ma20 = w.mean()
    vol20 = np.std(np.diff(np.log(w))) if len(w) > 5 else np.nan
    ret5 = c[i] / c[i - 5] - 1 if i >= 5 else np.nan
    ret21 = c[i] / c[i - 21] - 1 if i >= 21 else np.nan
    lo20 = a["l"][max(0, i - 19):i + 1].min()
    feat_rows.append(dict(dist_ma20=c[i] / ma20 - 1, ret5=ret5, ret21=ret21, vol20=vol20,
                          runup_lo20=c[i] / lo20 - 1,
                          snr21=(ret21 / (vol20 * np.sqrt(21))) if (vol20 and not np.isnan(ret21) and vol20 > 0) else np.nan))
d = pd.concat([d.reset_index(drop=True), pd.DataFrame(feat_rows)], axis=1)

feats = ["score", "score2", "score3", "score4", "score5", "exit_score", "entry_csr",
         "dist_ma20", "ret5", "ret21", "vol20", "runup_lo20", "snr21"]
print(f"\nquintile NET-sum (u) theo feature (n={len(d)}, tong NET {d.net.sum():+.1f}u):")
print(f"{'feature':<12}" + "".join(f"{'Q'+str(q):>8}" for q in range(1, 6)) + f"{'IC(net)':>9}")
rng = np.random.default_rng(0)
best = []
for f in feats:
    v = d[f]
    ok = v.notna()
    if ok.sum() < 200:
        continue
    q = pd.qcut(v[ok], 5, labels=False, duplicates="drop")
    sums = d.loc[ok].groupby(q).net.sum()
    ic = np.corrcoef(v[ok].rank(), d.loc[ok, "net"].rank())[0, 1]
    print(f"{f:<12}" + "".join(f"{sums.get(i, np.nan):>8.1f}" for i in range(5)) + f"{ic:>9.3f}")
    # best-quintile candidate
    bi = sums.idxmax()
    sub = d.loc[ok][q == bi]
    best.append((f, bi, len(sub), sub.net.sum()))

print("\nbest-quintile sub-cohort + permutation null (1000 shuffle, chon ngau nhien cung n):")
nets = d.net.to_numpy()
for f, bi, nsub, s in sorted(best, key=lambda x: -x[3])[:6]:
    null = np.array([nets[rng.choice(len(nets), nsub, replace=False)].sum() for _ in range(1000)])
    p = (null >= s).mean()
    print(f"  {f} Q{bi+1}: n={nsub}, NET={s:+.1f}u, null mean {null.mean():+.1f}u sd {null.std():.1f}, p={p:.3f}")

d.to_csv(f"{OUT}/rw_sep_features.csv", index=False)
print(f"\nsaved {OUT}/rw_sep_features.csv")
