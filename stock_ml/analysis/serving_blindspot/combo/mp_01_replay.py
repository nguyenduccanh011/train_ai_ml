"""MOMPM REPLAY — event study + 4 luật số-học trên trades gb_x08 s42.

mom_pm (PM-momentum VN30F1M, causal ATC) -> quantile trượt 252 obs (exclude current, min 120).
Delta pnl per trade = (1+pnl)*c[new_exit]/c[old_exit] - 1 - pnl  (tỷ lệ close, sạch slippage).
"""
import numpy as np
import pandas as pd
import duckdb

BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
pd.set_option("display.width", 250)

# ---------- mom_pm + causal quantile ----------
F = pd.read_parquet(BASE + "/pureml/p3_probe/p3b_intraday_features.parquet")
mom = F["mom_pm"].dropna()
vals = mom.values
q = np.full(len(vals), np.nan)
for i in range(len(vals)):
    past = vals[max(0, i - 252):i]
    if len(past) >= 120:
        q[i] = float((past < vals[i]).mean())
Q = pd.Series(q, index=pd.DatetimeIndex(mom.index)).dropna()
print(f"Q(mom_pm) causal 252: {len(Q)} ngày, {Q.index.min().date()} -> {Q.index.max().date()}")

# ---------- trades + closes ----------
E = pd.read_csv(BASE + "/exitmap/gbx08_enriched2.csv",
                parse_dates=["entry_date", "exit_date", "first_defer_dt"])
syms = sorted(E.symbol.unique())
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
SYM = {}
for s, g in bars.groupby("symbol"):
    SYM[s] = (pd.DatetimeIndex(g["date"]), g["close"].to_numpy(float))

def qat(d):
    v = Q.get(d)
    return float(v) if v is not None and not pd.isna(v) else np.nan

# decision bar = bar trước exit_date (per-symbol); q tại dec, dec-1..-3
rows = []
for idx, t in E.iterrows():
    di, c = SYM[t.symbol]
    xi = di.get_indexer([t.exit_date])[0]
    ei = di.get_indexer([t.entry_date])[0]
    r = dict(idx=idx, xi=xi, ei=ei, q_dec=np.nan, q_d1=np.nan, q_d2=np.nan, q_d3=np.nan)
    if xi > 0:
        r["q_dec"] = qat(di[xi - 1])
        for k in (1, 2, 3):
            if xi - 1 - k >= 0:
                r[f"q_d{k}"] = qat(di[xi - 1 - k])
    rows.append(r)
M = pd.DataFrame(rows).set_index("idx")
E = E.join(M)
E["rallied5"] = E.post_max_c >= 0.05
print(f"q_dec coverage: {E.q_dec.notna().sum()}/{len(E)} trades "
      f"(>=2022: {E[E.year_exit>=2022].q_dec.notna().sum()}/{len(E[E.year_exit>=2022])})")

BUCK = [0, .1, .3, .7, .9, 1.0001]
LAB = ["q<=.10", ".10-.30", ".30-.70", ".70-.90", "q>.90"]

def bucket_table(df, qcol, name):
    d = df[df[qcol].notna()].copy()
    d["b"] = pd.cut(d[qcol], BUCK, labels=LAB, include_lowest=True)
    g = d.groupby("b", observed=False).apply(lambda g: pd.Series(dict(
        n=len(g), rallied5=g.rallied5.mean(), post_max_c=g.post_max_c.mean(),
        post_end_c=g.post_end_c.mean(), post_min_c=g.post_min_c.mean(),
        giveback_u=g.giveback_u.mean(), eff=g.eff.mean())))
    print(f"\n== EVENT STUDY [{name}] bucket({qcol}) ==")
    print(g.round(3).to_string())

print("\n########## 1) EVENT STUDY ##########")
for sl, nm in [(E, "ALL"), (E[E.year_exit >= 2022], ">=2022"), (E[E.year_exit >= 2024], "2024+")]:
    bucket_table(sl, "q_dec", nm)
bucket_table(E[E.year_exit >= 2022], "q_d1", ">=2022 dec-1")
bucket_table(E[E.year_exit >= 2022], "q_d2", ">=2022 dec-2")

print("\n-- cohort dau vs q_dec (>=2022) --")
S = E[(E.year_exit >= 2022) & E.q_dec.notna()]
for nm, m in [("sold-then-rallied (pmax>=5%)", S.rallied5),
              ("sold-dung (pmax<5%)", ~S.rallied5),
              ("exit-roi-sap (pend<=-5%)", S.post_end_c <= -0.05),
              ("exit-roi-di-ngang (|pend|<5%)", S.post_end_c.abs() < 0.05)]:
    g = S[m]
    print(f"  {nm:34s} n={len(g):4d} q_dec mean={g.q_dec.mean():.3f} med={g.q_dec.median():.3f} "
          f"P(q<=.1)={ (g.q_dec<=.1).mean():.3f} P(q>=.9)={(g.q_dec>=.9).mean():.3f}")

# suppress victims: mkt_drop scan
P = pd.read_csv(BASE + "/exitmap/pdr_forensic_scan.csv",
                parse_dates=["entry_date", "exit_date", "sw_date"])
P["year_exit"] = P.exit_date.dt.year
P["q_sw"] = [qat(d) for d in P.sw_date]
PM = P[P.blocker == "mkt_drop"].copy()
print(f"\n-- suppress-victims (mkt_drop, n={len(PM)}, q_sw coverage {PM.q_sw.notna().sum()}) --")
d = PM[PM.q_sw.notna()].copy()
d["b"] = pd.cut(d.q_sw, BUCK, labels=LAB, include_lowest=True)
print(d.groupby("b", observed=False).apply(lambda g: pd.Series(dict(
    n=len(g), delta_mean=g.delta.mean(), delta_sum=g.delta.sum(),
    worse5=(g.delta <= -0.05).mean()))).round(3).to_string())
d2 = d[d.year_exit >= 2022]
print("  >=2022:")
print(d2.groupby("b", observed=False).apply(lambda g: pd.Series(dict(
    n=len(g), delta_mean=g.delta.mean(), delta_sum=g.delta.sum()))).round(3).to_string())

# ---------- replay machinery ----------
MEGA = set(E[(E.pnl_pct >= 0.4)].index)  # mega winners >= +40%

def report(name, deltas):
    """deltas: dict idx->delta pnl (only touched trades)."""
    if not deltas:
        print(f"\n== {name}: 0 trades cham ==")
        return
    dd = pd.Series(deltas)
    sub = E.loc[dd.index].copy()
    sub["d"] = dd
    yr = sub.groupby("year_exit").d.agg(["count", "sum"]).round(2)
    ge22 = sub[sub.year_exit >= 2022].d.sum()
    ge24 = sub[sub.year_exit >= 2024].d.sum()
    n22 = (sub.year_exit >= 2022).sum()
    mega_hit = sub[sub.index.isin(MEGA) & (sub.d < -0.02)]
    print(f"\n== {name}: n={len(sub)} dU_all={sub.d.sum():+.2f}u | "
          f">=2022 {ge22:+.2f}u (n={n22}) | 2024+ {ge24:+.2f}u | "
          f"tot-hon>=5% {(sub.d>=.05).sum()} / te-hon>=5% {(sub.d<=-.05).sum()}")
    print("   theo nam: " + " ".join(f"{y}:{v:+.2f}({c})" for y, (c, v) in yr.iterrows()))
    if len(mega_hit):
        print("   MEGA BI CHAM (d<-2%): " + "; ".join(
            f"{r.symbol} {r.exit_date.date()} pnl{r.pnl_pct:+.2f} d{r.d:+.3f}" for _, r in mega_hit.iterrows()))
    else:
        mega_any = sub[sub.index.isin(MEGA)]
        print(f"   mega cham nhe (n={len(mega_any)}, d_sum={mega_any.d.sum():+.3f}) — khong lenh nao d<-2%")

print("\n########## 2) REPLAY 4 LUAT ##########")

# R1: hoãn exit k bar khi mom_pm mạnh tại decision bar
for thr in (0.8, 0.9):
    for k in (1, 2):
        deltas = {}
        for idx, t in E.iterrows():
            if pd.isna(t.q_dec) or t.q_dec < thr:
                continue
            di, c = SYM[t.symbol]
            xi = int(t.xi)
            if xi < 0 or xi + k >= len(c):
                continue
            deltas[idx] = (1 + t.pnl_pct) * c[xi + k] / c[xi] - 1 - t.pnl_pct
        report(f"R1 defer{k}bar khi q_dec>={thr}", deltas)

# R2: trail 8% (arm 15%) chỉ khi mom_pm kiệt tại bar
for thr in (0.05, 0.1, 0.2):
    deltas = {}
    for idx, t in E.iterrows():
        di, c = SYM[t.symbol]
        ei, xi = int(t.ei), int(t.xi)
        if ei < 0 or xi < 0:
            continue
        ep = t.entry_price
        peak = c[ei]
        hit = None
        for j in range(ei + 1, xi):
            peak = max(peak, c[j])
            if peak / ep - 1.0 >= 0.15 and c[j] <= peak * 0.92:
                qj = qat(di[j])
                if not np.isnan(qj) and qj <= thr:
                    hit = j
                    break
        if hit is None:
            continue
        deltas[idx] = (1 + t.pnl_pct) * c[hit + 1] / c[xi] - 1 - t.pnl_pct
    report(f"R2 trail8/arm15 chi khi q<= {thr}", deltas)

# R3: bỏ mkt_drop suppress khi mom_pm kiệt tại bar bị nuốt (dùng cf_pnl của scan)
key = E.set_index(["symbol", E.entry_date.dt.date.astype(str)]).index
kmap = {kk: i for i, kk in zip(E.index, key)}
for thr in (0.1, 0.2, 0.3):
    deltas = {}
    for _, r in PM.iterrows():
        if pd.isna(r.q_sw) or r.q_sw > thr:
            continue
        i = kmap.get((r.symbol, str(r.entry_date.date())))
        if i is None:
            continue
        deltas[i] = r.cf_pnl - r.pnl  # = -delta cua scan
    report(f"R3 un-suppress mkt_drop khi q_sw<={thr}", deltas)

# R4: chỉ giữ snr-defer khi PM khỏe; PM yếu -> bán ngay tại first_defer_dt
for thr in (0.2, 0.5):
    deltas = {}
    for idx, t in E[E.deferred_before == True].iterrows():
        di, c = SYM[t.symbol]
        fi = di.get_indexer([t.first_defer_dt])[0]
        xi = int(t.xi)
        if fi < 0 or xi < 0 or fi + 1 >= len(c) or fi + 1 > xi:
            continue
        qf = qat(t.first_defer_dt)
        if np.isnan(qf) or qf > thr:
            continue
        deltas[idx] = (1 + t.pnl_pct) * c[fi + 1] / c[xi] - 1 - t.pnl_pct
    report(f"R4 un-defer snr khi q(fd)<={thr}", deltas)

E[["symbol", "entry_date", "exit_date", "year_exit", "pnl_pct", "q_dec", "q_d1", "q_d2", "q_d3",
   "rallied5", "post_max_c", "post_end_c", "giveback_u", "label"]].to_csv(
    BASE + "/combo/mp_trades_qdec.csv", index=False)
print("\nsaved mp_trades_qdec.csv")
