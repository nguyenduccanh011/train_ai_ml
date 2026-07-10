"""MOMPM REPLAY phần 2 — chiều NGƯỢC (đúng chiều screen P3):
R1b defer exit khi PM kiệt (q<=thr) — chống bán đáy cú xả (sold-then-rallied).
R2b trail8/arm15 chỉ khi PM mạnh (q>=thr) — tighten phía fade-cú-kéo.
R3b un-suppress mkt_drop khi PM mạnh (q_sw>=thr).
R4b un-defer snr khi PM mạnh (q(fd)>=thr) — chỉ giữ defer khi PM kiệt.
Cùng máy móc delta như mp_01.
"""
import numpy as np
import pandas as pd
import duckdb

BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
pd.set_option("display.width", 250)

F = pd.read_parquet(BASE + "/pureml/p3_probe/p3b_intraday_features.parquet")
mom = F["mom_pm"].dropna()
vals = mom.values
q = np.full(len(vals), np.nan)
for i in range(len(vals)):
    past = vals[max(0, i - 252):i]
    if len(past) >= 120:
        q[i] = float((past < vals[i]).mean())
Q = pd.Series(q, index=pd.DatetimeIndex(mom.index)).dropna()

E = pd.read_csv(BASE + "/exitmap/gbx08_enriched2.csv",
                parse_dates=["entry_date", "exit_date", "first_defer_dt"])
syms = sorted(E.symbol.unique())
duck = duckdb.connect(r"f:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)
bars = duck.execute(
    "SELECT symbol, date, close FROM ohlcv WHERE timeframe='1D' AND symbol IN ({}) "
    "ORDER BY symbol, date".format(",".join(f"'{s}'" for s in syms))).df()
duck.close()
bars["date"] = pd.to_datetime(bars["date"])
SYM = {s: (pd.DatetimeIndex(g["date"]), g["close"].to_numpy(float)) for s, g in bars.groupby("symbol")}

def qat(d):
    v = Q.get(d)
    return float(v) if v is not None and not pd.isna(v) else np.nan

rows = []
for idx, t in E.iterrows():
    di, c = SYM[t.symbol]
    xi = di.get_indexer([t.exit_date])[0]
    ei = di.get_indexer([t.entry_date])[0]
    rows.append(dict(idx=idx, xi=xi, ei=ei, q_dec=qat(di[xi - 1]) if xi > 0 else np.nan))
E = E.join(pd.DataFrame(rows).set_index("idx"))
MEGA = set(E[(E.pnl_pct >= 0.4)].index)

def report(name, deltas):
    if not deltas:
        print(f"\n== {name}: 0 trades cham ==")
        return
    dd = pd.Series(deltas)
    sub = E.loc[dd.index].copy()
    sub["d"] = dd
    yr = sub.groupby("year_exit").d.agg(["count", "sum"]).round(2)
    ge22 = sub[sub.year_exit >= 2022].d.sum()
    ge24 = sub[sub.year_exit >= 2024].d.sum()
    mega_hit = sub[sub.index.isin(MEGA) & (sub.d < -0.02)]
    print(f"\n== {name}: n={len(sub)} dU_all={sub.d.sum():+.2f}u | "
          f">=2022 {ge22:+.2f}u (n={(sub.year_exit>=2022).sum()}) | 2024+ {ge24:+.2f}u | "
          f"tot>=5% {(sub.d>=.05).sum()} / te>=5% {(sub.d<=-.05).sum()}")
    print("   theo nam: " + " ".join(f"{y}:{v:+.2f}({c})" for y, (c, v) in yr.iterrows()))
    print(f"   mega d<-2%: n={len(mega_hit)} d_sum={mega_hit.d.sum():+.3f}" +
          ("  [" + "; ".join(f"{r.symbol} {r.exit_date.date()} d{r.d:+.2f}" for _, r in mega_hit.head(8).iterrows()) + "]" if len(mega_hit) else ""))

# R1b: defer k bar khi PM kiệt
for thr in (0.1, 0.2):
    for k in (1, 2):
        deltas = {}
        for idx, t in E.iterrows():
            if pd.isna(t.q_dec) or t.q_dec > thr:
                continue
            di, c = SYM[t.symbol]
            xi = int(t.xi)
            if xi < 0 or xi + k >= len(c):
                continue
            deltas[idx] = (1 + t.pnl_pct) * c[xi + k] / c[xi] - 1 - t.pnl_pct
        report(f"R1b defer{k}bar khi q_dec<={thr}", deltas)

# R2b: trail8/arm15 chỉ khi PM mạnh
for thr in (0.8, 0.9, 0.95):
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
                if not np.isnan(qj) and qj >= thr:
                    hit = j
                    break
        if hit is None:
            continue
        deltas[idx] = (1 + t.pnl_pct) * c[hit + 1] / c[xi] - 1 - t.pnl_pct
    report(f"R2b trail8/arm15 chi khi q>={thr}", deltas)

# R3b: un-suppress mkt_drop khi PM mạnh
P = pd.read_csv(BASE + "/exitmap/pdr_forensic_scan.csv",
                parse_dates=["entry_date", "exit_date", "sw_date"])
P["q_sw"] = [qat(d) for d in P.sw_date]
PM = P[P.blocker == "mkt_drop"]
key = E.set_index(["symbol", E.entry_date.dt.date.astype(str)]).index
kmap = {kk: i for i, kk in zip(E.index, key)}
for thr in (0.7, 0.8, 0.9):
    deltas = {}
    for _, r in PM.iterrows():
        if pd.isna(r.q_sw) or r.q_sw < thr:
            continue
        i = kmap.get((r.symbol, str(r.entry_date.date())))
        if i is not None:
            deltas[i] = r.cf_pnl - r.pnl
    report(f"R3b un-suppress mkt_drop khi q_sw>={thr}", deltas)

# R4b: un-defer snr khi PM mạnh
for thr in (0.7, 0.8, 0.9):
    deltas = {}
    for idx, t in E[E.deferred_before == True].iterrows():
        di, c = SYM[t.symbol]
        fi = di.get_indexer([t.first_defer_dt])[0]
        xi = int(t.xi)
        if fi < 0 or xi < 0 or fi + 1 >= len(c) or fi + 1 > xi:
            continue
        qf = qat(t.first_defer_dt)
        if np.isnan(qf) or qf < thr:
            continue
        deltas[idx] = (1 + t.pnl_pct) * c[fi + 1] / c[xi] - 1 - t.pnl_pct
    report(f"R4b un-defer snr khi q(fd)>={thr}", deltas)
