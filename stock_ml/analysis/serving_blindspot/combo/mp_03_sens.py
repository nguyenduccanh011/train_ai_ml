"""MOMPM sensitivity — R1b (defer khi PM kiệt) k=1/2/3/5 x thr .1/.2/.3, ALL vs force-only.
Kiểm tra kèm: mean fwd ret k-bar sau exit theo bucket q_dec (horizon mismatch check)."""
import numpy as np
import pandas as pd
import duckdb

BASE = r"f:/PROJECTS/train_ai_ml/stock_ml/analysis/serving_blindspot"
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
    r = dict(idx=idx, xi=xi, q_dec=qat(di[xi - 1]) if xi > 0 else np.nan)
    for k in (1, 2, 3, 5):
        r[f"fwd{k}"] = c[xi + k] / c[xi] - 1 if (xi >= 0 and xi + k < len(c)) else np.nan
    rows.append(r)
E = E.join(pd.DataFrame(rows).set_index("idx"))

print("-- fwd ret sau exit theo bucket q_dec (>=2022) --")
S = E[(E.year_exit >= 2022) & E.q_dec.notna()].copy()
S["b"] = pd.cut(S.q_dec, [0, .1, .2, .3, .7, .9, 1.0001],
                labels=["<=.1", ".1-.2", ".2-.3", ".3-.7", ".7-.9", ">.9"], include_lowest=True)
print(S.groupby("b", observed=False)[["fwd1", "fwd2", "fwd3", "fwd5", "post_max_c", "post_end_c"]]
      .mean().round(4).to_string())
print("n:", S.groupby("b", observed=False).size().to_dict())

def run(df, name):
    print(f"\n-- R1b grid [{name}] --")
    print(f"{'thr':>4} {'k':>2} {'n':>4} {'dU_all':>8} {'dU>=22':>8} {'dU24+':>8} {'mega<-2%':>9}")
    MEGA = set(E[E.pnl_pct >= 0.4].index)
    for thr in (0.1, 0.2, 0.3):
        for k in (1, 2, 3, 5):
            d = df[(df.q_dec <= thr) & df[f"fwd{k}"].notna()].copy()
            d["dd"] = (1 + d.pnl_pct) * (1 + d[f"fwd{k}"]) - 1 - d.pnl_pct
            mh = d[d.index.isin(MEGA) & (d.dd < -0.02)]
            print(f"{thr:>4} {k:>2} {len(d):>4} {d.dd.sum():>+8.2f} "
                  f"{d[d.year_exit>=2022].dd.sum():>+8.2f} {d[d.year_exit>=2024].dd.sum():>+8.2f} "
                  f"{len(mh):>4}/{mh.dd.sum():+.2f}")

run(E, "ALL trades")
run(E[E.label.str.startswith("force")], "force-only")
