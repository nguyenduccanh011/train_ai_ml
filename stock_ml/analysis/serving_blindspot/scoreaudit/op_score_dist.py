"""Score-distribution audit for the operating-point retest.

Pull exit_score per (symbol,date) from run_signals for champion 2646 (velocity head) and
candidates xl_swal06 / xl_swal2 / pv_full. Recompute the causal per-symbol z (252/60,
same as _causal_zscore_by_symbol) and compare:
  - raw + z distribution stats
  - fire-rate of the sell band z>thr for thr in grid
  - quantile-matched threshold: z* s.t. candidate fire-rate == champion fire-rate at 2.0
  - rank-corr between candidate z and velocity z (same bars)
Also fire-rate under alternative z windows (189/126) for pv_full.
"""
import numpy as np
import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
RUNS = {
    "champ2646": "template/n2_2643_wavestruct_la05_lamp02-32a8dfee",
    "xl_swal06": "template/xl_swal06-60044c24",
    "xl_swal2": "template/xl_swal2-9c1f7e71",
    "pv_full": "template/pv_full-b82daabb",
}
GRID = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]


def causal_z(s: pd.Series, sym: pd.Series, window=252, min_periods=60) -> pd.Series:
    g = s.groupby(sym, sort=False)
    mu = g.transform(lambda x: x.rolling(window, min_periods=min_periods).mean())
    sd = g.transform(lambda x: x.rolling(window, min_periods=min_periods).std())
    return (s - mu) / (sd + 1e-9)


con = psycopg2.connect(**PG)
frames = {}
for name, rid in RUNS.items():
    df = pd.read_sql("SELECT symbol, date, score, exit_score FROM run_signals WHERE run_id=%s "
                     "ORDER BY symbol, date", con, params=(rid,))
    df["z"] = causal_z(df["exit_score"], df["symbol"])
    frames[name] = df
con.close()

champ_z = frames["champ2646"]["z"]
champ_rate = float((champ_z > 2.0).mean())
print(f"champion velocity head: fire-rate z>2.0 = {champ_rate:.5f} (n={champ_z.notna().sum()})")

rows = []
for name, df in frames.items():
    z = df["z"].dropna()
    raw = df["exit_score"]
    r = {
        "head": name,
        "raw_mean": raw.mean(), "raw_std": raw.std(),
        "raw_skew": raw.skew(), "raw_kurt": raw.kurt(),
        "z_skew": z.skew(), "z_kurt": z.kurt(),
    }
    for t in GRID:
        r[f"fire@{t}"] = float((z > t).mean())
    # quantile-matched threshold: candidate z* with same fire-rate as champ at 2.0
    r["z*_qmatch"] = float(np.quantile(z, 1.0 - champ_rate))
    rows.append(r)
T = pd.DataFrame(rows).set_index("head")
pd.set_option("display.width", 250)
print("\n== exit_score raw/z stats + sell-band fire-rates ==")
print(T.round(4).to_string())

# rank-corr of candidate z vs velocity z on identical bars
base = frames["champ2646"][["symbol", "date", "z"]].rename(columns={"z": "z_vel"})
print("\n== spearman corr z(candidate) vs z(velocity), same bars ==")
for name in ("xl_swal06", "xl_swal2", "pv_full"):
    m = frames[name][["symbol", "date", "z"]].merge(base, on=["symbol", "date"])
    m = m.dropna()
    print(f"{name}: rho={m['z'].corr(m['z_vel'], method='spearman'):.4f} (n={len(m)})")

# alternative z windows for pv_full (and champ as reference)
print("\n== fire-rate z>2.0 under alternative z_norm windows ==")
for name in ("champ2646", "pv_full", "xl_swal06"):
    df = frames[name]
    for w, mp in ((189, 60), (126, 60), (126, 40)):
        z = causal_z(df["exit_score"], df["symbol"], w, mp).dropna()
        print(f"{name} w={w}/mp={mp}: fire@2.0={float((z > 2.0).mean()):.5f} "
              f"z*_qmatch={float(np.quantile(z, 1.0 - champ_rate)):.3f}")
