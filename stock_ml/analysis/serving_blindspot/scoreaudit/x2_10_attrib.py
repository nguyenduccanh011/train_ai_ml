"""x2_10: attribution kenh sell thu 2 (exit_score2 = reward_risk h10 trong stack gb_x08).

(1) Signal-level: tu fold checkpoints cua run x2 (co exit_score2), tinh causal z 252/60,
    dem theo nam: so bar zX2>thr, va so bar ADDITIVE (khong trung sell_ml/sell_force cua
    gb_x08 tai lap trong sa_scores.parquet).
(2) Trade-level: so run_trades cua tung run x2 vs gb_x08 (template/gb_x08-32a8dfee):
    trades doi exit theo nam + per-symbol PnL cho mega-check (VCI/VIC/GEX/LPB).
Usage: python x2_10_attrib.py <template_id_cua_run_z> <run_id_x2> <thr> [...]
  vi du: python x2_10_attrib.py 2804 template/x2_rr10_z25-xxxx 2.5
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import pandas as pd
import psycopg2

REPO = Path(__file__).resolve().parents[4]
SA = Path(__file__).parent / "sa_scores.parquet"
PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
GB_RUN = "template/gb_x08-32a8dfee"
MEGA = ["VCI", "VIC", "GEX", "LPB"]
W, MP = 252, 60


def causal_z(s: pd.Series, g: pd.Series) -> pd.Series:
    def _cz(x):
        m = x.rolling(W, min_periods=MP).mean()
        sd = x.rolling(W, min_periods=MP).std()
        return (x - m) / (sd + 1e-12)
    return s.groupby(g, sort=False).transform(_cz)


def load_folds(tid: int) -> pd.DataFrame:
    dirs = glob.glob(str(REPO / "results" / f"tmpl_{tid}_*" / "folds"))
    assert dirs, f"no folds for tmpl {tid}"
    fr = [pd.read_parquet(f) for f in sorted(glob.glob(dirs[0] + "/*.parquet"))]
    df = pd.concat(fr, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def signal_attrib(tid: int, thr: float) -> pd.DataFrame:
    df = load_folds(tid)
    assert "exit_score2" in df.columns, f"tmpl {tid}: folds thieu exit_score2!"
    df["zX2"] = causal_z(df["exit_score2"], df["symbol"])
    sa = pd.read_parquet(SA, columns=["symbol", "date", "sell_ml", "sell_force"])
    sa["date"] = pd.to_datetime(sa["date"])
    m = df.merge(sa, on=["symbol", "date"], how="left")
    m["year"] = m["date"].dt.year
    m["ch2"] = m["zX2"] > thr
    m["base_sell"] = m["sell_ml"].fillna(False) | m["sell_force"].fillna(False)
    m["ch2_add"] = m["ch2"] & ~m["base_sell"]
    g = m.groupby("year").agg(bars=("ch2", "size"), ch2=("ch2", "sum"), ch2_add=("ch2_add", "sum"))
    g["ch2_pct"] = (100 * g["ch2"] / g["bars"]).round(2)
    return g


def trades(run_id: str) -> pd.DataFrame:
    con = psycopg2.connect(**PG)
    t = pd.read_sql("SELECT symbol, entry_date, exit_date, pnl_pct, exit_reason, holding_days "
                    "FROM run_trades WHERE run_id=%s", con, params=(run_id,))
    con.close()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["exit_date"] = pd.to_datetime(t["exit_date"])
    return t


def trade_attrib(run_id: str) -> None:
    gb = trades(GB_RUN)
    x2 = trades(run_id)
    key = ["symbol", "entry_date"]
    mm = gb.merge(x2, on=key, how="outer", suffixes=("_gb", "_x2"), indicator=True)
    changed = mm[(mm["_merge"] != "both") | (mm["exit_date_gb"] != mm["exit_date_x2"])]
    changed = changed.copy()
    changed["year"] = changed["entry_date"].dt.year
    print(f"\n  trades gb={len(gb)} x2={len(x2)}; trades doi exit/entry: {len(changed)}")
    if len(changed):
        agg = changed.groupby("year").agg(
            n=("symbol", "size"),
            pnl_gb=("pnl_pct_gb", lambda s: s.sum(skipna=True)),
            pnl_x2=("pnl_pct_x2", lambda s: s.sum(skipna=True)))
        agg["dpnl"] = (agg["pnl_x2"] - agg["pnl_gb"]).round(3)
        print(agg.to_string())
    # mega-check
    gbm = gb[gb.symbol.isin(MEGA)].groupby("symbol")["pnl_pct"].agg(["sum", "count"])
    x2m = x2[x2.symbol.isin(MEGA)].groupby("symbol")["pnl_pct"].agg(["sum", "count"])
    mc = gbm.join(x2m, lsuffix="_gb", rsuffix="_x2", how="outer")
    print("\n  MEGA-check (sum pnl_pct / n trades):")
    print(mc.to_string())


def main():
    args = sys.argv[1:]
    for i in range(0, len(args), 3):
        tid, run_id, thr = int(args[i]), args[i + 1], float(args[i + 2])
        print(f"\n===== tmpl {tid} thr {thr} run {run_id} =====")
        print(signal_attrib(tid, thr).to_string())
        trade_attrib(run_id)


if __name__ == "__main__":
    main()
