"""Regime-slice + knock-on report for gb_ variants vs champion 2646 and candidate 2730 (seed 42).
Usage: python gb_regime.py <run_id> [<run_id> ...]
Slices: pnl of trades by ENTRY year (2020, >=2022, >=2023, >=2024, >=2025) — hard protocol.
Knock-on: champ-only entries (blocked because a deferred trade held the slot).
"""
import sys
from pathlib import Path

import pandas as pd
import psycopg2

PG = dict(host="localhost", port=5433, dbname="stockml", user="stockml", password="stockml_dev")
SQ = Path(__file__).resolve().parents[1]

champ = pd.read_csv(SQ / "st_champ2646_s42_trades.csv", parse_dates=["entry_date"])
cand = pd.read_csv(SQ / "sv_snr08_s42_trades.csv", parse_dates=["entry_date"])


def slices(df: pd.DataFrame) -> dict:
    y = df.entry_date.dt.year
    return {
        "2020": df[y == 2020].pnl_pct.sum(),
        ">=2022": df[y >= 2022].pnl_pct.sum(),
        ">=2023": df[y >= 2023].pnl_pct.sum(),
        ">=2024": df[y >= 2024].pnl_pct.sum(),
        ">=2025": df[y >= 2025].pnl_pct.sum(),
        "all": df.pnl_pct.sum(),
        "n": len(df),
    }


def knockon(var: pd.DataFrame) -> tuple[int, float, int, float]:
    m = champ.merge(var, on=["symbol", "entry_date"], how="outer",
                    suffixes=("_c", "_v"), indicator=True)
    co = m[m._merge == "left_only"]     # champ-only = blocked in variant
    vo = m[m._merge == "right_only"]
    return len(co), co.pnl_pct_c.sum(), len(vo), vo.pnl_pct_v.sum()


rows = {"champ2646": slices(champ), "cand2730": slices(cand)}
kon = {"cand2730": knockon(cand)}

con = psycopg2.connect(**PG)
for rid in sys.argv[1:]:
    df = pd.read_sql("SELECT symbol, entry_date, pnl_pct FROM run_trades WHERE run_id=%s",
                     con, params=(rid,))
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    name = rid.split("/")[-1].split("-")[0]
    rows[name] = slices(df)
    kon[name] = knockon(df)
con.close()

T = pd.DataFrame(rows).T
print("== pnl by ENTRY-year slice (seed 42) ==")
print(T.round(2).to_string())
print("\n== knock-on vs champion (champ-only = entries blocked by held slots) ==")
for k, (nc, pc, nv, pv) in kon.items():
    print(f"{k:12s} champ_only n={nc:3d} pnl_lost={pc:+.3f} | var_only n={nv:2d} pnl={pv:+.3f}")
