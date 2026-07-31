"""Does the gated oversold-bounce 2024 edge (+2.9 on the 100-liq univ) SURVIVE on the exact
61-symbol CHAMPION universe? If it holds on the same names the momentum book trades, it's a real
decorrelated dead-year source worth single-book regime-switch integration. If it evaporates on the
61 large-caps, it was a small-cap-liquidity artifact. Reuses mr_02 build_cfg + mr_03 gate.
"""
from __future__ import annotations
import duckdb, numpy as np, pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from stock_ml.analysis.meanrev_sleeve.mr_02_holdsweep import build_cfg
from stock_ml.analysis.meanrev_sleeve.mr_03_bandgate import market_dd, wf_gated

ROOT = "f:/PROJECTS/train_ai_ml"


def load_champ():
    syms = sorted(pd.read_parquet(
        f"{ROOT}/bundles/bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf/prediction_history.parquet"
    )["symbol"].unique())
    con = duckdb.connect(f"{ROOT}/market_data/market.duckdb", read_only=True)
    inl = ",".join(repr(s) for s in syms)
    oh = con.execute(f"SELECT symbol,date,open,high,low,close,volume,traded_value FROM ohlcv "
                     f"WHERE timeframe='1D' AND symbol IN ({inl}) AND date>='2017-06-01' "
                     f"ORDER BY symbol,date").fetchdf()
    con.close(); oh["date"] = pd.to_datetime(oh["date"])
    return oh, len(syms)


if __name__ == "__main__":
    oh, nsym = load_champ()
    dd = market_dd(oh)
    Y = list(range(2022, 2027))
    print(f"CHAMPION universe: {nsym} symbols, {len(oh)} bars")
    print("cfg                          | Σtot  WR  |  " + "  ".join(f"{y}" for y in Y))
    for setup in ("oversold_any", "deep_oversold", "dip_uptrend"):
        for hold in (8,):
            for dd_floor in (-1.0, -0.15):
                D = build_cfg(oh, hold, setup)
                T = wf_gated(D, hold, dd, dd_floor)
                if len(T) == 0:
                    print(f"{setup:13s} h{hold} gate{dd_floor:+.2f} | no trades"); continue
                yr = {y: g["pnl"].sum() for y, g in T.groupby("yr")}
                wr = 100 * (T.pnl > 0).mean()
                per = "  ".join(f"{yr.get(y,0):+5.1f}" for y in Y)
                tag = "NOGATE" if dd_floor == -1.0 else f"dd>{dd_floor:+.2f}"
                print(f"{setup:13s} h{hold} {tag:9s} | {T.pnl.sum():+5.1f} {wr:3.0f}% | {per}  (n={len(T)})", flush=True)
    print("CHAMPUNIV_DONE")
