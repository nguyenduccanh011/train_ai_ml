"""Week-0 guardrail acceptance tests (BLINDSPOT_REPORT.md Phan 3) on the best
candidate w0_pyr_u10_r02 seed-42 trade frame vs champion 2646 seed-42 frame.

Guardrails checked:
 G1 bigwin-rate >= 19% (share of trades with pnl_pct > 0.15)
 G2 bigwin pnl share ~100%
 G3 hold>20-bar cohort WR ~84%
 G4 trade count within 10% of champion 1386/seed
 G5 per-year pnl 2022: candidate not worse than control by >5u
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent / "best_trades"
CAND = pd.read_csv(D / "trades_w0_pyr_u10_r02.csv", parse_dates=["entry_date", "exit_date"])
CTRL = pd.read_csv(D / "trades_n2_2643_wavestruct_la05_lamp02.csv", parse_dates=["entry_date", "exit_date"])
Y_CAND = pd.read_csv(D / "yearly_stats_w0_pyr_u10_r02.csv").set_index("year")
Y_CTRL = pd.read_csv(D / "yearly_stats_n2_2643_wavestruct_la05_lamp02.csv").set_index("year")


def stats(df: pd.DataFrame) -> dict:
    n = len(df)
    big = df[df["pnl_pct"] > 0.15]
    hold20 = df[df["holding_days"] > 20]
    return {
        "n_trades": n,
        "total_pnl_u": float(df["pnl_pct"].sum()),
        "bigwin_n": len(big),
        "bigwin_rate_pct": 100.0 * len(big) / n,
        "bigwin_pnl_u": float(big["pnl_pct"].sum()),
        "bigwin_pnl_share_pct": 100.0 * float(big["pnl_pct"].sum()) / float(df["pnl_pct"].sum()),
        "hold20_n": len(hold20),
        "hold20_share_pct": 100.0 * len(hold20) / n,
        "hold20_wr_pct": 100.0 * float((hold20["pnl_pct"] > 0).mean()),
        "hold20_mean_pct": 100.0 * float(hold20["pnl_pct"].mean()),
        "hold20_pnl_share_pct": 100.0 * float(hold20["pnl_pct"].sum()) / float(df["pnl_pct"].sum()),
        "weight_min": float(df["weight"].min()),
        "weight_max": float(df["weight"].max()),
        "n_weight_gt1": int((df["weight"] > 1.0).sum()),
    }


cand, ctrl = stats(CAND), stats(CTRL)
print("CAND " + json.dumps(cand))
print("CTRL " + json.dumps(ctrl))

py = pd.DataFrame({
    "cand_pnl": Y_CAND["total_pnl"],
    "ctrl_pnl": Y_CTRL["total_pnl"],
})
py["delta"] = py["cand_pnl"] - py["ctrl_pnl"]
print("PER_YEAR\n" + py.round(3).to_string())

g = {
    "G1_bigwin_rate>=19%": {"cand": round(cand["bigwin_rate_pct"], 2), "ctrl": round(ctrl["bigwin_rate_pct"], 2),
                            "pass": cand["bigwin_rate_pct"] >= 19.0},
    "G2_bigwin_share~100%": {"cand": round(cand["bigwin_pnl_share_pct"], 1), "ctrl": round(ctrl["bigwin_pnl_share_pct"], 1),
                             "pass": cand["bigwin_pnl_share_pct"] >= 90.0},
    "G3_hold20_WR~84%": {"cand": round(cand["hold20_wr_pct"], 2), "ctrl": round(ctrl["hold20_wr_pct"], 2),
                         "pass": cand["hold20_wr_pct"] >= 80.0},
    "G4_trades_within_10%_of_1386": {"cand": cand["n_trades"], "ref": 1386,
                                     "pass": abs(cand["n_trades"] - 1386) / 1386 <= 0.10},
    "G5_2022_pnl_not_worse_by_5u": {"cand": round(float(py.loc[2022, "cand_pnl"]), 3),
                                    "ctrl": round(float(py.loc[2022, "ctrl_pnl"]), 3),
                                    "delta": round(float(py.loc[2022, "delta"]), 3),
                                    "pass": float(py.loc[2022, "delta"]) > -5.0},
}
for k, v in g.items():
    print(f"GUARD {k}: {json.dumps(v)}")
print("ALL_PASS" if all(v["pass"] for v in g.values()) else "SOME_FAIL")
