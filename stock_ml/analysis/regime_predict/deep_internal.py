"""Deep INTERNAL structure analysis (not surface aggregate).

Questions the surface NAV/CAGR hide:
  1. Is the dead-year (2024/2026) near-zero mean a homogeneous "uniformly mediocre" or a
     HETEROGENEOUS "big winners + big losers" distribution? (hidden sub-population = masked alpha)
  2. CAPTURE ratio (realized/mfe) — is the amplitude HELD or given back? Is capture predictable at
     entry, regime-robustly? (a NEW internal dimension, orthogonal to direction which is fragile)
  3. Which entry-observable head/feature best separates dead-year winners from losers, and does that
     separation HOLD across regimes (regime-robust internal quality signal)?
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from trade_forensics import load_trades

HEADS = "f:/PROJECTS/train_ai_ml/bundles/bundle_n2_consw20_conv04_vg_combo_hb_nbpbw_2027-01-01_wf/prediction_history.parquet"


def ic(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    return round(float(spearmanr(a[m], b[m]).correlation), 3)


def main():
    tr = load_trades()
    # join full head vector (score2..score5 incl score4 = mfe head) on entry signal date
    heads = pd.read_parquet(HEADS)
    heads["date"] = pd.to_datetime(heads["date"])
    tr = tr.merge(heads[["symbol", "date", "score2", "score3", "score4", "score5"]],
                  left_on=["symbol", "entry_signal_date"], right_on=["symbol", "date"],
                  how="left").drop(columns=["date"])

    print("=== 1. DEAD-YEAR HETEROGENEITY (pnl distribution by entry year) ===")
    print(f"{'yr':6}{'n':>5}{'mean':>8}{'med':>8}{'%win':>7}{'p10':>8}{'p90':>8}{'topDmean':>10}{'botDmean':>10}{'std':>7}")
    for y, g in tr.groupby("entry_year"):
        p = g["pnl_pct"]
        topD = p[p >= p.quantile(0.9)].mean()
        botD = p[p <= p.quantile(0.1)].mean()
        print(f"{y:<6}{len(g):>5}{p.mean():>8.3f}{p.median():>8.3f}{(p>0).mean():>7.2f}"
              f"{p.quantile(0.1):>8.3f}{p.quantile(0.9):>8.3f}{topD:>10.3f}{botD:>10.3f}{p.std():>7.3f}")

    print("\n=== 2. CAPTURE ratio (realized/mfe, mfe>0.03) by year ===")
    cap = tr[tr["mfe"] > 0.03].copy()
    cg = cap.groupby("entry_year")["capture"].agg(n="count", mean="mean", median="median")
    print(cg.round(3).to_string())

    print("\n=== 3. ENTRY-OBSERVABLE signal -> PnL IC by year (regime-robustness of selection) ===")
    feats = ["score", "score2", "score3", "score4", "score5", "dist_ma20", "breadth", "disp_dm20", "rvol20", "mfe"]
    print(f"{'feat':10}" + "".join(f"{y:>7}" for y in range(2020, 2027)) + f"{'pooled':>8}")
    for f in feats:
        if f not in tr.columns:
            continue
        row = {y: ic(g[f].to_numpy(float), g["pnl_pct"].to_numpy(float)) for y, g in tr.groupby("entry_year")}
        pooled = ic(tr[f].to_numpy(float), tr["pnl_pct"].to_numpy(float))
        print(f"{f:10}" + "".join(f"{row.get(y, float('nan')):>7}" if not np.isnan(row.get(y, np.nan)) else f"{'--':>7}" for y in range(2020, 2027)) + f"{pooled:>8}")

    print("\n=== 4. ENTRY signal -> CAPTURE IC by year (can we predict retention, not direction?) ===")
    print(f"{'feat':10}" + "".join(f"{y:>7}" for y in range(2020, 2027)) + f"{'pooled':>8}")
    for f in feats:
        if f not in cap.columns:
            continue
        row = {y: ic(g[f].to_numpy(float), g["capture"].to_numpy(float)) for y, g in cap.groupby("entry_year")}
        pooled = ic(cap[f].to_numpy(float), cap["capture"].to_numpy(float))
        print(f"{f:10}" + "".join(f"{row.get(y, float('nan')):>7}" if not np.isnan(row.get(y, np.nan)) else f"{'--':>7}" for y in range(2020, 2027)) + f"{pooled:>8}")


if __name__ == "__main__":
    main()
