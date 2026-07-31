"""G2 pre-check #2: counterfactual conditional HARD take-profit on champion trades.

If any bar's HIGH during the hold reaches entry*(1+tp), exit at that level (limit fill) minus cost,
applied ONLY when the entry-day regime is in a 'drift' bucket. Directly captures the transient
amplitude the entry head flags (D3) before the dead-year round-trip. Exits only EARLIER than actual.
Question: does a regime-conditional take-profit lift 2024/2026 without gutting the runner years?
"""
import numpy as np
import pandas as pd
from trade_forensics import load_trades, _price_panel

COST = 0.006


def tp_pnl(seg, entry_price, actual_pnl, tp):
    lvl = entry_price * (1.0 + tp)
    for hi in seg["high"].to_numpy():
        if hi >= lvl:
            return tp - COST
    return actual_pnl


def main():
    tr = load_trades()
    px = _price_panel(sorted(tr["symbol"].unique()))
    by_sym = {s: g.set_index("date")[["high", "low"]] for s, g in px.groupby("symbol")}
    segs = []
    for row in tr.itertuples(index=False):
        pl = by_sym.get(row.symbol)
        seg = pl.loc[(pl.index >= row.entry_date) & (pl.index <= row.exit_date)] if pl is not None else None
        segs.append(seg)

    conditions = {
        "always": lambda r: True,
        "disp_low": lambda r: r.disp_bucket == "low",
        "not_bull": lambda r: r.mkt_bull == 0,
        "disp_low|not_bull": lambda r: (r.disp_bucket == "low") or (r.mkt_bull == 0),
    }
    tps = [0.05, 0.08, 0.10, 0.12, 0.15]

    base_year = tr.groupby("entry_year")["pnl_pct"].sum()
    base_tot = tr["pnl_pct"].sum()
    print(f"BASELINE total={base_tot:.1f}u  per-year=" +
          " ".join(f"{y}:{v:+.1f}" for y, v in base_year.items()))
    print(f"{'cond':20} {'tp':6} {'total':>7} {'d2024':>7} {'d2026':>7} {'d2020':>7} {'d2021':>7} {'d2022':>7} {'d2023':>7} {'d2025':>7} {'ntrig':>6}")

    rows = list(tr.itertuples(index=False))
    for cname, cond in conditions.items():
        for tp in tps:
            new_pnl = np.array(tr["pnl_pct"], dtype=float).copy()
            ntrig = 0
            for k, (r, seg) in enumerate(zip(rows, segs)):
                if seg is None or len(seg) == 0 or not r.entry_price or not cond(r):
                    continue
                p = tp_pnl(seg, r.entry_price, r.pnl_pct, tp)
                if p != r.pnl_pct:
                    ntrig += 1
                new_pnl[k] = p
            nt = pd.Series(new_pnl).groupby(tr["entry_year"].values).sum()
            tot = new_pnl.sum()
            d = {y: nt.get(y, 0) - base_year.get(y, 0) for y in range(2020, 2027)}
            print(f"{cname:20} {tp:<6} {tot:>7.1f} "
                  f"{d[2024]:>+7.1f} {d[2026]:>+7.1f} {d[2020]:>+7.1f} {d[2021]:>+7.1f} "
                  f"{d[2022]:>+7.1f} {d[2023]:>+7.1f} {d[2025]:>+7.1f} {ntrig:>6}")


if __name__ == "__main__":
    main()
