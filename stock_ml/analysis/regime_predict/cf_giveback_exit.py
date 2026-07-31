"""G2 pre-check: counterfactual conditional give-back trailing exit on champion trades.

For each trade, reconstruct a give-back trail (arm after +arm% MFE, exit if price falls gb% from
peak) applied ONLY when the entry-day regime is in a 'drift' bucket. Exits can only be EARLIER than
actual (tighter), never later. Measure per-year pnl delta vs the champion's actual exits. Approx
fill = peak*(1-gb) minus 0.006 roundtrip cost. Decisive question: does a regime-conditional tighter
exit lift the dead years (2024/2026) WITHOUT gutting the runner years (2020/2021)?
"""
import numpy as np
import pandas as pd
import duckdb
from trade_forensics import load_trades, _price_panel

COST = 0.006


def cf_pnl(seg, entry_price, actual_pnl, arm, gb):
    """Path-based give-back trail. seg: DataFrame(date index) with high/low ordered over the hold."""
    peak = entry_price
    armed = False
    for hi, lo in zip(seg["high"].to_numpy(), seg["low"].to_numpy()):
        peak = max(peak, hi)
        if peak / entry_price - 1.0 >= arm:
            armed = True
        if armed and lo <= peak * (1.0 - gb):
            return peak * (1.0 - gb) / entry_price - 1.0 - COST
    return actual_pnl  # trail never triggered -> keep the actual exit


def main():
    tr = load_trades()
    px = _price_panel(sorted(tr["symbol"].unique()))
    by_sym = {s: g.set_index("date")[["high", "low"]] for s, g in px.groupby("symbol")}

    # precompute per-trade hold segments once
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
        "disp_low&not_bull": lambda r: (r.disp_bucket == "low") and (r.mkt_bull == 0),
    }
    grids = [(0.05, 0.06), (0.05, 0.10), (0.08, 0.08), (0.10, 0.10)]

    base_year = tr.groupby("entry_year")["pnl_pct"].sum()
    base_tot = tr["pnl_pct"].sum()
    print(f"BASELINE total={base_tot:.1f}u  per-year=" +
          " ".join(f"{y}:{v:+.1f}" for y, v in base_year.items()))
    print(f"{'cond':20} {'arm/gb':10} {'total':>7} {'d2024':>7} {'d2026':>7} {'d2020':>7} {'d2021':>7} {'d2022':>7} {'d2023':>7} {'d2025':>7} {'ntrig':>6}")

    rows = list(tr.itertuples(index=False))
    for cname, cond in conditions.items():
        for arm, gb in grids:
            new_pnl = np.array(tr["pnl_pct"], dtype=float).copy()
            ntrig = 0
            for k, (r, seg) in enumerate(zip(rows, segs)):
                if seg is None or len(seg) == 0 or not r.entry_price or not cond(r):
                    continue
                p = cf_pnl(seg, r.entry_price, r.pnl_pct, arm, gb)
                if p != r.pnl_pct:
                    ntrig += 1
                new_pnl[k] = p
            nt = pd.Series(new_pnl).groupby(tr["entry_year"].values).sum()
            tot = new_pnl.sum()
            d = {y: nt.get(y, 0) - base_year.get(y, 0) for y in range(2020, 2027)}
            print(f"{cname:20} {f'{arm}/{gb}':10} {tot:>7.1f} "
                  f"{d[2024]:>+7.1f} {d[2026]:>+7.1f} {d[2020]:>+7.1f} {d[2021]:>+7.1f} "
                  f"{d[2022]:>+7.1f} {d[2023]:>+7.1f} {d[2025]:>+7.1f} {ntrig:>6}")


if __name__ == "__main__":
    main()
