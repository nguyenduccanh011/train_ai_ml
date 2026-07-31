"""Compare dispersion-gate variants vs champion gb_x08: composite, per-year, NAV/CAGR."""
import sqlalchemy as sa, pandas as pd
e = sa.create_engine("postgresql+psycopg2://stockml:stockml_dev@localhost:5433/stockml")

RUNS = {
    "gb_x08(base)": "template/gb_x08-32a8dfee",
    "ds_p33": None, "ds_p50": None, "ds_p66": None,
}
with e.connect() as c:
    for nm in ("ds_p33", "ds_p50", "ds_p66"):
        r = list(c.execute(sa.text(
            "select run_id from leaderboard_runs where run_name=:n and superseded=false "
            "order by generated_at desc limit 1"), {"n": nm}))
        if r:
            RUNS[nm] = r[0][0]

    print("=== HEADLINE (composite / pnl / avg / trades / NAV-CAGR) ===")
    hdr = f"{'variant':14} {'comp':>7} {'pnl':>7} {'avg':>7} {'tr':>5} {'CAGR':>7} {'NAV':>7} {'mddNAV':>7} {'f22':>6}"
    print(hdr)
    rows = {}
    for nm, rid in RUNS.items():
        if not rid:
            print(f"{nm:14} MISSING"); continue
        lb = list(c.execute(sa.text(
            "select composite_score,total_pnl,avg_pnl,trades from leaderboard_runs where run_id=:r"),
            {"r": rid}))[0]
        nav = list(c.execute(sa.text(
            "select cagr_adv,nav_adv,maxdd_nav,nav_f22_adv from leaderboard_nav where run_id=:r"),
            {"r": rid}))
        nav = nav[0] if nav else (None, None, None, None)
        rows[nm] = (lb, nav)
        def f(x, d=2):
            return f"{x:.{d}f}" if x is not None else "  --"
        print(f"{nm:14} {f(lb[0],1):>7} {f(lb[1],1):>7} {f(lb[2],4):>7} {lb[3]:>5} "
              f"{f(nav[0],4):>7} {f(nav[1],2):>7} {f(nav[2],4):>7} {f(nav[3],2):>6}")

    print("\n=== PER-YEAR avg_pnl (dead years 2024/2026 in focus) ===")
    yr_tables = {}
    for nm, rid in RUNS.items():
        if not rid:
            continue
        ys = pd.read_sql(sa.text(
            "select year,trades,total_pnl,avg_pnl,win_rate from run_yearly_stats where run_id=:r order by year"),
            c, params={"r": rid})
        yr_tables[nm] = ys.set_index("year")
    years = sorted(set().union(*[set(t.index) for t in yr_tables.values()]))
    print(f"{'year':6} " + " ".join(f"{nm.split('(')[0]:>22}" for nm in yr_tables))
    print(f"{'':6} " + " ".join(f"{'tr/avg/tot':>22}" for _ in yr_tables))
    for y in years:
        cells = []
        for nm, ys in yr_tables.items():
            if y in ys.index:
                r = ys.loc[y]
                cells.append(f"{int(r['trades']):>4}/{r['avg_pnl']:>+.3f}/{r['total_pnl']:>+6.1f}")
            else:
                cells.append(f"{'--':>22}")
        print(f"{y:6} " + " ".join(cells))
