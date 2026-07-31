"""Audit: is the 61-symbol vn_stock_default universe point-in-time or hindsight-curated,
and how much of the champion (gtrail tmpl 3377) P&L rides on non-PIT names?"""
import duckdb
import psycopg2

UNIV = ["ACB","AAS","AAV","ACV","BCG","BCM","BID","BSR","BVH","CTG",
        "DCM","DGC","DIG","DPM","EIB","FPT","FRT","GAS","GEX","GMD",
        "HCM","HDB","HDG","HPG","HSG","KBC","KDH","LPB","MBB","MSN",
        "MWG","NKG","NLG","NT2","NVL","OCB","PC1","PDR","PLX","PNJ",
        "POW","PVD","PVS","REE","SAB","SBT","SHB","SSI","STB","TCB",
        "TPB","VCB","VCI","VDS","VHM","VIC","VJC","VND","VNM","VPB","VTP"]

dd = duckdb.connect("F:/PROJECTS/train_ai_ml/market_data/market.duckdb", read_only=True)

# 1) listing dates (first 1D bar) of universe members
rows = dd.execute("""
    select symbol, min(date) fb, max(date) lb
    from ohlcv where timeframe='1D' and symbol in ({})
    group by symbol order by fb desc
""".format(",".join("'%s'" % s for s in UNIV))).fetchall()
print("=== universe first bars (latest 8) ===")
for r in rows[:8]:
    print(r)
missing = set(UNIV) - {r[0] for r in rows}
print("universe symbols missing from duckdb:", missing)
late = [(s, str(fb)) for s, fb, _ in rows if str(fb) > "2020-01-01"]
print("universe symbols with first bar AFTER 2020-01-01:", late)

# 2) PIT liquidity proxy: top-61 by traded value over windows
def topN(start, end, n=61):
    q = dd.execute(f"""
        select symbol, sum(traded_value) tv
        from ohlcv where timeframe='1D' and date >= '{start}' and date < '{end}'
        group by symbol order by tv desc limit {n}
    """).fetchall()
    return [r[0] for r in q]

for label, s, e in [("2019H2 (2019-07..2020-01)", "2019-07-01", "2020-01-01"),
                    ("2019-07..2020-06", "2019-07-01", "2020-07-01"),
                    ("2020 full", "2020-01-01", "2021-01-01"),
                    ("2022", "2022-01-01", "2023-01-01"),
                    ("2024", "2024-01-01", "2025-01-01"),
                    ("2025H2-26 (2025-07..2026-07)", "2025-07-01", "2026-07-01")]:
    t = topN(s, e)
    ov = len(set(t) & set(UNIV))
    print(f"overlap univ vs top61-by-traded-value {label}: {ov}/61")

pit61 = topN("2019-07-01", "2020-07-01")  # PIT proxy at backtest start
pit_ok = set(UNIV) & set(pit61)
non_pit = set(UNIV) - set(pit61)
print(f"\nPIT-eligible (in top61 by value 2019-07..2020-06): {len(pit_ok)}")
print(f"NON-PIT universe members ({len(non_pit)}):", sorted(non_pit))

# 3) buy&hold drift: equal-weight multiple 2020-01 -> 2026-06
def bh(symbols=None, exclude=None):
    filt = ""
    if symbols is not None:
        filt = "and symbol in ({})".format(",".join("'%s'" % s for s in symbols))
    if exclude is not None:
        filt = "and symbol not in ({})".format(",".join("'%s'" % s for s in exclude))
    q = dd.execute(f"""
        with px as (
          select symbol, date, close,
                 row_number() over (partition by symbol order by date) rn_a,
                 row_number() over (partition by symbol order by date desc) rn_d
          from ohlcv where timeframe='1D' and date >= '2020-01-01' and date <= '2026-06-30' {filt}
        ),
        firsts as (select symbol, close c0, date t0 from px where rn_a=1),
        lasts  as (select symbol, close c1, date t1 from px where rn_d=1)
        select f.symbol, f.t0, l.t1, l.c1/f.c0 mult
        from firsts f join lasts l using(symbol)
        where f.t0 <= '2020-03-01'  -- existed at backtest start
    """).fetchall()
    return q

univ_bh = bh(symbols=UNIV)
rest_bh = bh(exclude=UNIV)
import statistics as st
um = [r[3] for r in univ_bh]
rm = [r[3] for r in rest_bh]
print(f"\nB&H 2020-01->2026-06 (symbols existing by 2020-03):")
print(f"  universe  n={len(um)} mean_mult={st.mean(um):.2f} median={st.median(um):.2f}")
print(f"  non-univ  n={len(rm)} mean_mult={st.mean(rm):.2f} median={st.median(rm):.2f}")
# PIT top-61 alternative portfolio
pit_bh = bh(symbols=pit61)
pm = [r[3] for r in pit_bh]
print(f"  PIT-top61 n={len(pm)} mean_mult={st.mean(pm):.2f} median={st.median(pm):.2f}")

# 4) trade attribution on the audited runs
pg = psycopg2.connect(host="localhost", port=5433, dbname="stockml",
                      user="stockml", password="stockml_dev")
cur = pg.cursor()
for run in ["template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138",
            "template/x2_struct_to_k10_cs5ma50_r7ec_gtos-69338138",
            "template/x2_struct_to_k10_cs5ma50_r7ec_osdef-69338138",
            "template/x2_struct_to_k10_cs5ma50_r7earlycut-69338138"]:
    cur.execute("select symbol, pnl_pct from run_trades where run_id=%s", (run,))
    tr = cur.fetchall()
    a = [(s, p) for s, p in tr if s in non_pit]
    b = [(s, p) for s, p in tr if s not in non_pit]
    sa, sb = sum(p for _, p in a), sum(p for _, p in b)
    print(f"\n{run.split('cs5ma50_')[-1]}: trades={len(tr)}")
    print(f"  NON-PIT ({len(non_pit)} syms): n={len(a)} sum_pnl={sa:.2f} mean/trade={sa/len(a)*100 if a else 0:.2f}%  share_of_total_pnl={sa/(sa+sb)*100:.1f}%")
    print(f"  PIT-elig: n={len(b)} sum_pnl={sb:.2f} mean/trade={sb/len(b)*100 if b else 0:.2f}%")

# 5) top contributing symbols in gtrail, flag PIT status
cur.execute("""select symbol, count(*), sum(pnl_pct) from run_trades
              where run_id='template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138'
              group by symbol order by sum(pnl_pct) desc limit 15""")
print("\n=== gtrail top-15 symbols by sum pnl ===")
for s, n, p in cur.fetchall():
    print(f"  {s:5s} n={n:3d} sum_pnl={p:6.2f}  {'NON-PIT' if s in non_pit else 'pit'}")

# 6) did the run trade AAS/OCB before their listing? (sanity: it cannot; the point is
# their INCLUSION requires knowing in 2020 that they would list/be liquid)
cur.execute("""select symbol, min(entry_date) from run_trades
              where run_id='template/x2_struct_to_k10_cs5ma50_r7ec_gtrail-69338138'
              and symbol in ('OCB','AAS','VTP','BCM') group by symbol""")
print("\nfirst trades of late-listers:", cur.fetchall())
