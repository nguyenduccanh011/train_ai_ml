import psycopg2
import psycopg2.extras

conn = psycopg2.connect(
    host="localhost", port=5433,
    dbname="stockml", user="stockml", password="stockml_dev"
)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# ── 0. List all tables ──────────────────────────────────────────────────────
print("=" * 60)
print("ALL TABLES IN DB")
print("=" * 60)
cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
""")
tables = [r["table_name"] for r in cur.fetchall()]
print(tables)

# ── 1. Find template 2810 / pm2_hs10_zx25 ──────────────────────────────────
print("\n" + "=" * 60)
print("TEMPLATE LOOKUP: id=2810 or name like pm2_hs10_zx25")
print("=" * 60)
template_table = None
for t in tables:
    if "template" in t.lower():
        template_table = t
        break

if template_table:
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (template_table,))
    cols = [r["column_name"] for r in cur.fetchall()]
    print(f"Table '{template_table}' columns: {cols}")
    cur.execute(f"SELECT * FROM {template_table} WHERE id = 2810 LIMIT 5")
    rows = cur.fetchall()
    print(f"  id=2810: {rows}")
    if "name" in cols:
        cur.execute(f"SELECT * FROM {template_table} WHERE name ILIKE %s LIMIT 5", ("%pm2_hs10_zx25%",))
        rows = cur.fetchall()
        print(f"  name~pm2_hs10_zx25: {rows}")
else:
    print("No template table found")

# ── 2. Trades of template 2810, entry_date >= 2026-01-01 ───────────────────
print("\n" + "=" * 60)
print("TRADES of template_id=2810, entry_date >= 2026-01-01")
print("=" * 60)
trade_table = None
for t in tables:
    if "trade" in t.lower():
        trade_table = t
        break

if trade_table:
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (trade_table,))
    cols = [r["column_name"] for r in cur.fetchall()]
    print(f"Table '{trade_table}' columns: {cols}")

    # Try template_id direct
    if "template_id" in cols:
        cur.execute(f"""
            SELECT fold_id, model_id, entry_date, exit_date, ticker, pnl
            FROM {trade_table}
            WHERE template_id = 2810
              AND entry_date >= '2026-01-01'
            ORDER BY entry_date
            LIMIT 50
        """)
    elif "run_id" in cols:
        # Need join via runs -> folds -> templates
        cur.execute(f"""
            SELECT t.fold_id, t.model_id, t.entry_date, t.exit_date, t.ticker, t.pnl
            FROM {trade_table} t
            JOIN runs r ON r.id = t.run_id
            JOIN folds f ON f.id = r.fold_id
            WHERE f.template_id = 2810
              AND t.entry_date >= '2026-01-01'
            ORDER BY t.entry_date
            LIMIT 50
        """ if "folds" in tables else f"""
            SELECT * FROM {trade_table} LIMIT 3
        """)
    else:
        cur.execute(f"SELECT * FROM {trade_table} LIMIT 3")

    rows = cur.fetchall()
    print(f"  Count: {len(rows)}")
    for r in rows:
        print(dict(r))
else:
    print("No trade table found")
    # Try to find anything with 'entry' or 'exit'
    for t in tables:
        if any(k in t.lower() for k in ["signal", "position", "order", "fill"]):
            cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (t,))
            cols2 = [r["column_name"] for r in cur.fetchall()]
            print(f"  Candidate table '{t}': {cols2}")

# ── 3. Folds of template 2810 with test_end_date > 2025-12-31 ─────────────
print("\n" + "=" * 60)
print("FOLDS of template 2810 with test_end_date > 2025-12-31")
print("=" * 60)
fold_table = None
for t in tables:
    if "fold" in t.lower():
        fold_table = t
        break

if fold_table:
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (fold_table,))
    cols = [r["column_name"] for r in cur.fetchall()]
    print(f"Table '{fold_table}' columns: {cols}")

    date_col = next((c for c in cols if "test_end" in c), None)
    if date_col and "template_id" in cols:
        cur.execute(f"""
            SELECT id, template_id, {date_col}, train_start, train_end, test_start
            FROM {fold_table}
            WHERE template_id = 2810
              AND {date_col} > '2025-12-31'
            ORDER BY {date_col}
        """)
        rows = cur.fetchall()
        print(f"  Count: {len(rows)}")
        for r in rows:
            print(dict(r))
        # Also show all folds of 2810
        cur.execute(f"SELECT * FROM {fold_table} WHERE template_id = 2810 ORDER BY id")
        rows2 = cur.fetchall()
        print(f"\n  ALL folds of template 2810 ({len(rows2)} rows):")
        for r in rows2:
            print(dict(r))
    else:
        print(f"  Cols available: {cols}")
        cur.execute(f"SELECT * FROM {fold_table} WHERE template_id = 2810 LIMIT 10" if "template_id" in cols else f"SELECT * FROM {fold_table} LIMIT 5")
        for r in cur.fetchall():
            print(dict(r))
else:
    print("No fold table found")

# ── 4. OHLCV data availability ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("OHLCV / PRICE DATA AVAILABILITY")
print("=" * 60)
price_candidates = [t for t in tables if any(k in t.lower() for k in ["price", "ohlcv", "ohlc", "candle", "bar", "stock"])]
print(f"Price table candidates: {price_candidates}")
for t in price_candidates:
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (t,))
    cols = [r["column_name"] for r in cur.fetchall()]
    date_col = next((c for c in cols if c in ("date", "trade_date", "timestamp", "ts", "datetime", "candle_date")), None)
    if date_col:
        cur.execute(f"SELECT MIN({date_col}), MAX({date_col}), COUNT(*) FROM {t}")
        row = cur.fetchone()
        print(f"  {t}.{date_col}: min={row[0]}, max={row[1]}, count={row[2]}")
    else:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t}: {cols[:8]}... count={cur.fetchone()[0]}")

# ── 5. Experiments / runs config for template 2810 ────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENTS / RUNS / CONFIG for template 2810")
print("=" * 60)
exp_candidates = [t for t in tables if any(k in t.lower() for k in ["experiment", "run", "config", "model", "param"])]
print(f"Candidates: {exp_candidates}")
for t in exp_candidates:
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position", (t,))
    cols = [r["column_name"] for r in cur.fetchall()]
    if "template_id" in cols:
        cur.execute(f"SELECT * FROM {t} WHERE template_id = 2810 LIMIT 5")
        rows = cur.fetchall()
        if rows:
            print(f"\n  [{t}] template_id=2810 ({len(rows)} rows):")
            for r in rows:
                print(dict(r))

cur.close()
conn.close()
