"""score_nav_leaderboard: cham CAGR/NAV THAT cho leaderboard tu bang run_trades.

Thuoc chuan: nh_nav2.NavSim2 + shuffle_stats (K=25, roundtrip 0.006, settle_lag=2,
n=20 permutation; hai che do: advance_fee=0.0008 va None). Trades dump thang tu
Postgres run_trades (gia as-is — convention da verify bit-exact voi anchor
gb_t2783 x14.14 adv / x13.24 noadv / f22 3.47, xem HB2943_VALIDATION.md).

Ghi ket qua vao bang MOI `leaderboard_nav` (CREATE TABLE IF NOT EXISTS, KHONG
dung den leaderboard_runs). Idempotent: upsert theo run_id; run da cham voi cung
config_hash cua thuoc thi bo qua (tru khi --force).

CAGR = shuffle-mean-NAV ^ (1/years) - 1, voi years = span lich giao dich cua NAV
sim (ngay giao dich dau->cuoi trong cua so 2020-01-01..DATE_HI, ~6.5 nam, dong
nhat moi run — dung min/max entry_date se cong ao cho model vao lenh muon).

Chay:
  python stock_ml/scripts/ops/score_nav_leaderboard.py               # batch mac dinh (active vn_stock)
  python stock_ml/scripts/ops/score_nav_leaderboard.py --limit 50    # thu 50 run diem composite cao nhat
  python stock_ml/scripts/ops/score_nav_leaderboard.py --force       # cham lai tat ca
Xem them: F:/PROJECTS/hb2943_work/navboard/NAVBOARD_NOTES.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

# --- thuoc chuan: nh_nav2 nam ngoai repo (workdir hb2943) ---
NH_NAV2_DIR = os.environ.get("NH_NAV2_DIR", "F:/PROJECTS/hb2943_work")
sys.path.insert(0, NH_NAV2_DIR)
from nh_nav2 import DATE_HI, DB_PATH, NavSim2, shuffle_stats  # noqa: E402

WORKDIR = Path(os.environ.get("NAVBOARD_DIR", "F:/PROJECTS/hb2943_work/navboard"))
TMP_CSV = WORKDIR / "_tmp_trades.csv"

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://stockml:stockml_dev@localhost:5433/stockml"
).replace("postgresql+asyncpg://", "postgresql://")

MEASURE = dict(
    measure="nh_nav2.NavSim2+shuffle_stats",
    K=25,
    roundtrip=0.006,
    settle_lag=2,
    advance_fee_adv=0.0008,
    n_perm=20,
    date_lo="2020-01-01",
    date_lo_f22="2022-01-01",
    date_hi=DATE_HI,
    price_db=DB_PATH,
    years_basis="nav_calendar_span",
)
CONFIG_HASH = hashlib.md5(json.dumps(MEASURE, sort_keys=True).encode()).hexdigest()[:16]

DDL = """
CREATE TABLE IF NOT EXISTS leaderboard_nav (
    run_id       VARCHAR(512) PRIMARY KEY
                 REFERENCES leaderboard_runs(run_id) ON DELETE CASCADE,
    nav_adv      DOUBLE PRECISION,
    nav_noadv    DOUBLE PRECISION,
    cagr_adv     DOUBLE PRECISION,
    cagr_noadv   DOUBLE PRECISION,
    maxdd_nav    DOUBLE PRECISION,
    nav_f22_adv  DOUBLE PRECISION,
    years        DOUBLE PRECISION,
    n_trades_sim INTEGER,
    config_hash  VARCHAR(40),
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

UPSERT = """
INSERT INTO leaderboard_nav
    (run_id, nav_adv, nav_noadv, cagr_adv, cagr_noadv, maxdd_nav,
     nav_f22_adv, years, n_trades_sim, config_hash, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (run_id) DO UPDATE SET
    nav_adv = EXCLUDED.nav_adv,
    nav_noadv = EXCLUDED.nav_noadv,
    cagr_adv = EXCLUDED.cagr_adv,
    cagr_noadv = EXCLUDED.cagr_noadv,
    maxdd_nav = EXCLUDED.maxdd_nav,
    nav_f22_adv = EXCLUDED.nav_f22_adv,
    years = EXCLUDED.years,
    n_trades_sim = EXCLUDED.n_trades_sim,
    config_hash = EXCLUDED.config_hash,
    computed_at = now()
"""


def fetch_run_list(cur, market, include_superseded, run_like, limit):
    q = (
        "SELECT r.run_id FROM leaderboard_runs r "
        "WHERE EXISTS (SELECT 1 FROM run_trades t WHERE t.run_id = r.run_id)"
    )
    params: list = []
    if market:
        q += " AND r.market = %s"
        params.append(market)
    if not include_superseded:
        q += " AND r.superseded = false"
    if run_like:
        q += " AND r.run_id ILIKE %s"
        params.append(f"%{run_like}%")
    q += " ORDER BY r.composite_score DESC NULLS LAST"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    cur.execute(q, params)
    return [r[0] for r in cur.fetchall()]


def score_run(con, run_id):
    """Tra ve (row_tuple, skip_reason). row_tuple=None neu skip."""
    trades = pd.read_sql(
        "SELECT symbol, entry_date, exit_date, entry_price, exit_price "
        "FROM run_trades WHERE run_id = %s "
        "AND exit_date IS NOT NULL AND entry_price IS NOT NULL "
        "AND exit_price IS NOT NULL AND entry_date IS NOT NULL",
        con,
        params=(run_id,),
    )
    if len(trades) < 5:
        return None, f"chi {len(trades)} trades dong"
    trades.to_csv(TMP_CSV, index=False)

    sim = NavSim2(str(TMP_CSV), date_lo=MEASURE["date_lo"])
    if len(sim.trades) < 5 or len(sim.calendar) < 30:
        return None, (
            f"sim khong du du lieu (trades khop gia={len(sim.trades)}, skipped_db={sim.skipped_db})"
        )
    adv = shuffle_stats(
        sim,
        K=MEASURE["K"],
        roundtrip=MEASURE["roundtrip"],
        settle_lag=MEASURE["settle_lag"],
        advance_fee=MEASURE["advance_fee_adv"],
        n=MEASURE["n_perm"],
    )
    noadv = shuffle_stats(
        sim,
        K=MEASURE["K"],
        roundtrip=MEASURE["roundtrip"],
        settle_lag=MEASURE["settle_lag"],
        advance_fee=None,
        n=MEASURE["n_perm"],
    )
    years = (pd.Timestamp(sim.calendar[-1]) - pd.Timestamp(sim.calendar[0])).days / 365.25
    if years < 0.5:
        return None, f"cua so qua ngan ({years:.2f} nam)"

    nav_f22 = None
    sim22 = NavSim2(str(TMP_CSV), date_lo=MEASURE["date_lo_f22"])
    if len(sim22.trades) >= 5 and len(sim22.calendar) >= 30:
        nav_f22 = shuffle_stats(
            sim22,
            K=MEASURE["K"],
            roundtrip=MEASURE["roundtrip"],
            settle_lag=MEASURE["settle_lag"],
            advance_fee=MEASURE["advance_fee_adv"],
            n=MEASURE["n_perm"],
        )["mean"]

    row = (
        run_id,
        adv["mean"],
        noadv["mean"],
        adv["mean"] ** (1.0 / years) - 1.0,
        noadv["mean"] ** (1.0 / years) - 1.0,
        adv["dd_mean"],
        nav_f22,
        years,
        len(sim.trades),
        CONFIG_HASH,
    )
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--market", default="vn_stock", help="market filter (default vn_stock; '' = tat ca)"
    )
    ap.add_argument(
        "--include-superseded",
        action="store_true",
        help="cham ca run superseded (mac dinh chi active)",
    )
    ap.add_argument("--run-like", default=None, help="loc run_id theo substring")
    ap.add_argument(
        "--limit", type=int, default=None, help="chi cham N run diem composite cao nhat"
    )
    ap.add_argument("--force", action="store_true", help="cham lai ca run da co cung config_hash")
    args = ap.parse_args()

    WORKDIR.mkdir(parents=True, exist_ok=True)
    con = psycopg2.connect(DSN)
    cur = con.cursor()
    cur.execute(DDL)
    con.commit()

    run_ids = fetch_run_list(
        cur, args.market or None, args.include_superseded, args.run_like, args.limit
    )
    done: set = set()
    if not args.force:
        cur.execute("SELECT run_id FROM leaderboard_nav WHERE config_hash = %s", (CONFIG_HASH,))
        done = {r[0] for r in cur.fetchall()}
    todo = [r for r in run_ids if r not in done]
    print(
        f"config_hash={CONFIG_HASH} | ung vien={len(run_ids)} "
        f"da cham truoc do={len(run_ids) - len(todo)} | can cham={len(todo)}",
        flush=True,
    )

    n_ok = n_skip = 0
    t_start = time.time()
    for i, run_id in enumerate(todo, 1):
        t0 = time.time()
        try:
            row, reason = score_run(con, run_id)
        except Exception as exc:  # noqa: BLE001 — 1 run hong khong duoc chan batch
            row, reason = None, f"EXC {type(exc).__name__}: {exc}"
        if row is None:
            n_skip += 1
            print(f"[{i}/{len(todo)}] SKIP {run_id}: {reason}", flush=True)
            continue
        cur.execute(UPSERT, row)
        con.commit()
        n_ok += 1
        _, nav_a, nav_n, cagr_a, _, dd, f22, yrs, ntr, _ = row
        print(
            f"[{i}/{len(todo)}] {run_id}: NAV x{nav_a:.2f}/x{nav_n:.2f} "
            f"CAGR {cagr_a * 100:.1f}% DD {dd * 100:.1f}% "
            f"f22 {'x%.2f' % f22 if f22 else '—'} ({ntr} tr, {yrs:.2f}y, "
            f"{time.time() - t0:.1f}s)",
            flush=True,
        )

    dt = time.time() - t_start
    print(
        f"XONG: cham={n_ok} skip={n_skip} / todo={len(todo)} trong {dt / 60:.1f} phut", flush=True
    )
    con.close()


if __name__ == "__main__":
    main()
