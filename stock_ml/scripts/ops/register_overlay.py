"""register_overlay: score the OFFICIAL Stage-2 overlay CAGR for leaderboard runs using the
in-repo unified engine (stock_ml.portfolio) — the replacement for the out-of-repo nh_nav2
yardstick.

Every run is scored under the BOARD REFERENCE CONFIG (K=10, causal, T+2, flat 1-tỷ ADV floor;
see stock_ml.db.overlay_scoring.REFERENCE_CONSTANTS) so the leaderboard ranks SIGNAL quality
under one deploy-style portfolio policy. Per-strategy overrides (K / liquidity / window) are
passed via flags; the same knobs power the live sandbox endpoint.

Writes ONLY the overlay columns of leaderboard_nav (cagr_overlay/maxdd_overlay/overlay_k/
overlay_note/overlay_config_hash) — nh_nav2 nav/t2 columns are left untouched. Idempotent: a
run already scored under the SAME overlay_config_hash is skipped unless --force.

Data context (market panel + NAV price marks) is injected via DuckDBContext; point it at the
market.duckdb + ohlcv.db to score against (defaults to the pinned serving snapshot).

Run:
  python stock_ml/scripts/ops/register_overlay.py --run-id template/foo-abcd1234   # single
  python stock_ml/scripts/ops/register_overlay.py --pinned --limit 50              # batch
  python stock_ml/scripts/ops/register_overlay.py --run-id X --k 6 --liqcol-adv10-ty 5.0
"""

from __future__ import annotations

import argparse
import time

import psycopg2

from stock_ml.db.overlay_scoring import (
    default_context,
    overlay_config_hash,
    pg_dsn,
    reference_config,
    score_overlay,
)

_UPSERT = """
INSERT INTO leaderboard_nav
    (run_id, cagr_overlay, maxdd_overlay, overlay_k, overlay_note, overlay_config_hash, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (run_id) DO UPDATE SET
    cagr_overlay = EXCLUDED.cagr_overlay,
    maxdd_overlay = EXCLUDED.maxdd_overlay,
    overlay_k = EXCLUDED.overlay_k,
    overlay_note = EXCLUDED.overlay_note,
    overlay_config_hash = EXCLUDED.overlay_config_hash,
    computed_at = now()
"""


def fetch_batch(cur, *, pinned_only, limit, run_like, order_by="composite"):
    q = (
        "SELECT r.run_id FROM leaderboard_runs r "
        "LEFT JOIN leaderboard_nav n ON n.run_id = r.run_id "
        "WHERE EXISTS (SELECT 1 FROM run_trades t WHERE t.run_id = r.run_id) "
        "AND EXISTS (SELECT 1 FROM run_signals s WHERE s.run_id = r.run_id)"
    )
    params: list = []
    if pinned_only:
        q += " AND r.state = 'pinned'"
    else:
        q += " AND r.superseded = false"
    if run_like:
        q += " AND r.run_id ILIKE %s"
        params.append(f"%{run_like}%")
    # cagr = the nh_nav2 nav yardstick (cagr_adv); composite = leaderboard rank.
    order_col = "n.cagr_adv" if order_by == "cagr" else "r.composite_score"
    q += f" ORDER BY {order_col} DESC NULLS LAST"
    if limit:
        q += " LIMIT %s"
        params.append(limit)
    cur.execute(q, params)
    return [r[0] for r in cur.fetchall()]


def existing_hash(cur, run_id):
    cur.execute("SELECT overlay_config_hash FROM leaderboard_nav WHERE run_id = %s", (run_id,))
    row = cur.fetchone()
    return row[0] if row else None


def build_note(C):
    floor = (
        f"floor{C.liqcol_adv10_ty:g}ty"
        if C.liqcol_adv10_ty is not None and C.liqcol_adv252_ty == 0.0
        else (f"liqcol{C.liqcol_adv10_ty:g}ty" if C.liqcol_adv10_ty is not None else "nofloor")
    )
    win = "" if C.date_lo == "2020-01-01" else f" from{C.date_lo}"
    return f"stock_ml.portfolio k{C.k} {C.stat_mode} T+{C.tplus} {floor}{win}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-id", default=None, help="score a single run (else batch)")
    ap.add_argument("--base-run", default=None, help="borrow BASE trades from this run_id")
    ap.add_argument("--pinned", action="store_true", help="batch: only pinned runs")
    ap.add_argument("--limit", type=int, default=None, help="batch: top-N by composite")
    ap.add_argument("--run-like", default=None, help="batch: filter run_id substring")
    ap.add_argument(
        "--order-by", choices=("composite", "cagr"), default="composite", help="batch ranking"
    )
    ap.add_argument("--force", action="store_true", help="re-score even if same config_hash")
    # per-strategy overlay overrides (default = board reference config)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--liqcol-adv10-ty", type=float, default=None)
    ap.add_argument("--liqcol-adv252-ty", type=float, default=None)
    ap.add_argument("--date-lo", default=None, help="entry-start window (default 2020-01-01)")
    args = ap.parse_args()

    overrides = {}
    if args.k is not None:
        overrides["k"] = args.k
    if args.liqcol_adv10_ty is not None:
        overrides["liqcol_adv10_ty"] = args.liqcol_adv10_ty
    if args.liqcol_adv252_ty is not None:
        overrides["liqcol_adv252_ty"] = args.liqcol_adv252_ty
    if args.date_lo is not None:
        overrides["date_lo"] = args.date_lo
    C = reference_config(**overrides)
    cfg_hash = overlay_config_hash(C)
    note = build_note(C)

    ctx = default_context()
    con = psycopg2.connect(pg_dsn())
    cur = con.cursor()

    if args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = fetch_batch(
            cur,
            pinned_only=args.pinned,
            limit=args.limit,
            run_like=args.run_like,
            order_by=args.order_by,
        )
    print(f"config_hash={cfg_hash} note='{note}' | runs={len(run_ids)}", flush=True)

    n_ok = n_skip = n_err = 0
    t_start = time.time()
    for i, run_id in enumerate(run_ids, 1):
        if not args.force and existing_hash(cur, run_id) == cfg_hash:
            n_skip += 1
            continue
        t0 = time.time()
        try:
            r = score_overlay(con, run_id, ctx, C, base_run=args.base_run)
        except Exception as exc:  # noqa: BLE001 — one bad run must not kill the batch
            n_err += 1
            print(f"[{i}/{len(run_ids)}] ERR {run_id}: {type(exc).__name__}: {exc}", flush=True)
            con.rollback()
            continue
        cur.execute(_UPSERT, (run_id, r["cagr"], r["maxdd"], C.k, note, cfg_hash))
        con.commit()
        n_ok += 1
        print(
            f"[{i}/{len(run_ids)}] {run_id}: CAGR {r['cagr'] * 100:.1f}% DD {r['maxdd'] * 100:.1f}% "
            f"(n_gated={r['n_gated']}/{r['n_base']}, {time.time() - t0:.1f}s)",
            flush=True,
        )

    print(
        f"XONG: ok={n_ok} skip={n_skip} err={n_err} / {len(run_ids)} trong "
        f"{(time.time() - t_start) / 60:.1f} phut",
        flush=True,
    )
    con.close()


if __name__ == "__main__":
    main()
