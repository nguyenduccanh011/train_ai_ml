"""register_overlay: score the OFFICIAL Stage-2 overlay CAGR for leaderboard runs using the
in-repo unified engine (stock_ml.portfolio) — the replacement for the out-of-repo nh_nav2
yardstick.

Every run is scored under the BOARD REFERENCE CONFIG (K=10, causal, T+2, flat 1-tỷ ADV floor;
see stock_ml.db.overlay_scoring.REFERENCE_CONSTANTS) so the leaderboard ranks SIGNAL quality
under one deploy-style portfolio policy. Per-strategy overrides (K / liquidity / window) are
passed via flags; the same knobs power the live sandbox endpoint.

Writes one ``run_overlay`` row per (run, strategy) — keyed by ``overlay_key`` (the config
identity), so scoring a second strategy ADDS a book instead of deleting the first. The
``leaderboard_nav`` overlay columns are written ONLY for the board reference strategy, which is
what keeps the ranking a comparison of SIGNAL quality rather than a mix of portfolio policies.
Idempotent per strategy: same ``scoring_hash`` (config + declared panel) is skipped unless
--force, and a prior metrics-only score never satisfies a --detail request.

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

from stock_ml.db.overlay_persist import persist_overlay
from stock_ml.db.overlay_scoring import (
    default_context,
    panel_fingerprint,
    pg_dsn,
    reference_config,
    score_overlay,
)
from stock_ml.portfolio import build_panel_bundle, overlay_key, scoring_hash


def fetch_batch(cur, *, pinned_only, limit, run_like, order_by="composite", only_missing=False):
    q = (
        "SELECT r.run_id FROM leaderboard_runs r "
        "LEFT JOIN leaderboard_nav n ON n.run_id = r.run_id "
        "WHERE EXISTS (SELECT 1 FROM run_trades t WHERE t.run_id = r.run_id) "
        "AND EXISTS (SELECT 1 FROM run_signals s WHERE s.run_id = r.run_id)"
    )
    params: list = []
    if only_missing:
        # skip runs that already carry ANY overlay (reference OR official per-strategy) so a
        # broad backfill fills only the genuinely-empty runs and never churns/errors on the
        # promoted OUTPUT runs (which keep their own config).
        q += " AND n.cagr_overlay IS NULL"
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


def already_scored(cur, run_id: str, key: str, cfg_hash: str, *, detail: bool) -> bool:
    """Has THIS strategy of THIS run already been scored on THIS panel?

    Scoped by ``(run_id, overlay_key)`` — the old check read leaderboard_nav, which holds one row
    per run, so scoring a second strategy either skipped wrongly or overwrote the first. And a
    prior METRICS-ONLY score must not satisfy a ``--detail`` request: that is precisely how the
    board ended up with fresh numbers sitting on top of month-old books.
    """
    cur.execute(
        "SELECT scoring_hash, has_detail FROM run_overlay WHERE run_id=%s AND overlay_key=%s",
        (run_id, key),
    )
    row = cur.fetchone()
    if not row or row[0] != cfg_hash:
        return False
    return bool(row[1]) or not detail


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
    ap.add_argument(
        "--only-missing", action="store_true", help="batch: skip runs that already have any overlay"
    )
    ap.add_argument("--force", action="store_true", help="re-score even if same config_hash")
    # detail tab: write the 5 detail tables (equity/holdings/trades/skipped/pending) as well as
    # metrics. Default = single-run / pinned scoring gets the full tab; a broad board batch stays
    # metrics-only (keeps run_portfolio_daily from exploding). Override either way.
    ap.add_argument("--detail", dest="detail", action="store_true", default=None,
                    help="force-write the full danh-mục detail tables")
    ap.add_argument("--no-detail", dest="detail", action="store_false",
                    help="force metrics-only (skip detail tables)")
    # per-strategy overlay overrides (default = board reference config)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--liqcol-adv10-ty", type=float, default=None)
    ap.add_argument("--liqcol-adv252-ty", type=float, default=None)
    ap.add_argument("--date-lo", default=None, help="entry-start window (default 2020-01-01)")
    ap.add_argument("--label", default=None,
                    help="human name of this strategy in the danh-mục picker (default: the note)")
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
    ctx = default_context()  # asserts the declared-panel artifact identity (fail-loud)
    panel_fp = panel_fingerprint()
    key = overlay_key(C)  # WHICH strategy (config only) — the book's primary key
    cfg_hash = scoring_hash(C, panel_fp)  # WHICH scoring (config + panel) — idempotency
    note = build_note(C)
    label = args.label or note

    # Build the market panel ONCE and reuse across the whole batch (all runs share this C) — the
    # cross-sectional rank is the ~60s/run cost; a batch of thousands is infeasible without this.
    bundle = build_panel_bundle(ctx, C)
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
            only_missing=args.only_missing,
        )
    # default: single run or a pinned batch gets the full detail tab; a broad board batch is
    # metrics-only. --detail / --no-detail override.
    detail = args.detail if args.detail is not None else (bool(args.run_id) or args.pinned)
    print(
        f"overlay_key={key} scoring_hash={cfg_hash} label='{label}' | runs={len(run_ids)} | "
        f"detail={detail} | panel={panel_fp}",
        flush=True,
    )

    n_ok = n_skip = n_err = 0
    t_start = time.time()
    for i, run_id in enumerate(run_ids, 1):
        if not args.force and already_scored(cur, run_id, key, cfg_hash, detail=detail):
            n_skip += 1
            continue
        t0 = time.time()
        try:
            r = score_overlay(
                con, run_id, ctx, C, base_run=args.base_run, bundle=bundle, emit_pending=detail
            )
            persist_overlay(con, run_id, r, C, cfg_hash, panel_fp, note, detail=detail,
                            label=label, base_run=args.base_run)
        except Exception as exc:  # noqa: BLE001 — one bad run must not kill the batch
            n_err += 1
            print(f"[{i}/{len(run_ids)}] ERR {run_id}: {type(exc).__name__}: {exc}", flush=True)
            con.rollback()
            continue
        con.commit()
        n_ok += 1
        off = f" OFFPANEL={r['offpanel_frac'] * 100:.0f}%" if r["offpanel_frac"] > 0 else ""
        print(
            f"[{i}/{len(run_ids)}] {run_id}: CAGR {r['cagr'] * 100:.1f}% DD {r['maxdd'] * 100:.1f}% "
            f"(n_gated={r['n_gated']}/{r['n_base']}, conv_miss={r['conv_miss_frac'] * 100:.0f}%{off}, "
            f"{time.time() - t0:.1f}s)",
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
