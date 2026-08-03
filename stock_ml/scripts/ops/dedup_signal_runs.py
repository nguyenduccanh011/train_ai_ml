"""dedup_signal_runs: runs that differ only in PORTFOLIO POLICY stop pretending to be models.

Before ``(run_id, overlay_key)`` existed, the only way to hold two portfolio policies side by side
was to register each as its own run. So the board lists 12 rows named ``x2_struct_to_k10c2m005``,
``…_k16preempt``, ``…_t2ret7g`` … that share ONE signal set and ONE set of BASE trades. Scored
under the same config they return the same number (measured 2026-08-03: all 12 → CAGR 100.6%),
which is the definition of "not a different model".

RELABEL, DO NOT DELETE. Two reasons:
  · Their portfolio policy exists only in the run NAME. It was never stored, and their old books
    are gone, so it cannot be recovered — inventing a config to "fold them properly" would
    fabricate provenance, which is the failure this whole restructure removes.
  · The storage argument does not hold: the family is 2.6M of 368M ``run_signals`` rows (0.71%).
    Deleting irreplaceable rows to reclaim 0.7 GB of 96 GB is a bad trade.

What it does instead: point each duplicate at its canonical run (``parent_run_id``) and move it to
``state='retired'``. Two mechanisms that already exist, no new concept:
  · the dashboard hides ``state === 'retired'`` from the ranking (js/app.js), and a human can later
    purge retired rows deliberately through the existing bulk-delete path;
  · model-details reads Stage-1 from ``parent_run_id`` and labels the row a variant, so the
    duplicate keeps working as a link instead of 404-ing.

NOT ``superseded``: that column is DERIVED by the aggregator ("an older run of the same name") and
recomputed on every rebuild, so writing a different meaning into it is both a semantic overload and
silently reverted on the next aggregation.

Anyone who wants "K=16 preempt" back registers it as a NAMED overlay on the canonical run — which
is now possible, and is the whole point.

  python stock_ml/scripts/ops/dedup_signal_runs.py --prefix x2_struct_to          # dry-run
  python stock_ml/scripts/ops/dedup_signal_runs.py --prefix x2_struct_to --apply
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import psycopg2

from stock_ml.db.overlay_scoring import pg_dsn

_SIG_FP = """
SELECT run_id, md5(string_agg(symbol || ':' || date || ':' || signal, ',' ORDER BY symbol, date))
FROM run_signals WHERE run_id = ANY(%s) GROUP BY run_id
"""
_BASE_FP = """
SELECT run_id, md5(string_agg(
    symbol || ':' || entry_date || ':' || coalesce(exit_date::text,'-') || ':' ||
    round(entry_price::numeric, 4), ',' ORDER BY symbol, entry_date))
FROM run_trades WHERE run_id = ANY(%s) GROUP BY run_id
"""


def pick_canonical(cur, group: list[str]) -> str:
    """The run the others point at: a pinned one if the group has one, else the shortest name.

    Shortest name is not arbitrary — these families are built by suffixing a policy onto a base
    name, so the shortest is the one without a policy in it.
    """
    cur.execute(
        "SELECT run_id FROM leaderboard_runs WHERE run_id = ANY(%s) AND state='pinned'", (group,)
    )
    pinned = [r[0] for r in cur.fetchall()]
    return sorted(pinned or group, key=lambda r: (len(r), r))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True, help="only runs whose run_id contains this")
    ap.add_argument("--apply", action="store_true", help="write (default: report only)")
    a = ap.parse_args()

    con = psycopg2.connect(pg_dsn())
    cur = con.cursor()
    cur.execute(
        "SELECT run_id FROM leaderboard_runs WHERE run_id ILIKE %s AND state <> 'retired' "
        "AND superseded = false ORDER BY run_id",
        (f"%{a.prefix}%",),
    )
    runs = [r[0] for r in cur.fetchall()]
    if not runs:
        print("không có run nào khớp (hoặc đã superseded)")
        return
    cur.execute(_SIG_FP, (runs,))
    sig = dict(cur.fetchall())
    cur.execute(_BASE_FP, (runs,))
    base = dict(cur.fetchall())

    groups: dict[tuple, list[str]] = defaultdict(list)
    for rid in runs:
        if rid in sig and rid in base:  # a run with no base trades is not a duplicate of anything
            groups[(sig[rid], base[rid])].append(rid)

    plan: list[tuple[str, str]] = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        canon = pick_canonical(cur, sorted(group))
        print(f"\nnhóm {key[0][:8]}/{key[1][:8]}: {len(group)} run trùng CẢ tín hiệu LẪN base")
        print(f"  giữ làm gốc: {canon}")
        for rid in sorted(group):
            if rid == canon:
                continue
            plan.append((rid, canon))
            print(f"    -> {rid}")

    if not plan:
        print("\nkhông có gì để gộp")
        return
    print(f"\n{len(plan)} run sẽ trỏ về gốc + chuyển state=retired (KHÔNG xoá dòng nào)")
    if not a.apply:
        print("(dry-run — thêm --apply để ghi)")
        return
    for rid, canon in plan:
        cur.execute(
            "UPDATE leaderboard_runs SET parent_run_id = %s, state = 'retired' WHERE run_id = %s",
            (canon, rid),
        )
    con.commit()
    print(f"đã cập nhật {len(plan)} run")
    con.close()


if __name__ == "__main__":
    main()
