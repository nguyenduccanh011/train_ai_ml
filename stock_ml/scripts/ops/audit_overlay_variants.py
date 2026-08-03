"""audit_overlay_variants: which "runs" are really just a PORTFOLIO strategy of another run?

Before the (run_id, overlay_key) restructure the only way to keep two portfolio policies side by
side was to register each as its own run — so families like ``x2_struct_to_k10_cs5ma50_t2ret7g``
carry a full duplicate of their parent's Stage-1 output. Measured 2026-08-02: 27 such rows, each
holding its own 96.951 ``run_signals`` rows; the table is 96 GB.

REPORT ONLY. Two runs sharing a signal set is strong evidence, but ONLY identical BASE trades
prove the difference is portfolio-side: names like ``_r7earlycut`` / ``_gtos`` also encode engine
knobs that change the base backtest, and folding one of those into an overlay would silently
rewrite what the run means. Measured counter-example: ``x2_struct_to`` has 2485 base trades vs
``…_k10_cs5ma50``'s 2491 — same signals, DIFFERENT base. So the decision stays with a human.

  python stock_ml/scripts/ops/audit_overlay_variants.py --prefix x2_struct_to
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import psycopg2

from stock_ml.db.overlay_scoring import pg_dsn

_SIG_FP = """
SELECT run_id, count(*), md5(string_agg(symbol || ':' || date || ':' || signal, ',' ORDER BY symbol, date))
FROM run_signals WHERE run_id = ANY(%s) GROUP BY run_id
"""
_BASE_FP = """
SELECT run_id, count(*), md5(string_agg(
    symbol || ':' || entry_date || ':' || coalesce(exit_date::text,'-') || ':' ||
    round(entry_price::numeric, 4), ',' ORDER BY symbol, entry_date))
FROM run_trades WHERE run_id = ANY(%s) GROUP BY run_id
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", default="", help="only runs whose run_id contains this")
    a = ap.parse_args()

    con = psycopg2.connect(pg_dsn())
    cur = con.cursor()
    cur.execute(
        "SELECT run_id FROM leaderboard_runs WHERE run_id ILIKE %s ORDER BY run_id",
        (f"%{a.prefix}%",),
    )
    runs = [r[0] for r in cur.fetchall()]
    if not runs:
        print("không có run nào khớp")
        return
    cur.execute(_SIG_FP, (runs,))
    sig = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    cur.execute(_BASE_FP, (runs,))
    base = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    by_sig: dict[str, list[str]] = defaultdict(list)
    for rid, (_n, fp) in sig.items():
        by_sig[fp].append(rid)

    print(f"{len(runs)} run | {len(by_sig)} bộ tín hiệu khác nhau\n")
    foldable, distinct, no_data = [], [], []
    for fp, group in sorted(by_sig.items(), key=lambda kv: -len(kv[1])):
        if len(group) < 2:
            continue
        print(f"— nhóm {fp[:10]}: {len(group)} run cùng tín hiệu ({sig[group[0]][0]} dòng/run)")
        by_base: dict[str, list[str]] = defaultdict(list)
        for rid in sorted(group):
            b = base.get(rid)
            by_base[b[1] if b else "NO_BASE"].append(rid)
        for bfp, brun in by_base.items():
            tag = "NO BASE TRADES" if bfp == "NO_BASE" else f"base {bfp[:10]}"
            n = base.get(brun[0], (0,))[0]
            print(f"    {tag} ({n} lệnh): {len(brun)} run")
            for rid in brun:
                print(f"      · {rid}")
            if bfp == "NO_BASE":
                no_data.extend(brun)
            elif len(brun) > 1:
                foldable.append(brun)   # same signals AND same base -> pure portfolio variants
            else:
                distinct.extend(brun)
        print()

    n_fold = sum(len(g) - 1 for g in foldable)
    print(f"KẾT LUẬN: {n_fold} run có thể gộp thành overlay (cùng tín hiệu VÀ cùng base trades)")
    for g in foldable:
        print(f"  giữ {g[0]}  <- gộp {g[1:]}")
    print(f"{len(distinct)} run KHÁC base ⇒ là run thật, KHÔNG gộp: {distinct[:8]}")
    if no_data:
        print(f"{len(no_data)} run không có base trades (mồ côi, xem lại riêng): {no_data[:8]}")
    con.close()


if __name__ == "__main__":
    main()
