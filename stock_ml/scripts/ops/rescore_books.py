"""rescore_books: give every Stage-2 book a strategy key, on the CURRENT declared panel.

Two jobs, both consequences of OVERLAY_IDENTITY_RESTRUCTURE (migration 0035):

1. ``--from-overlay-runs`` — the ``overlay/*`` rows in ``leaderboard_runs`` were a workaround for
   the missing key: one fake run per portfolio strategy. Their ``overlay_note`` carries the label,
   the exact config dict and the base run, so each one is re-scored as a proper ``run_overlay`` row
   ON ITS PARENT and its book written under that key. The old rows are left alone (old links keep
   working); retiring them is a separate, reviewed step.

2. ``--purge-legacy`` — books written before the key exist under the ``legacy`` sentinel. They came
   from an unknown config on an unknown panel, so they cannot be re-keyed, only re-made. Measured
   2026-08-02: every one of them predated the panel its own headline number was scored on. A wrong
   book is worse than no book — the page cannot tell, but a reader will believe it.

Nothing is deleted without ``--purge-legacy``, and that flag honours ``--dry-run``.

  python stock_ml/scripts/ops/rescore_books.py --from-overlay-runs
  python stock_ml/scripts/ops/rescore_books.py --purge-legacy --dry-run
"""

from __future__ import annotations

import argparse
import ast
import re
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
from stock_ml.portfolio import (
    PortfolioConstants,
    build_panel_bundle,
    overlay_key,
    scoring_hash,
)
from stock_ml.portfolio.api import _bundle_key

_DETAIL_TABLES = (
    "run_equity",
    "run_portfolio_daily",
    "run_trades_overlay",
    "run_skipped",
    "run_pending",
)
LEGACY_KEY = "legacy"

# "OVERLAY TIER <label> | … k10 {'tplus': 2, …} | panel … | base=template/foo"
_NOTE_RE = re.compile(
    r"^OVERLAY TIER\s+(?P<label>.*?)\s*\|.*?(?P<cfg>\{.*?\}).*?\|.*?base=(?P<base>\S+)",
    re.DOTALL,
)


def parse_overlay_note(note: str) -> dict | None:
    """Recover (label, config, base_run) from the note the tier registrar wrote.

    ``ast.literal_eval`` not ``eval``: the note is data read back out of the database, and a
    config that can execute is a config that can be an injection.
    """
    m = _NOTE_RE.match(note or "")
    if not m:
        return None
    try:
        cfg = ast.literal_eval(m.group("cfg"))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(cfg, dict):
        return None
    return {"label": m.group("label").strip(), "config": cfg, "base_run": m.group("base").strip()}


def legacy_counts(cur) -> dict[str, int]:
    out = {}
    for tbl in _DETAIL_TABLES:
        cur.execute(f"SELECT count(*) FROM {tbl} WHERE overlay_key=%s", (LEGACY_KEY,))
        out[tbl] = cur.fetchone()[0]
    return out


def from_overlay_runs(con, cur, ctx, panel_fp: str, *, dry_run: bool, force: bool) -> None:
    cur.execute(
        "SELECT r.run_id, r.parent_run_id, n.overlay_note FROM leaderboard_runs r "
        "JOIN leaderboard_nav n ON n.run_id = r.run_id "
        "WHERE r.run_id LIKE 'overlay/%' ORDER BY r.run_id"
    )
    rows = cur.fetchall()
    print(f"{len(rows)} overlay/* run(s) to fold into their parent", flush=True)
    bundles: dict[tuple, dict] = {}
    for i, (old_id, parent, note) in enumerate(rows, 1):
        spec = parse_overlay_note(note)
        if not spec or not parent:
            print(f"[{i}/{len(rows)}] SKIP {old_id}: note not parseable / no parent", flush=True)
            continue
        # PortfolioConstants defaults, NOT reference_config: the tier notes are the serving
        # tiers.yaml overrides and rescore_deploy_tiers.py registered them that way. The
        # difference is not cosmetic — the board reference already pins liqcol_adv252_ty=0.0,
        # so under it "Cân bằng — veto big-quiet" (adv252 default 10.0) would collapse into the
        # plain 5-tỷ floor and two distinct tiers would share ONE key. The dry-run caught exactly
        # that collision; keep this base aligned with serving or the two repos disagree on what a
        # strategy IS.
        C = PortfolioConstants(**spec["config"])
        key = overlay_key(C)
        sh = scoring_hash(C, panel_fp)
        print(
            f"[{i}/{len(rows)}] {old_id} -> {parent} key={key[:8]} '{spec['label']}' "
            f"cfg={spec['config']}",
            flush=True,
        )
        if dry_run:
            continue
        bk = _bundle_key(C)
        if bk not in bundles:
            bundles[bk] = build_panel_bundle(ctx, C)
        t0 = time.time()
        try:
            r = score_overlay(
                con, parent, ctx, C, base_run=spec["base_run"], bundle=bundles[bk],
                emit_pending=True,
            )  # fmt: skip
            persist_overlay(
                con, parent, r, C, sh, panel_fp, note, detail=True,
                label=spec["label"], base_run=spec["base_run"], source_run=old_id,
            )  # fmt: skip
        except Exception as exc:  # noqa: BLE001 — one tier must not kill the batch
            con.rollback()
            print(f"      ERR {type(exc).__name__}: {exc}", flush=True)
            continue
        con.commit()
        print(
            f"      OK CAGR {r['cagr'] * 100:.1f}% DD {r['maxdd'] * 100:.1f}% "
            f"({r['n_gated']}/{r['n_base']} lệnh, {time.time() - t0:.0f}s)",
            flush=True,
        )
    _ = force


def rescore_legacy(con, cur, ctx, panel_fp: str, *, dry_run: bool) -> None:
    """Give every run that still owns a ``legacy`` book a REAL one, under the board reference
    strategy, on the current panel.

    The reference config is not a guess at what produced the old book — that is unknowable. It is
    a named strategy whose provenance the page can state, which is the whole point: the reader
    stops seeing an unlabelled book and starts seeing 'board reference, panel X, scored today'.
    """
    # ``overlay/*`` rows are excluded: they own no signals or base trades of their own (they were
    # placeholders for a strategy), so scoring them would just fail. --from-overlay-runs already
    # moved their content onto the parent; what is left here is only their stale book, and that
    # is --purge-legacy's job.
    cur.execute(
        "SELECT DISTINCT run_id FROM run_equity WHERE overlay_key=%s AND run_id NOT LIKE 'overlay/%%' "
        "ORDER BY run_id",
        (LEGACY_KEY,),
    )
    runs = [r[0] for r in cur.fetchall()]
    C = reference_config()
    key, sh = overlay_key(C), scoring_hash(C, panel_fp)
    print(f"{len(runs)} run có sổ legacy -> chấm lại dưới chiến lược tham chiếu key={key[:8]}",
          flush=True)
    if dry_run:
        for r in runs:
            print(f"  · {r}", flush=True)
        return
    bundle = build_panel_bundle(ctx, C)
    for i, run_id in enumerate(runs, 1):
        t0 = time.time()
        try:
            r = score_overlay(con, run_id, ctx, C, bundle=bundle, emit_pending=True)
            persist_overlay(con, run_id, r, C, sh, panel_fp, "board reference", detail=True,
                            label="Tham chiếu board (K10 · sàn 1 tỷ · T+2 causal)")  # fmt: skip
        except Exception as exc:  # noqa: BLE001 — one run must not kill the batch
            con.rollback()
            print(f"[{i}/{len(runs)}] ERR {run_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        con.commit()
        print(
            f"[{i}/{len(runs)}] {run_id}: CAGR {r['cagr'] * 100:.1f}% DD {r['maxdd'] * 100:.1f}% "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )


def backfill_source(con, cur, *, dry_run: bool) -> None:
    """One-off: fill ``run_overlay.source_run`` for books folded before the column existed.

    Reads the placeholder's own note — the only record of which strategy it was — and writes the
    answer into a column so nothing ever has to parse that string again. After this the note is
    documentation, not a data structure.
    """
    cur.execute(
        "SELECT r.run_id, r.parent_run_id, n.overlay_note FROM leaderboard_runs r "
        "JOIN leaderboard_nav n ON n.run_id = r.run_id "
        "WHERE r.run_id LIKE 'overlay/%' AND r.parent_run_id IS NOT NULL ORDER BY r.run_id"
    )
    n = 0
    for old_id, parent, note in cur.fetchall():
        spec = parse_overlay_note(note)
        if not spec:
            print(f"  SKIP {old_id}: note không đọc được", flush=True)
            continue
        key = overlay_key(PortfolioConstants(**spec["config"]))
        print(f"  {old_id} -> {parent} / {key[:8]}", flush=True)
        if dry_run:
            continue
        cur.execute(
            "UPDATE run_overlay SET source_run=%s WHERE run_id=%s AND overlay_key=%s "
            "AND source_run IS NULL",
            (old_id, parent, key),
        )
        n += cur.rowcount
    if not dry_run:
        con.commit()
        print(f"đã nối {n} sổ về placeholder gốc", flush=True)


def adopt_orphans(con, cur, *, dry_run: bool) -> None:
    """Turn the surviving ``legacy`` books into DECLARED rows instead of a hidden convention.

    After the purge, what remains are books on runs with no BASE trades left (migration 0033 split
    BASE out of OUTPUT-only runs), so they can never be re-scored. Keeping them as bare
    ``overlay_key='legacy'`` rows would leave one shape of book that exists in the detail tables
    but in no index — a special case every reader has to know about, which is the same class of
    implicit knowledge this restructure removes.

    So each gets a real ``run_overlay`` row that says exactly what it is: no config, no panel, no
    metrics. ``panel_fp=NULL`` makes the API's staleness check report *unknown* rather than clean,
    and the picker shows it by name. Nothing is deleted, nothing is invented.
    """
    cur.execute(
        "SELECT DISTINCT run_id FROM run_equity WHERE overlay_key=%s ORDER BY run_id", (LEGACY_KEY,)
    )
    runs = [r[0] for r in cur.fetchall()]
    print(f"{len(runs)} run còn sổ không tái tạo được -> khai báo tường minh", flush=True)
    for r in runs:
        print(f"  · {r}", flush=True)
    if dry_run or not runs:
        return
    for run_id in runs:
        cur.execute(
            "INSERT INTO run_overlay (run_id, overlay_key, label, has_detail, computed_at) "
            "VALUES (%s,%s,%s,true, now()) ON CONFLICT (run_id, overlay_key) DO UPDATE SET "
            "label = EXCLUDED.label, has_detail = true",
            (run_id, LEGACY_KEY, "Sổ cũ — KHÔNG rõ cấu hình/panel, không so được"),
        )
    con.commit()
    print(f"đã khai báo {len(runs)} sổ cũ (config/panel = NULL, cố ý)", flush=True)


def purge_legacy(con, cur, *, dry_run: bool) -> None:
    """Drop pre-restructure books — but ONLY where a real one now stands in their place.

    A legacy row whose run has since been scored under a named strategy is a pure duplicate:
    deleting it removes an unlabelled copy of something we can now name. A legacy row on a run
    with NO replacement is different — some runs (OUTPUT-only ones whose BASE was split out by
    migration 0033) have 0 base trades and can never be re-scored, so purging theirs would
    destroy the only record that exists. Those are reported and kept; the page already marks them
    'chưa khai chiến lược' in red, which is honest. Deleting the irreplaceable to tidy a table is
    not a call this script gets to make.
    """
    before = legacy_counts(cur)
    total = sum(before.values())
    print(f"legacy rows: {before} (tổng {total})", flush=True)
    if not total:
        return
    cur.execute(
        "SELECT DISTINCT e.run_id FROM run_equity e WHERE e.overlay_key=%s AND EXISTS ("
        "  SELECT 1 FROM run_overlay o WHERE o.run_id = e.run_id AND o.has_detail)"
        " ORDER BY 1",
        (LEGACY_KEY,),
    )
    replaced = [r[0] for r in cur.fetchall()]
    cur.execute(
        "SELECT DISTINCT e.run_id FROM run_equity e WHERE e.overlay_key=%s AND NOT EXISTS ("
        "  SELECT 1 FROM run_overlay o WHERE o.run_id = e.run_id AND o.has_detail)"
        " ORDER BY 1",
        (LEGACY_KEY,),
    )
    orphan = [r[0] for r in cur.fetchall()]
    # An overlay/* placeholder is "replaced" when its content now lives on its PARENT under the
    # key its own note describes — the row moved, it did not disappear. Verify that key exists
    # with a book rather than assuming, so a half-finished fold cannot be mistaken for a done one.
    still_orphan = []
    for run_id in orphan:
        if not run_id.startswith("overlay/"):
            still_orphan.append(run_id)
            continue
        cur.execute(
            "SELECT r.parent_run_id, n.overlay_note FROM leaderboard_runs r "
            "JOIN leaderboard_nav n ON n.run_id = r.run_id WHERE r.run_id=%s",
            (run_id,),
        )
        row = cur.fetchone()
        spec = parse_overlay_note(row[1]) if row else None
        if not row or not row[0] or not spec:
            still_orphan.append(run_id)
            continue
        key = overlay_key(PortfolioConstants(**spec["config"]))
        cur.execute(
            "SELECT has_detail FROM run_overlay WHERE run_id=%s AND overlay_key=%s",
            (row[0], key),
        )
        moved = cur.fetchone()
        (replaced if moved and moved[0] else still_orphan).append(run_id)
    orphan = still_orphan
    print(f"  {len(replaced)} run đã có sổ thay thế -> xoá được", flush=True)
    print(f"  {len(orphan)} run KHÔNG tái tạo được -> GIỮ, chờ người quyết:", flush=True)
    for r in orphan:
        print(f"      · {r}", flush=True)
    if dry_run or not replaced:
        print("--dry-run: không xoá gì" if dry_run else "không có gì để xoá", flush=True)
        return
    n = 0
    for tbl in _DETAIL_TABLES:
        cur.execute(
            f"DELETE FROM {tbl} WHERE overlay_key=%s AND run_id = ANY(%s)", (LEGACY_KEY, replaced)
        )
        n += cur.rowcount
    # ...and the index row itself, else the book survives in run_overlay as a has_detail=true
    # row with zero detail — a ghost the danh-mục picker still lists.
    cur.execute(
        "DELETE FROM run_overlay WHERE overlay_key=%s AND run_id = ANY(%s)", (LEGACY_KEY, replaced)
    )
    n += cur.rowcount
    con.commit()
    print(f"đã xoá {n} dòng sổ cũ trên {len(replaced)} run (đều đã có bản khai đầy đủ)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-overlay-runs", action="store_true",
                    help="fold overlay/* runs into run_overlay rows on their parent (+ detail)")
    ap.add_argument("--rescore-legacy", action="store_true",
                    help="re-score every run that still owns a pre-restructure book, under the "
                         "board reference strategy on the current panel")
    ap.add_argument("--backfill-source", action="store_true",
                    help="fill run_overlay.source_run for books folded before the column existed")
    ap.add_argument("--adopt-orphans", action="store_true",
                    help="declare the un-rescorable books as explicit run_overlay rows")
    ap.add_argument("--purge-legacy", action="store_true",
                    help="delete pre-restructure books (unknown config/panel)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (a.from_overlay_runs or a.rescore_legacy or a.purge_legacy or a.adopt_orphans
            or a.backfill_source):
        ap.error("chọn --from-overlay-runs / --rescore-legacy / --purge-legacy / "
                 "--adopt-orphans / --backfill-source")

    con = psycopg2.connect(pg_dsn())
    cur = con.cursor()
    if a.from_overlay_runs or a.rescore_legacy:
        ctx = default_context()  # fail-loud on panel artifact drift
        fp = panel_fingerprint()
        print(f"panel {fp}", flush=True)
    if a.from_overlay_runs:
        from_overlay_runs(con, cur, ctx, fp, dry_run=a.dry_run, force=a.force)
    if a.rescore_legacy:
        rescore_legacy(con, cur, ctx, fp, dry_run=a.dry_run)
    if a.purge_legacy:
        purge_legacy(con, cur, dry_run=a.dry_run)
    if a.backfill_source:
        backfill_source(con, cur, dry_run=a.dry_run)
    if a.adopt_orphans:
        adopt_orphans(con, cur, dry_run=a.dry_run)
    con.close()


if __name__ == "__main__":
    main()
