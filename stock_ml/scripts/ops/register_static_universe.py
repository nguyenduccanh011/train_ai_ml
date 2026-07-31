"""Register the two STATIC bundle universes as versioned + LOCKED sets (ENGINE_UPGRADE Phase 1d / §9.9).

The 3 static bundles (wavestruct / consw20 / velov) shipped ``config.json`` slug ``vn_stock_default``
(61 symbols) while actually running **150** symbols — the config field was untruthful, so a config could
not reproduce the model (breaks object-model goal #2). This pins the two distinct ground-truth 150-lists
(snapshotted from bundle manifests into ``data/static_universe_sets.json``) as versioned, **LOCKED**
universe sets so configs can reference a reproducible identity:

  - ``vn_top150_adv``  — wavestruct ≡ consw20 (identical 150)
  - ``vn_univ150c``    — velov (different 150, shares 109/150 with vn_top150_adv)

This is a **0-number-change** metadata registration (it records the existing universe, changes nothing
that runs). Idempotent (re-run verifies sha + skips existing). Reversible: sets are locked, so
``delete()`` refuses them — to remove, ``update_meta(slug, is_locked=False)`` then delete.

Usage:  python stock_ml/scripts/ops/register_static_universe.py [--dry-run]
        (needs DATABASE_URL to the train Postgres)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, so `stock_ml.*` imports

DATA = Path(__file__).resolve().parent / "data" / "static_universe_sets.json"


def _sha(symbols: list[str]) -> str:
    return hashlib.sha256((",".join(sorted(set(symbols)))).encode()).hexdigest()


async def _run(dry_run: bool) -> int:
    from stock_ml.db.engine import AsyncSessionLocal
    from stock_ml.db.repositories.universe_repo import UniverseRepository

    spec = json.loads(DATA.read_text(encoding="utf-8"))
    rc = 0
    async with AsyncSessionLocal() as session:
        repo = UniverseRepository(session)
        for s in spec["sets"]:
            slug, want = s["slug"], s["symbols_sha256"]
            if _sha(s["symbols"]) != want:
                print(f"[FAIL] {slug}: data-file symbols sha != declared (corrupt data file)")
                return 1

            existing = await repo.get_by_slug_all(slug)
            if existing is not None:
                snap = await repo.get_version_snapshot(slug, existing.version)
                cur = _sha([d["symbol"] for d in snap]) if snap else None
                ok = cur == want
                print(f"[skip] {slug}: exists (v{existing.version} locked={existing.is_locked} "
                      f"sha={'OK' if ok else 'DIFF'})")
                if not ok:
                    print(f"       WARNING existing sha {cur} != expected {want}")
                    rc = 2
                continue

            if dry_run:
                print(f"[dry-run] would create+lock {slug}: {s['symbol_count']} symbols sha={want[:16]}")
                continue

            await repo.create(
                slug=slug, name=s["name"], market=s["market"],
                description=f"{s['derivation']} · pinned nguyên trạng (ENGINE_UPGRADE §9.9) · "
                            f"source {s['source_bundle']}",
                symbols=[{"symbol": x} for x in s["symbols"]],
            )
            await repo.update_meta(slug, is_locked=True, notes=f"symbols_sha256={want}")
            await session.commit()

            back = await repo.get_by_slug_all(slug)
            snap = await repo.get_version_snapshot(slug, back.version)
            ok = bool(snap) and _sha([d["symbol"] for d in snap]) == want and back.is_locked
            print(f"[create] {slug}: {s['symbol_count']} symbols v{back.version} "
                  f"locked={back.is_locked} sha={'OK' if ok else 'FAIL'}")
            if not ok:
                rc = 1
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="Register static bundle universes as locked versioned sets")
    p.add_argument("--dry-run", action="store_true", help="show what would happen, write nothing")
    return asyncio.run(_run(p.parse_args().dry_run))


if __name__ == "__main__":
    sys.exit(main())
