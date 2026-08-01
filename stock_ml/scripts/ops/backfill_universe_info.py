"""Backfill universe_slug/version for existing leaderboard runs (Phase 2b).

Updates runs that have universe_slug=NULL by:
1. Finding the config file from the run_id
2. Parsing universe.slug from config
3. Querying current version from DB
4. Updating leaderboard_runs with slug and version

Usage:
    python stock_ml/scripts/backfill_universe_info.py
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def backfill_universe_info():
    """Backfill universe info for runs missing it."""
    from stock_ml.db.engine import sync_engine
    from stock_ml.db.models.run import LeaderboardRunModel
    from stock_ml.db.repositories.universe_repo import UniverseRepository
    from sqlalchemy.orm import Session

    print("Backfilling universe_slug/version for leaderboard runs...")
    print()

    # Find all runs with NULL universe_slug
    with Session(sync_engine) as session:
        runs_to_update = (
            session.query(LeaderboardRunModel)
            .filter(LeaderboardRunModel.universe_slug.is_(None))
            .all()
        )

        print(f"Found {len(runs_to_update)} runs with NULL universe_slug")
        print()

        updated_count = 0
        no_config_count = 0
        no_universe_count = 0

        for run in runs_to_update:
            # Extract bundle from run_id (format: {bundle}/{run_name}#{hash})
            parts = run.run_id.split("/")
            if len(parts) < 2:
                no_config_count += 1
                continue

            bundle = parts[0]

            # Try to find config file in results/
            config_paths = [
                REPO_ROOT / "results" / bundle / run.run_name / "config.resolved.yaml",
                REPO_ROOT / "stock_ml" / "config" / "experiments" / "done" / f"{run.run_name}.yaml",
                REPO_ROOT
                / "stock_ml"
                / "config"
                / "experiments"
                / "failed"
                / f"{run.run_name}.yaml",
                REPO_ROOT
                / "stock_ml"
                / "config"
                / "experiments"
                / "pending"
                / f"{run.run_name}.yaml",
            ]

            config = None
            for path in config_paths:
                if path.exists():
                    try:
                        with open(path) as f:
                            config = yaml.safe_load(f)
                        break
                    except Exception:
                        pass

            if not config:
                no_config_count += 1
                continue

            # Parse universe slug
            universe_config = config.get("universe")
            if not universe_config or not isinstance(universe_config, dict):
                no_universe_count += 1
                continue

            slug = universe_config.get("slug")
            if not slug:
                no_universe_count += 1
                continue

            # Query current version
            repo = UniverseRepository(session)
            u = repo.get_by_slug_all(slug)  # Include soft-deleted

            if not u:
                no_universe_count += 1
                continue

            # Update run
            run.universe_slug = slug
            run.universe_version = u.version

            notes = run.metadata_notes or ""
            if "[backfilled_universe:" not in notes:
                run.metadata_notes = f"{notes}[backfilled_universe: {slug}@v{u.version}]".strip()

            updated_count += 1
            print(f"  ✓ {run.run_name}: {slug}@v{u.version}")

        session.commit()
        print()
        print("Summary:")
        print(f"  Updated: {updated_count}")
        print(f"  No config file: {no_config_count}")
        print(f"  No universe in config: {no_universe_count}")
        print(
            f"  Total: {updated_count + no_config_count + no_universe_count}/{len(runs_to_update)}"
        )

    return updated_count


if __name__ == "__main__":
    try:
        count = backfill_universe_info()
        sys.exit(0 if count >= 0 else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
