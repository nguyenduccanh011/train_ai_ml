from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from src.leaderboard.fairness import annotate_rows, load_config, resolve_baseline
from src.leaderboard.loader import run_dir_to_row
from src.leaderboard.schema import LeaderboardRow, export_json_schema

LEADERBOARD_JSON = "leaderboard.json"
LEADERBOARD_CSV = "leaderboard.csv"
SUMMARY_JSON = "summary.json"
INDEX_JSON = "index.json"
SCHEMA_JSON = "schema.json"


def discover_run_dirs(experiments_dir: str | Path) -> list[Path]:
    root = Path(experiments_dir)
    return sorted(path.parent for path in root.glob("**/ranking_row.json"))


def rebuild_leaderboard(
    experiments_dir: str | Path, output_dir: str | Path
) -> list[LeaderboardRow]:
    rows = _prepare_rows(run_dir_to_row(run_dir) for run_dir in discover_run_dirs(experiments_dir))
    write_outputs(rows, output_dir)
    return rows


def append_or_update(
    run_dir: str | Path, output_dir: str | Path, *, bundle: str | None = None
) -> LeaderboardRow:
    out_dir = Path(output_dir)
    rows = _read_rows(out_dir / LEADERBOARD_JSON)
    row = run_dir_to_row(run_dir, bundle=bundle)
    rows = _prepare_rows([existing for existing in rows if existing.run_id != row.run_id] + [row])
    write_outputs(rows, out_dir)
    return row


def validate_leaderboard(path: str | Path) -> list[LeaderboardRow]:
    return _prepare_rows(_read_rows(Path(path)))


def write_outputs(rows: list[LeaderboardRow], output_dir: str | Path) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _atomic_write_text(out_dir / LEADERBOARD_JSON, _rows_json(rows))
    _write_csv(rows, out_dir / LEADERBOARD_CSV)
    _atomic_write_text(
        out_dir / SUMMARY_JSON, json.dumps(_summary(rows), indent=2, ensure_ascii=False) + "\n"
    )
    _atomic_write_text(
        out_dir / INDEX_JSON, json.dumps(_index(rows), indent=2, ensure_ascii=False) + "\n"
    )
    _atomic_write_text(
        out_dir / SCHEMA_JSON, json.dumps(export_json_schema(), indent=2, ensure_ascii=False) + "\n"
    )


def _prepare_rows(rows: Iterable[LeaderboardRow]) -> list[LeaderboardRow]:
    prepared = [row.model_copy(update={"superseded": False}) for row in rows]
    _check_duplicate_run_ids(prepared)
    latest_by_name: dict[tuple[str, str], str] = {}
    for row in prepared:
        key = (row.bundle, row.run_name)
        if key not in latest_by_name or row.generated_at > latest_by_name[key]:
            latest_by_name[key] = row.generated_at
    prepared = [
        row.model_copy(
            update={"superseded": row.generated_at < latest_by_name[(row.bundle, row.run_name)]}
        )
        for row in prepared
    ]
    prepared = sorted(prepared, key=lambda row: row.composite_score, reverse=True)
    return annotate_rows(prepared, resolve_baseline(prepared, load_config()))


def _check_duplicate_run_ids(rows: Iterable[LeaderboardRow]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row.run_id in seen:
            duplicates.append(row.run_id)
        seen.add(row.run_id)
    if duplicates:
        raise ValueError(f"duplicate run_id: {', '.join(sorted(set(duplicates)))}")


def _read_rows(path: Path) -> list[LeaderboardRow]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [LeaderboardRow(**row) for row in data]


def _rows_json(rows: list[LeaderboardRow]) -> str:
    return (
        json.dumps(
            [row.model_dump(mode="json", exclude_none=True) for row in rows],
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def _write_csv(rows: list[LeaderboardRow], path: Path) -> None:
    fieldnames = [
        "run_id",
        "bundle",
        "run_name",
        "config_hash",
        "generated_at",
        "superseded",
        "strategy",
        "feature_set",
        "entry_model",
        "exit_model_type",
        "exit_model_enabled",
        "target_type",
        "target_forward_window",
        "target_gain_threshold",
        "target_loss_threshold",
        "trades",
        "wr",
        "avg_pnl",
        "total_pnl",
        "pf",
        "avg_hold",
        "sharpe",
        "max_drawdown",
        "mdd_per_symbol",
        "yearly_consistency",
        "composite_score",
        "score_mode",
        "n_symbols",
        "first_test_year",
        "last_test_year",
        "cost_commission",
        "cost_tax",
        "cost_slippage",
        "fairness_group_key",
        "is_baseline",
        "same_symbols_as_baseline",
        "same_window_as_baseline",
        "same_cost_as_baseline",
        "same_target_as_baseline",
        "warnings",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_row(row))
    os.replace(tmp_path, path)


def _flatten_row(row: LeaderboardRow) -> dict[str, object]:
    data = row.model_dump(mode="json")
    target = data.pop("target")
    cost = data.pop("cost_profile")
    warnings = data.pop("warnings")
    data.update(
        {
            "target_type": target["type"],
            "target_forward_window": target["forward_window"],
            "target_gain_threshold": target["gain_threshold"],
            "target_loss_threshold": target["loss_threshold"],
            "cost_commission": cost["commission"],
            "cost_tax": cost["tax"],
            "cost_slippage": cost["slippage"],
            "warnings": " | ".join(warnings),
        }
    )
    return data


def _summary(rows: list[LeaderboardRow]) -> dict[str, object]:
    groups: dict[str, list[LeaderboardRow]] = defaultdict(list)
    for row in rows:
        groups[row.fairness_group_key].append(row)
    baseline = next((row for row in rows if row.is_baseline), None)
    return {
        "row_count": len(rows),
        "active_row_count": sum(not row.superseded for row in rows),
        "fairness_group_count": len(groups),
        "baseline_run_id": baseline.run_id if baseline else None,
        "baseline_fairness_group_key": baseline.fairness_group_key if baseline else None,
        "top_3_per_fairness_group": {
            key: [
                row.run_id
                for row in sorted(group, key=lambda item: item.composite_score, reverse=True)[:3]
            ]
            for key, group in sorted(groups.items())
        },
    }


def _index(rows: list[LeaderboardRow]) -> dict[str, int]:
    return {row.run_id: idx for idx, row in enumerate(rows)}


def _atomic_write_text(path: Path, content: str) -> None:
    import time

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    for delay in (0.1, 0.25, 0.5, 1.0, 2.0):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(tmp_path, path)
