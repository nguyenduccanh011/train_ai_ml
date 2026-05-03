import json
import shutil
from pathlib import Path

import pytest
from src.leaderboard.aggregator import (
    INDEX_JSON,
    LEADERBOARD_CSV,
    LEADERBOARD_JSON,
    SCHEMA_JSON,
    SUMMARY_JSON,
    append_or_update,
    rebuild_leaderboard,
    validate_leaderboard,
)
from src.leaderboard.hooks import append_or_update as hook_append_or_update

FIXTURES = Path(__file__).parent / "fixtures"


def _copy_fixture(src_name: str, dest: Path) -> None:
    shutil.copytree(FIXTURES / src_name, dest)


def test_rebuild_writes_all_output_files(tmp_path):
    experiments = tmp_path / "experiments"
    _copy_fixture("rule_run", experiments / "bundle_a" / "rule_run")
    _copy_fixture("v22_run", experiments / "bundle_b" / "v22_run")

    rows = rebuild_leaderboard(experiments, tmp_path / "leaderboard")

    assert len(rows) == 2
    assert [row.composite_score for row in rows] == sorted(
        [row.composite_score for row in rows], reverse=True
    )
    output_dir = tmp_path / "leaderboard"
    for filename in [LEADERBOARD_JSON, LEADERBOARD_CSV, SUMMARY_JSON, INDEX_JSON, SCHEMA_JSON]:
        assert (output_dir / filename).exists()

    data = json.loads((output_dir / LEADERBOARD_JSON).read_text(encoding="utf-8"))
    summary = json.loads((output_dir / SUMMARY_JSON).read_text(encoding="utf-8"))
    index = json.loads((output_dir / INDEX_JSON).read_text(encoding="utf-8"))

    assert len(data) == 2
    assert summary["row_count"] == 2
    assert set(index) == {row["run_id"] for row in data}


def test_append_or_update_replaces_same_run_id(tmp_path):
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")

    rows = json.loads((tmp_path / LEADERBOARD_JSON).read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["run_id"] == "rule/rule#35a72911"


def test_validate_rejects_duplicate_run_ids(tmp_path):
    append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")
    rows = validate_leaderboard(tmp_path / LEADERBOARD_JSON)
    duplicate = [row.model_dump(mode="json") for row in rows[:1]] * 2
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate run_id"):
        validate_leaderboard(duplicate_path)


def test_hook_append_or_update_writes_leaderboard_outputs(tmp_path):
    row = hook_append_or_update(FIXTURES / "rule_run", tmp_path, bundle="rule")

    rows = json.loads((tmp_path / LEADERBOARD_JSON).read_text(encoding="utf-8"))
    assert row.run_id == "rule/rule#35a72911"
    assert rows[0]["run_id"] == row.run_id
