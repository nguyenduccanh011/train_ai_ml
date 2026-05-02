"""Regression: 11 champion trades CSV phải match golden hash exact.

Test này hash file ở `stock_ml/results/trades_<v>.csv` và so với
`stock_ml/tests/regression/golden/checksums.txt`. Không tự rerun pipeline (~6 phút).

Cách dùng:
- Local: chạy `python run_pipeline.py --version v22 --compare ... --device cpu --force --no-export`
  để regen results, rồi `pytest stock_ml/tests/regression/test_champions.py`.
- Pre-commit: hook `regression-champions` chạy test này khi touch src/, run_pipeline.py, config/.

Yêu cầu reproducibility: CPU mode (LightGBM GPU OpenCL non-deterministic, xem
docs/refactor/diary/2026-04-27.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

CHAMPIONS = [
    "v22",
    "v22_with_exit_model",
    "v32",
    "v34",
    "v35b",
    "v37a",
    "v37a_exit",
    "v37d",
    "v39d",
    "v42_a",
    "v19_3",
    "rule",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "stock_ml" / "results"
GOLDEN_DIR = REPO_ROOT / "stock_ml" / "tests" / "regression" / "golden"
CHECKSUMS_FILE = GOLDEN_DIR / "checksums.txt"


def _parse_checksums() -> dict[str, str]:
    """Parse `<sha256> *<filename>` lines from checksums.txt → {filename: hash}."""
    if not CHECKSUMS_FILE.exists():
        pytest.fail(f"Golden checksums missing: {CHECKSUMS_FILE}")
    out: dict[str, str] = {}
    for line in CHECKSUMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        h, name = parts
        out[name.lstrip("*")] = h
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def golden_hashes() -> dict[str, str]:
    return _parse_checksums()


@pytest.mark.regression
@pytest.mark.parametrize("version", CHAMPIONS)
def test_champion_trades_hash_matches_golden(version: str, golden_hashes: dict[str, str]) -> None:
    fname = f"trades_{version}.csv"
    actual_path = RESULTS_DIR / fname
    if not actual_path.exists():
        pytest.skip(
            f"{actual_path} không tồn tại. Regen bằng: python run_pipeline.py --version v22 "
            f"--compare v32,v34,v35b,v37a,v37a_exit,v37d,v39d,v42_a,v19_3,rule "
            f"--device cpu --force --no-export"
        )

    expected = golden_hashes.get(fname)
    assert expected, f"Golden không có entry cho {fname}"

    actual = _sha256(actual_path)
    assert actual == expected, (
        f"{fname} hash mismatch.\n"
        f"  Expected (golden): {expected}\n"
        f"  Actual (results/): {actual}\n"
        f"Có thể do: (1) chạy GPU thay vì CPU, (2) code thay đổi ảnh hưởng output, "
        f"(3) data hoặc cache thay đổi. Xem docs/refactor/diary/2026-04-27.md."
    )


@pytest.mark.regression
def test_golden_dir_has_all_champions(golden_hashes: dict[str, str]) -> None:
    """Sanity: checksums.txt phải có đủ entry cho mọi champion."""
    expected = {f"trades_{v}.csv" for v in CHAMPIONS}
    missing = expected - set(golden_hashes.keys())
    assert not missing, f"Golden checksums thiếu: {sorted(missing)}"
