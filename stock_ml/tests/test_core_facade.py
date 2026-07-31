"""stock_ml.core must be import-clean (no DB/web deps) and expose the serving API.

A separate production service depends on stock_ml.core alone; if importing it
drags in sqlalchemy/fastapi/asyncpg, the "lightweight serving" promise is broken.
The import-clean check runs in a fresh subprocess (the pytest process has already
imported heavy deps via other tests, so sys.modules here is not representative).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_core_imports_clean_no_db_or_web():
    code = (
        "import sys\n"
        "import stock_ml.core\n"
        "heavy = sorted({m.split('.')[0] for m in sys.modules "
        "if m.split('.')[0] in ('sqlalchemy','fastapi','asyncpg','psycopg2',"
        "'uvicorn','starlette','aiosqlite','alembic')})\n"
        "print('HEAVY=' + ','.join(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"import stock_ml.core failed:\n{proc.stderr}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("HEAVY=")][-1]
    heavy = line[len("HEAVY=") :].strip()
    assert heavy == "", f"stock_ml.core pulled heavy deps: {heavy}"


def test_core_exposes_serving_api():
    import stock_ml.core as core

    for name in (
        "load_bundle",
        "generate_signals_from_bundle",
        "build_feature_frame",
        "predict_slot_signals",
        "recombine_signals",
        "ExperimentConfig",
        "LoadedBundle",
    ):
        assert hasattr(core, name), f"stock_ml.core missing {name}"
    assert isinstance(core.__version__, str)
