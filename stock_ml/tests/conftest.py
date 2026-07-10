"""Pytest bootstrap for the stock_ml suite.

Two things have to happen before any test module imports ``stock_ml.*``:

1. **sys.path** — ``stock_ml`` is a namespace package (no ``stock_ml/__init__.py``),
   so ``import stock_ml.db...`` only resolves when the repo root is on
   ``sys.path``. Pytest only prepends the rootdir of the test package
   (``stock_ml/``), so we add the repo root here. ``src.*`` imports keep working
   because pytest already put ``stock_ml/`` on the path.

2. **DATABASE_URL** — ``stock_ml.db`` eagerly builds SQLAlchemy engines at import
   time from ``settings.database_url``, whose default targets Postgres
   (``asyncpg``). Tests run on SQLite and asyncpg is not a test dependency, so we
   pin SQLite unless the environment already chose a URL (``setdefault`` keeps an
   explicit Postgres override, e.g. a CI integration job).
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
