"""Seed database with feature sets and targets for MACD template."""

import io
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from sqlalchemy.orm import sessionmaker
from stock_ml.src.features.leading_v2 import FEATURE_COLS

from stock_ml.db.engine import sync_engine
from stock_ml.db.models.template import (
    FeatureSetCatalogModel,
    TargetCatalogModel,
)

Session = sessionmaker(bind=sync_engine)
session = Session()

try:
    # Seed leading_v2 feature set
    existing_fs = session.query(FeatureSetCatalogModel).filter_by(name="leading_v2").first()

    if existing_fs:
        print(f"[SKIP] Feature set 'leading_v2' already exists (ID: {existing_fs.id})")
    else:
        fs = FeatureSetCatalogModel(
            name="leading_v2",
            column_count=len(FEATURE_COLS),
            columns=FEATURE_COLS,
            description="Extended feature set with 35 per-symbol features (moving averages, momentum, trend, volatility, volume)",
            is_active=True,
        )
        session.add(fs)
        session.flush()
        print(f"[OK] Created feature set 'leading_v2' (ID: {fs.id}, {len(FEATURE_COLS)} features)")

    # Seed trend_regime target
    existing_target = session.query(TargetCatalogModel).filter_by(name="trend_regime").first()

    if existing_target:
        print(f"[SKIP] Target 'trend_regime' already exists (ID: {existing_target.id})")
    else:
        target = TargetCatalogModel(
            name="trend_regime",
            type="trend_regime",
            params={
                "trend_method": "dual_ma",
                "short_window": 10,
                "long_window": 40,
                "classes": 3,
            },
            output_dtype="int32",
            description="3-class trend regime: downtrend (0), uptrend (1), sideways (2)",
            is_active=True,
        )
        session.add(target)
        session.flush()
        print(f"[OK] Created target 'trend_regime' (ID: {target.id})")

    session.commit()
    print("\n[SUCCESS] Database seeded with feature sets and targets")

except Exception as e:
    session.rollback()
    print(f"[ERROR] {e}")
    import traceback

    traceback.print_exc()
finally:
    session.close()
