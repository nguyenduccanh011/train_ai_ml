"""Create MACD + MA20 rule components in database."""

import sys
from pathlib import Path

# Fix UTF-8 encoding on Windows
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add both repo root and stock_ml to path
REPO_ROOT = Path(__file__).resolve().parents[1].parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "stock_ml"))

from sqlalchemy.orm import sessionmaker

from stock_ml.db.engine import sync_engine
from stock_ml.db.models.template import ModelComponentModel

Session = sessionmaker(bind=sync_engine)
session = Session()

try:
    # Check if already exists
    existing_entry = session.query(ModelComponentModel).filter_by(name="macd_ma20_entry").first()

    if existing_entry:
        print(f"✓ Entry component already exists: ID {existing_entry.id}")
    else:
        entry_comp = ModelComponentModel(
            name="macd_ma20_entry",
            role="entry",
            algorithm="rule",
            component_type="rule",
            params={
                "conditions": [
                    {"feature": "macd_hist", "op": ">", "value": 0},
                    {"feature": "sma_20_ratio", "op": "<", "value": 1.0},
                    {"feature": "close_to_open", "op": ">", "value": 1.0},
                ],
                "logic": "AND",
                "score_feature": "macd_hist",
            },
            description="Entry: MACD HIS > 0, MA20 < Close, Close > Open",
            is_active=True,
        )
        session.add(entry_comp)
        session.flush()
        print(f"✓ Created entry component: ID {entry_comp.id}")

    # Create exit component
    existing_exit = session.query(ModelComponentModel).filter_by(name="macd_ma20_exit").first()

    if existing_exit:
        print(f"✓ Exit component already exists: ID {existing_exit.id}")
    else:
        exit_comp = ModelComponentModel(
            name="macd_ma20_exit",
            role="exit",
            algorithm="rule",
            component_type="rule",
            params={
                "conditions": [
                    {"feature": "macd_hist", "op": "<", "value": 0},
                    {"feature": "sma_20_ratio", "op": ">", "value": 1.0},
                    {"feature": "close_to_open", "op": "<", "value": 1.0},
                ],
                "logic": "AND",
                "score_feature": "macd_hist",
            },
            description="Exit: MACD HIS < 0, Close < MA20, Close < Open",
            is_active=True,
        )
        session.add(exit_comp)
        session.flush()
        print(f"✓ Created exit component: ID {exit_comp.id}")

    session.commit()
    print("\n✓ All components saved to database")

    # List all rule components
    print("\n=== Rule Components in Database ===")
    all_rules = session.query(ModelComponentModel).filter_by(algorithm="rule").all()

    for comp in all_rules:
        print(f"  {comp.role:6} | {comp.name:30} | ID: {comp.id}")

except Exception as e:
    session.rollback()
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
finally:
    session.close()
