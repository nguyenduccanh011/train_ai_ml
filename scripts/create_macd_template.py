"""Create MACD + MA20 strategy template in database."""

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

from stock_ml.db.engine import sync_engine
from stock_ml.db.models.template import (
    ComponentSlotModel,
    FeatureSetCatalogModel,
    StrategyTemplateModel,
    TargetCatalogModel,
)

Session = sessionmaker(bind=sync_engine)
session = Session()

try:
    # Get feature set (leading_v2)
    feature_set = session.query(FeatureSetCatalogModel).filter_by(name="leading_v2").first()

    if not feature_set:
        print("ERROR: Feature set 'leading_v2' not found in DB")
        print("Available feature sets:")
        all_fs = session.query(FeatureSetCatalogModel).all()
        for fs in all_fs:
            print(f"  - {fs.name} (ID: {fs.id})")
        sys.exit(1)

    print(f"[OK] Feature set: {feature_set.name} (ID: {feature_set.id})")

    # Get target (trend_regime)
    target = session.query(TargetCatalogModel).filter_by(name="trend_regime").first()

    if not target:
        print("ERROR: Target 'trend_regime' not found in DB")
        print("Available targets:")
        all_targets = session.query(TargetCatalogModel).all()
        for t in all_targets:
            print(f"  - {t.name} (ID: {t.id}, type: {t.type})")
        sys.exit(1)

    print(f"[OK] Target: {target.name} (ID: {target.id})")

    # Check if template already exists
    existing = session.query(StrategyTemplateModel).filter_by(name="macd_ma20_rule_v1").first()

    if existing:
        print(f"\n[SKIP] Template already exists: ID {existing.id}")
        print(f"       Name: {existing.name}")
    else:
        # Create template
        template = StrategyTemplateModel(
            name="macd_ma20_rule_v1",
            description="Rule-based MACD + MA20 crossover strategy",
            hypothesis="MACD histogram > 0 AND MA20 below close for entry; opposite for exit",
            market="vn_stock",
            strategy="rule_only",
            direction="long",
            feature_set_id=feature_set.id,
            target_id=target.id,
            signal_mode="entry_first",
            signal_threshold=0.0,
            model_mode="rule_only",
            split_config={
                "type": "walk_forward_year",
                "train_years": 2,
                "test_years": 1,
                "gap_days": 25,
                "first_test_year": 2020,
                "last_test_year": 2024,
            },
            engine_config={
                "max_hold_bars": 20,
                "min_hold_bars": 1,
                "hard_stop_pct": -0.08,
                "costs": {
                    "commission": 0.0015,
                    "tax": 0.001,
                    "slippage": 0.0015,
                },
            },
            validation_config={
                "n_seeds": 1,
            },
            seed=42,
            is_active=True,
        )
        session.add(template)
        session.flush()

        # Create component slots (entry + exit)
        entry_slot = ComponentSlotModel(
            template_id=template.id,
            slot_type="entry",
            rule_component_id=1,  # macd_ma20_entry
            ml_component_id=None,
            feature_set_name=None,  # Use global
            target_config=None,  # Use global
        )
        session.add(entry_slot)

        exit_slot = ComponentSlotModel(
            template_id=template.id,
            slot_type="exit",
            rule_component_id=2,  # macd_ma20_exit
            ml_component_id=None,
            feature_set_name=None,  # Use global
            target_config=None,  # Use global
        )
        session.add(exit_slot)

        session.commit()
        print(f"\n[OK] Template created: ID {template.id}")
        print(f"     Name: {template.name}")
        print("     Entry Slot: rule_component_id=1")
        print("     Exit Slot: rule_component_id=2")

    # List all templates
    print("\n=== All Templates in Database ===")
    all_templates = session.query(StrategyTemplateModel).filter_by(is_active=True).all()

    for t in all_templates:
        slots = ", ".join([s.slot_type for s in t.component_slots])
        print(f"  {t.name:30} | ID: {t.id:3} | slots: {slots}")

    print("\n[SUCCESS] Template ready for submission!")
    print("Next: curl -X POST http://localhost:8000/api/templates/ID/submit")

except Exception as e:
    session.rollback()
    print(f"[ERROR] {e}")
    import traceback

    traceback.print_exc()
finally:
    session.close()
