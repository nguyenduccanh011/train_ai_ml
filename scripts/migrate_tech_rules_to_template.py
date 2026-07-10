"""
Migrate technical_rules baseline model to use template + component slots.

Creates:
1. FeatureSetCatalogModel: technical_v1
2. TargetCatalogModel: forward_return_regression (if not exists)
3. ModelComponentModel: rule_based_entry, fixed_loss_exit
4. StrategyTemplateModel: technical_rules_v1
5. ComponentSlotModel: entry, exit slots
6. Updates LeaderboardRunModel.template_id
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from stock_ml.db.engine import sync_engine
from stock_ml.db.models.run import LeaderboardRunModel
from stock_ml.db.models.template import (
    ComponentSlotModel,
    FeatureSetCatalogModel,
    ModelComponentModel,
    StrategyTemplateModel,
    TargetCatalogModel,
)


def ensure_feature_set(session: Session) -> int:
    """Ensure technical_v1 feature set exists. Returns its ID."""
    stmt = select(FeatureSetCatalogModel).where(FeatureSetCatalogModel.name == "technical_v1")
    fs = session.execute(stmt).scalar_one_or_none()

    if fs:
        print(f"[OK] Feature set 'technical_v1' already exists (ID: {fs.id})")
        return fs.id

    # Create new feature set
    fs = FeatureSetCatalogModel(
        name="technical_v1",
        column_count=0,
        columns=[],
        description="Technical indicators: MACD, MA20",
        is_active=True,
    )
    session.add(fs)
    session.flush()
    print(f"[OK] Created feature set 'technical_v1' (ID: {fs.id})")
    return fs.id


def ensure_target(session: Session) -> int:
    """Ensure forward_return_regression target exists. Returns its ID."""
    stmt = select(TargetCatalogModel).where(
        TargetCatalogModel.name == "forward_return_regression_5d"
    )
    tgt = session.execute(stmt).scalar_one_or_none()

    if tgt:
        print(f"[OK] Target 'forward_return_regression_5d' already exists (ID: {tgt.id})")
        return tgt.id

    # Create new target
    tgt = TargetCatalogModel(
        name="forward_return_regression_5d",
        type="forward_return_regression",
        params={"forward_window": 5},
        output_dtype="float32",
        description="5-day forward return (regression target)",
        is_active=True,
    )
    session.add(tgt)
    session.flush()
    print(f"[OK] Created target 'forward_return_regression_5d' (ID: {tgt.id})")
    return tgt.id


def ensure_components(session: Session) -> tuple[int, int]:
    """Ensure entry and exit components exist. Returns (entry_id, exit_id)."""
    # Entry component
    entry_stmt = select(ModelComponentModel).where(ModelComponentModel.name == "rule_based_entry")
    entry_comp = session.execute(entry_stmt).scalar_one_or_none()

    if not entry_comp:
        entry_comp = ModelComponentModel(
            name="rule_based_entry",
            role="entry",
            algorithm="macd_ma20_crossover",
            component_type="rule",
            params={
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ma_period": 20,
            },
            description="Rule-based entry: MACD crossover + MA20 filter",
            is_default=False,
            is_active=True,
        )
        session.add(entry_comp)
        session.flush()
        print(f"[OK] Created entry component 'rule_based_entry' (ID: {entry_comp.id})")
    else:
        print(f"[OK] Entry component 'rule_based_entry' already exists (ID: {entry_comp.id})")

    # Exit component
    exit_stmt = select(ModelComponentModel).where(ModelComponentModel.name == "fixed_loss_exit")
    exit_comp = session.execute(exit_stmt).scalar_one_or_none()

    if not exit_comp:
        exit_comp = ModelComponentModel(
            name="fixed_loss_exit",
            role="exit",
            algorithm="fixed_loss",
            component_type="rule",
            params={
                "loss_pct": -0.1155,  # max_loss from model
            },
            description="Rule-based exit: Fixed loss threshold",
            is_default=False,
            is_active=True,
        )
        session.add(exit_comp)
        session.flush()
        print(f"[OK] Created exit component 'fixed_loss_exit' (ID: {exit_comp.id})")
    else:
        print(f"[OK] Exit component 'fixed_loss_exit' already exists (ID: {exit_comp.id})")

    return entry_comp.id, exit_comp.id


def create_template(
    session: Session,
    feature_set_id: int,
    target_id: int,
    entry_comp_id: int,
    exit_comp_id: int,
) -> int:
    """Create strategy template. Returns template ID."""
    stmt = select(StrategyTemplateModel).where(StrategyTemplateModel.name == "technical_rules_v1")
    template = session.execute(stmt).scalar_one_or_none()

    if template:
        print(f"[OK] Template 'technical_rules_v1' already exists (ID: {template.id})")
        return template.id

    template = StrategyTemplateModel(
        name="technical_rules_v1",
        description="Technical Rules (MACD+MA20) - Phase 0.4 refactor baseline",
        hypothesis="Rule-based entry/exit with technical indicators outperforms random entry",
        market="vn_stock",
        strategy="macd_ma20",
        direction="long",
        universe_slug="vn_large_cap",
        feature_set_id=feature_set_id,
        target_id=target_id,
        signal_mode="entry_first",
        signal_threshold=0.0,
        model_mode="rule_only",
        split_config={
            "test_start": "2020-01-01",
            "test_end": "2026-05-30",
        },
        engine_config={
            "commission": 0.001,
            "tax": 0.0,
            "slippage": 0.0,
        },
        validation_config={},
        seed=42,
        is_active=True,
        schema_version=2,
        config_hash="a1b2c3d4",
        version=1,
    )
    session.add(template)
    session.flush()
    print(f"[OK] Created template 'technical_rules_v1' (ID: {template.id})")

    # Create component slots
    entry_slot = ComponentSlotModel(
        template_id=template.id,
        slot_type="entry",
        ml_component_id=None,
        rule_component_id=entry_comp_id,
        feature_set_name=None,  # Use global
        target_config=None,  # Use global
    )
    session.add(entry_slot)

    exit_slot = ComponentSlotModel(
        template_id=template.id,
        slot_type="exit",
        ml_component_id=None,
        rule_component_id=exit_comp_id,
        feature_set_name=None,
        target_config=None,
    )
    session.add(exit_slot)
    session.flush()
    print("[OK] Created component slots (entry, exit)")

    return template.id


def update_run_template(session: Session, template_id: int) -> None:
    """Update LeaderboardRunModel to use new template."""
    stmt = select(LeaderboardRunModel).where(
        LeaderboardRunModel.run_id == "baseline/tech_rules#a1b2c3d4"
    )
    run = session.execute(stmt).scalar_one_or_none()

    if not run:
        print("[WARN] Run not found: baseline/tech_rules#a1b2c3d4")
        return

    old_template_id = run.template_id
    run.template_id = template_id
    session.add(run)
    session.flush()
    print(f"[OK] Updated run.template_id: {old_template_id} → {template_id}")


def main():
    print("\n=== Migrating technical_rules baseline to template ===\n")

    with Session(sync_engine) as session:
        try:
            # 1. Ensure feature set
            feature_set_id = ensure_feature_set(session)

            # 2. Ensure target
            target_id = ensure_target(session)

            # 3. Ensure components
            entry_comp_id, exit_comp_id = ensure_components(session)

            # 4. Create template
            template_id = create_template(
                session,
                feature_set_id=feature_set_id,
                target_id=target_id,
                entry_comp_id=entry_comp_id,
                exit_comp_id=exit_comp_id,
            )

            # 5. Update run
            update_run_template(session, template_id)

            # Commit
            session.commit()
            print("\n[OK] Migration completed successfully!")
            print(f"  Template ID: {template_id}")
            print(f"  Feature Set ID: {feature_set_id}")
            print(f"  Target ID: {target_id}")

        except Exception as e:
            session.rollback()
            print(f"\n[ERROR] Migration failed: {e}")
            raise


if __name__ == "__main__":
    main()
