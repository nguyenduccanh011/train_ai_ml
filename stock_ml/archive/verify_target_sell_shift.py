"""Verification script for target_sell shift fix.

Tests that target_sell[i] predicts "should exit at i+1" (not at i).
"""
import sys
from pathlib import Path

import pandas as pd

# Add stock_ml to path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.target import TargetGenerator


def test_legacy_api():
    """Test legacy TargetGenerator API."""
    print("=" * 60)
    print("Testing Legacy API (src.data.target.TargetGenerator)")
    print("=" * 60)

    # Test case: price drops 5% at day 3 (index 2)
    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.0, 95.0, 95.0, 95.0],
        'symbol': ['TEST'] * 6
    })

    gen = TargetGenerator(
        target_type='early_wave_dual',
        forward_window=2,
        loss_threshold=0.03  # 3% threshold
    )
    result = gen.generate(df)

    print("\nInput close prices:", result['close'].values)
    print("target_sell:       ", result['target_sell'].values)
    print("\nAnalysis:")
    print("  - Before shift:")
    print("    - sell[0] = 0 (forward [100, 100] -> no drop)")
    print("    - sell[1] = 1 (forward [100, 95] -> drop 5%)")
    print("    - sell[2] = 1 (forward [95, 95] -> drop 5%)")
    print("    - sell[3] = 0 (forward [95, 95] -> no drop)")
    print("  - After shift -1:")
    print("    - target_sell[0] = sell[1] = 1")
    print("    - target_sell[1] = sell[2] = 1")
    print("    - target_sell[2] = sell[3] = 0")

    print("\nExpected (after shift -1):")
    print("  - target_sell[0] = 1 (predict day 1: should exit, day 2-3 drops)")
    print("  - target_sell[1] = 1 (predict day 2: should exit, day 3-4 drops)")
    print("  - target_sell[2] = 0 (predict day 3: no exit needed)")

    print("\nActual:")
    for i in range(len(result)):
        val = result['target_sell'].iloc[i]
        print(f"  - target_sell[{i}] = {val}")

    # Verify shift
    assert result['target_sell'].iloc[0] == 1.0, \
        f"Expected target_sell[0]=1.0, got {result['target_sell'].iloc[0]}"
    assert result['target_sell'].iloc[1] == 1.0, \
        f"Expected target_sell[1]=1.0, got {result['target_sell'].iloc[1]}"
    assert result['target_sell'].iloc[2] == 0.0, \
        f"Expected target_sell[2]=0.0, got {result['target_sell'].iloc[2]}"

    print("\n[OK] Legacy API shift verified!")
    return True


def test_new_component_api():
    """Test new component API."""
    print("\n" + "=" * 60)
    print("Testing New Component API (src.components.targets)")
    print("=" * 60)

    from src.components.targets.early_wave_dual import EarlyWaveDualTarget

    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.0, 95.0, 95.0, 95.0],
        'high': [101.0, 101.0, 101.0, 96.0, 96.0, 96.0],
        'low': [99.0, 99.0, 99.0, 94.0, 94.0, 94.0],
        'symbol': ['TEST'] * 6
    })

    target = EarlyWaveDualTarget(
        forward_window=2,
        loss_threshold=0.03
    )
    exit_labels = target.generate_exit_labels(df, forward_window=2, loss_threshold=0.03)

    print("\nInput close prices:", df['close'].values)
    print("exit_labels:       ", exit_labels.values)

    print("\nExpected (after shift -1):")
    print("  - exit_labels[0] = 1 (predict day 1: should exit)")
    print("  - exit_labels[1] = 1 (predict day 2: should exit)")
    print("  - exit_labels[2] = 0 (predict day 3: no exit needed)")

    print("\nActual:")
    for i in range(len(exit_labels)):
        val = exit_labels.iloc[i]
        print(f"  - exit_labels[{i}] = {val}")

    # Verify shift
    assert exit_labels.iloc[0] == 1.0, \
        f"Expected exit_labels[0]=1.0, got {exit_labels.iloc[0]}"
    assert exit_labels.iloc[1] == 1.0, \
        f"Expected exit_labels[1]=1.0, got {exit_labels.iloc[1]}"
    assert exit_labels.iloc[2] == 0.0, \
        f"Expected exit_labels[2]=0.0, got {exit_labels.iloc[2]}"

    print("\n[OK] New Component API shift verified!")
    return True


def test_equivalence():
    """Test that both APIs produce the same result."""
    print("\n" + "=" * 60)
    print("Testing API Equivalence")
    print("=" * 60)

    df = pd.DataFrame({
        'close': [100.0, 100.0, 100.0, 95.0, 95.0, 95.0],
        'high': [101.0, 101.0, 101.0, 96.0, 96.0, 96.0],
        'low': [99.0, 99.0, 99.0, 94.0, 94.0, 94.0],
        'symbol': ['TEST'] * 6
    })

    # Legacy API
    gen = TargetGenerator(
        target_type='early_wave_dual',
        forward_window=2,
        loss_threshold=0.03
    )
    legacy_result = gen.generate(df)

    # New API
    from src.components.targets.early_wave_dual import EarlyWaveDualTarget
    target = EarlyWaveDualTarget(forward_window=2, loss_threshold=0.03)
    new_result = target.generate_exit_labels(df, forward_window=2, loss_threshold=0.03)

    print("\nLegacy target_sell:", legacy_result['target_sell'].values)
    print("New exit_labels:   ", new_result.values)

    # Compare (allowing for NaN equality)
    for i in range(len(df)):
        legacy_val = legacy_result['target_sell'].iloc[i]
        new_val = new_result.iloc[i]
        if pd.isna(legacy_val) and pd.isna(new_val):
            continue
        assert legacy_val == new_val, \
            f"Mismatch at index {i}: legacy={legacy_val}, new={new_val}"

    print("\n[OK] Both APIs produce identical results!")
    return True


if __name__ == "__main__":
    try:
        test_legacy_api()
        test_new_component_api()
        test_equivalence()

        print("\n" + "=" * 60)
        print("[OK] ALL TESTS PASSED - target_sell shift verified!")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
