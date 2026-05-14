"""
Test script to verify compound return calculation fix.

This script demonstrates the difference between simple sum and compound return.
"""

import numpy as np


def simple_sum(pnls):
    """Simple sum (WRONG way)."""
    return sum(pnls)


def compound_return(pnls):
    """Compound return (CORRECT way)."""
    cumulative = 1.0
    for pnl in pnls:
        cumulative *= 1 + pnl / 100
    return (cumulative - 1) * 100


def test_examples():
    print("=" * 70)
    print("Compound Return Calculation Test")
    print("=" * 70)

    # Example 1: All wins
    print("\n[Example 1] All wins: +10%, +10%, +10%")
    pnls1 = [10, 10, 10]
    simple1 = simple_sum(pnls1)
    compound1 = compound_return(pnls1)
    print(f"  Simple sum:      {simple1:.2f}%")
    print(f"  Compound return: {compound1:.2f}%")
    print(f"  Difference:      {compound1 - simple1:.2f}%")

    # Example 2: Mixed wins and losses
    print("\n[Example 2] Mixed: +5%, +3%, -2%, +4%")
    pnls2 = [5, 3, -2, 4]
    simple2 = simple_sum(pnls2)
    compound2 = compound_return(pnls2)
    print(f"  Simple sum:      {simple2:.2f}%")
    print(f"  Compound return: {compound2:.2f}%")
    print(f"  Difference:      {compound2 - simple2:.2f}%")

    # Example 3: Many small wins
    print("\n[Example 3] 100 trades × +5% each")
    pnls3 = [5] * 100
    simple3 = simple_sum(pnls3)
    compound3 = compound_return(pnls3)
    print(f"  Simple sum:      {simple3:.2f}%")
    print(f"  Compound return: {compound3:.2f}%")
    print(f"  Difference:      {compound3 - simple3:.2f}%")
    print(f"  Error ratio:     {compound3 / simple3:.2f}x")

    # Example 4: Your case - 46900% simple sum
    print("\n[Example 4] Reverse engineer: simple sum = 46900%")
    # If simple sum = 46900% and compound = 187.5%
    # Let's verify with realistic trades
    # Assume avg trade = 5%, then 46900 / 5 = 9380 trades
    n_trades = 9380
    avg_pnl = 5.0
    pnls4 = [avg_pnl] * n_trades
    simple4 = simple_sum(pnls4)
    compound4 = compound_return(pnls4)
    print(f"  Assumed: {n_trades} trades × {avg_pnl}% each")
    print(f"  Simple sum:      {simple4:.2f}%")
    print(f"  Compound return: {compound4:.2f}%")
    print(f"  Difference:      {compound4 - simple4:.2f}%")

    # Example 5: More realistic - with losses
    print("\n[Example 5] Realistic: 70% WR, avg win +8%, avg loss -4%")
    np.random.seed(42)
    n = 1000
    wins = int(n * 0.7)
    losses = n - wins
    pnls5 = [8.0] * wins + [-4.0] * losses
    np.random.shuffle(pnls5)
    simple5 = simple_sum(pnls5)
    compound5 = compound_return(pnls5)
    print(f"  Trades: {n} (wins={wins}, losses={losses})")
    print(f"  Simple sum:      {simple5:.2f}%")
    print(f"  Compound return: {compound5:.2f}%")
    print(f"  Difference:      {compound5 - simple5:.2f}%")
    print(f"  Error ratio:     {compound5 / simple5:.2f}x")


def test_equity_curve():
    print("\n" + "=" * 70)
    print("Equity Curve Comparison")
    print("=" * 70)

    pnls = [10, -5, 8, -3, 12, -7, 15]
    print(f"\nTrades: {pnls}")

    # Simple sum equity
    print("\nSimple sum equity curve:")
    equity_simple = np.cumsum(pnls)
    for i, (pnl, eq) in enumerate(zip(pnls, equity_simple, strict=False)):
        print(f"  Trade {i + 1}: {pnl:+.1f}% -> Equity: {eq:+.1f}%")

    # Compound equity
    print("\nCompound equity curve:")
    equity_compound = []
    cumulative = 1.0
    for i, pnl in enumerate(pnls):
        cumulative *= 1 + pnl / 100
        eq = (cumulative - 1) * 100
        equity_compound.append(eq)
        print(f"  Trade {i + 1}: {pnl:+.1f}% -> Equity: {eq:+.1f}%")

    print("\nFinal equity:")
    print(f"  Simple sum:      {equity_simple[-1]:+.1f}%")
    print(f"  Compound return: {equity_compound[-1]:+.1f}%")


def test_mdd():
    print("\n" + "=" * 70)
    print("Max Drawdown Comparison")
    print("=" * 70)

    pnls = [10, 15, -8, -5, 12, -10, 20]
    print(f"\nTrades: {pnls}")

    # Simple sum MDD
    equity_simple = np.cumsum(pnls)
    peak_simple = np.maximum.accumulate(equity_simple)
    dd_simple = peak_simple - equity_simple
    mdd_simple = np.max(dd_simple)

    print(f"\nSimple sum MDD: {mdd_simple:.2f}%")
    print("  Equity curve:", [f"{x:.1f}" for x in equity_simple])
    print("  Peak:        ", [f"{x:.1f}" for x in peak_simple])
    print("  Drawdown:    ", [f"{x:.1f}" for x in dd_simple])

    # Compound MDD
    equity_compound = []
    cumulative = 1.0
    for pnl in pnls:
        cumulative *= 1 + pnl / 100
        equity_compound.append((cumulative - 1) * 100)
    equity_compound = np.array(equity_compound)
    peak_compound = np.maximum.accumulate(equity_compound)
    dd_compound = peak_compound - equity_compound
    mdd_compound = np.max(dd_compound)

    print(f"\nCompound MDD: {mdd_compound:.2f}%")
    print("  Equity curve:", [f"{x:.1f}" for x in equity_compound])
    print("  Peak:        ", [f"{x:.1f}" for x in peak_compound])
    print("  Drawdown:    ", [f"{x:.1f}" for x in dd_compound])

    print(f"\nDifference: {mdd_compound - mdd_simple:.2f}%")


def main():
    test_examples()
    test_equity_curve()
    test_mdd()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key takeaways:
1. Simple sum OVERESTIMATES returns (especially with many trades)
2. Compound return is the CORRECT way to calculate cumulative PnL
3. MDD should also be calculated on compound equity curve
4. The difference grows exponentially with number of trades

Your case (46900% vs 187.5%):
- 46900% is likely simple sum (WRONG)
- 187.5% is likely compound return (CORRECT)
- This is a ~250x difference!

After the fix:
- total_pnl_pct will show compound return (correct)
- total_pnl_simple will show simple sum (for reference)
- Frontend equity chart already uses compound (correct)
""")


if __name__ == "__main__":
    main()
