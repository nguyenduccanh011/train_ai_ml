from __future__ import annotations

import pytest
from src.backtest.pnl import (
    EquitySpotCalculator,
    FuturesContractCalculator,
    InversePerpCalculator,
    LinearUsdtPerpCalculator,
    get_pnl_calculator,
)


def test_equity_spot_calculator_costs_and_returns() -> None:
    calc = EquitySpotCalculator()

    assert calc.entry_cost(1_000_000, commission=0.0015, slippage=0.001) == pytest.approx(2_500)
    assert calc.exit_cost(
        1_100_000,
        commission=0.0015,
        tax=0.001,
        slippage=0.001,
    ) == pytest.approx(3_850)
    assert calc.bar_return(ret=0.02, position_size=0.5) == pytest.approx(0.01)
    assert calc.trade_return(100, 110, slippage=0.001) == pytest.approx(
        (110 * 0.999) / (100 * 1.001) - 1
    )


def test_linear_usdt_perp_calculator_costs_and_returns() -> None:
    calc = LinearUsdtPerpCalculator()

    assert calc.entry_cost(1_000, commission=0.0004, slippage=0.0002) == pytest.approx(0.6)
    assert calc.exit_cost(1_100, commission=0.0004, tax=0.001, slippage=0.0002) == pytest.approx(
        0.66
    )
    assert calc.bar_return(ret=-0.02, position_size=1.5) == pytest.approx(-0.03)
    assert calc.trade_return(100, 110, slippage=0.0002) == pytest.approx(
        (110 * 0.9998) / (100 * 1.0002) - 1
    )


def test_inverse_perp_calculator_costs_and_returns() -> None:
    calc = InversePerpCalculator()

    assert calc.entry_cost(1_000, commission=0.0004, slippage=0.0002) == pytest.approx(0.6)
    assert calc.exit_cost(1_100, commission=0.0004, tax=0.001, slippage=0.0002) == pytest.approx(
        0.66
    )
    assert calc.bar_return(ret=0.02, position_size=1.5) == pytest.approx(-0.03)
    assert calc.trade_return(100, 90, slippage=0.0002) == pytest.approx(
        (100 * 1.0002) / (90 * 0.9998) - 1
    )


def test_futures_contract_calculator_costs_and_returns() -> None:
    calc = FuturesContractCalculator()

    assert calc.entry_cost(1_000, commission=0.0004, slippage=0.0002) == pytest.approx(0.6)
    assert calc.exit_cost(1_100, commission=0.0004, tax=0.001, slippage=0.0002) == pytest.approx(
        0.66
    )
    assert calc.bar_return(ret=0.02, position_size=1.5) == pytest.approx(0.03)
    assert calc.trade_return(100, 110, slippage=0.0002) == pytest.approx(
        (110 * 0.9998) / (100 * 1.0002) - 1
    )


def test_contract_multiplier_scales_futures_bar_return() -> None:
    calc = FuturesContractCalculator()

    assert calc.bar_return(ret=0.02, position_size=0.5) * 10 == pytest.approx(0.1)


def test_futures_contract_roll_cost() -> None:
    calc = FuturesContractCalculator()

    assert calc.compute_roll_cost(1_000_000, roll_cost=0.0005) == pytest.approx(500)


def test_get_pnl_calculator_returns_supported_modes() -> None:
    assert isinstance(get_pnl_calculator("equity_spot"), EquitySpotCalculator)
    assert isinstance(get_pnl_calculator("linear_usdt_perp"), LinearUsdtPerpCalculator)
    assert isinstance(get_pnl_calculator("inverse_perp"), InversePerpCalculator)
    assert isinstance(get_pnl_calculator("futures_contract"), FuturesContractCalculator)


def test_get_pnl_calculator_rejects_unsupported_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported pnl_mode"):
        get_pnl_calculator("unsupported_mode")
