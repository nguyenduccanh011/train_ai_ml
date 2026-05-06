from __future__ import annotations

from abc import ABC, abstractmethod


class PnlCalculator(ABC):
    @abstractmethod
    def entry_cost(self, capital: float, commission: float, slippage: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def exit_cost(self, capital: float, commission: float, tax: float, slippage: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def bar_return(self, ret: float, position_size: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def trade_return(self, entry_price: float, exit_price: float, slippage: float) -> float:
        raise NotImplementedError


class EquitySpotCalculator(PnlCalculator):
    def entry_cost(self, capital: float, commission: float, slippage: float) -> float:
        return capital * (commission + slippage)

    def exit_cost(self, capital: float, commission: float, tax: float, slippage: float) -> float:
        return capital * (commission + tax + slippage)

    def bar_return(self, ret: float, position_size: float) -> float:
        return ret * position_size

    def trade_return(self, entry_price: float, exit_price: float, slippage: float) -> float:
        if entry_price <= 0:
            return 0.0
        effective_entry = entry_price * (1.0 + slippage)
        effective_exit = exit_price * (1.0 - slippage)
        return effective_exit / effective_entry - 1.0


class LinearUsdtPerpCalculator(EquitySpotCalculator):
    def exit_cost(self, capital: float, commission: float, tax: float, slippage: float) -> float:
        return capital * (commission + slippage)


class InversePerpCalculator(LinearUsdtPerpCalculator):
    def bar_return(self, ret: float, position_size: float) -> float:
        return -ret * position_size

    def trade_return(self, entry_price: float, exit_price: float, slippage: float) -> float:
        if entry_price <= 0 or exit_price <= 0:
            return 0.0
        effective_entry = entry_price * (1.0 + slippage)
        effective_exit = exit_price * (1.0 - slippage)
        return effective_entry / effective_exit - 1.0


class FuturesContractCalculator(LinearUsdtPerpCalculator):
    def compute_roll_cost(self, capital: float, roll_cost: float) -> float:
        return capital * roll_cost


_REGISTRY: dict[str, PnlCalculator] = {
    "equity_spot": EquitySpotCalculator(),
    "linear_usdt_perp": LinearUsdtPerpCalculator(),
    "inverse_perp": InversePerpCalculator(),
    "futures_contract": FuturesContractCalculator(),
}
SUPPORTED_PNL_MODES = frozenset(_REGISTRY)


def get_pnl_calculator(pnl_mode: str) -> PnlCalculator:
    try:
        return _REGISTRY[pnl_mode]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unsupported pnl_mode {pnl_mode!r}. Available: {available}") from exc
