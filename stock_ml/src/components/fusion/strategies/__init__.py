from src.components.fusion.base import FusionStrategy
from src.components.fusion.registry import get_strategy, register_strategy
from src.components.fusion.strategies.core import (
    AdaptiveTrailing,
    AtrStopLoss,
    EarlyLossCutExit,
    ExitModelExit,
    FastExitLossLegacy,
    HapPreemptExit,
    HardStopExit,
    MaCrossHybridExit,
    MinHoldProtection,
    PeakProtectDist,
    PeakProtectEma8Streak,
    ProfitLock,
    SignalHardCapExit,
    V22FastExit,
    V22HardCap,
    ZombieExit,
)
from src.components.fusion.strategies.entry import V19EntryCascade
from src.components.fusion.strategies.hold import LongHorizonCarry, V19SignalHoldGuard
from src.components.fusion.strategies.rule_signal import RuleSignalEntry, RuleSignalExit

register_strategy("rule_signal_entry", "entry", RuleSignalEntry)
register_strategy("rule_signal_exit", "exit_override", RuleSignalExit)
register_strategy("v19_entry_cascade", "entry", V19EntryCascade)
register_strategy("v19_signal_hold_guard", "hold", V19SignalHoldGuard)
register_strategy("hard_stop_exit", "exit_override", HardStopExit, always_on=True)
register_strategy("exit_model", "exit_override", ExitModelExit)
# Backward-compatible alias kept for old YAML/docs.
register_strategy("exit_model_exit", "exit_override", ExitModelExit)
register_strategy("signal_hard_cap", "exit_override", SignalHardCapExit)
register_strategy("fast_exit_loss", "exit_override", FastExitLossLegacy)
register_strategy("early_loss_cut", "exit_override", EarlyLossCutExit)
register_strategy("v22_hard_cap", "exit_override", V22HardCap)
register_strategy("hap_preempt", "exit_override", HapPreemptExit)
register_strategy("v22_fast_exit", "exit_override", V22FastExit)
register_strategy("atr_stop_loss", "exit_override", AtrStopLoss, always_on=True)
register_strategy("peak_protect_dist", "exit_override", PeakProtectDist, always_on=True)
register_strategy(
    "peak_protect_ema8_streak", "exit_override", PeakProtectEma8Streak, always_on=True
)
register_strategy("ma_cross_hybrid_exit", "exit_override", MaCrossHybridExit, always_on=True)
register_strategy("adaptive_trailing", "exit_override", AdaptiveTrailing, always_on=True)
register_strategy("profit_lock", "exit_override", ProfitLock, always_on=True)
register_strategy("zombie_exit", "exit_override", ZombieExit, always_on=True)
register_strategy("min_hold_protection", "hold", MinHoldProtection, always_on=True)
register_strategy("long_horizon_carry", "hold", LongHorizonCarry)

__all__ = [
    "AdaptiveTrailing",
    "AtrStopLoss",
    "EarlyLossCutExit",
    "FastExitLossLegacy",
    "HapPreemptExit",
    "HardStopExit",
    "LongHorizonCarry",
    "ExitModelExit",
    "MaCrossHybridExit",
    "MinHoldProtection",
    "PeakProtectDist",
    "PeakProtectEma8Streak",
    "ProfitLock",
    "RuleSignalEntry",
    "RuleSignalExit",
    "SignalHardCapExit",
    "V19EntryCascade",
    "V19SignalHoldGuard",
    "V22FastExit",
    "V22HardCap",
    "ZombieExit",
    "build_exit_strategies",
]


def build_exit_strategies(rule_names: list[str]) -> list[FusionStrategy]:
    strategies: list[FusionStrategy] = []
    for name in rule_names:
        strategies.append(get_strategy(name))
    return strategies
