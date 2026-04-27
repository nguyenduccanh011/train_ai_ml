"""Always-on core strategies (used by all ML champions: v19_3, v22, v32, ...)."""

from src.components.fusion.strategies.core.adaptive_trailing import AdaptiveTrailing
from src.components.fusion.strategies.core.atr_stop import AtrStopLoss
from src.components.fusion.strategies.core.fast_exit_loss import FastExitLossLegacy
from src.components.fusion.strategies.core.hard_stop import HardStopExit
from src.components.fusion.strategies.core.ma_cross_hybrid_exit import MaCrossHybridExit
from src.components.fusion.strategies.core.min_hold_protection import MinHoldProtection
from src.components.fusion.strategies.core.peak_protect_dist import PeakProtectDist
from src.components.fusion.strategies.core.peak_protect_ema import PeakProtectEma8Streak
from src.components.fusion.strategies.core.profit_lock import ProfitLock
from src.components.fusion.strategies.core.signal_hard_cap import SignalHardCapExit
from src.components.fusion.strategies.core.v22_fast_exit import V22FastExit
from src.components.fusion.strategies.core.v22_hard_cap import V22HardCap
from src.components.fusion.strategies.core.zombie_exit import ZombieExit

__all__ = [
    "AdaptiveTrailing",
    "AtrStopLoss",
    "FastExitLossLegacy",
    "HardStopExit",
    "MaCrossHybridExit",
    "MinHoldProtection",
    "PeakProtectDist",
    "PeakProtectEma8Streak",
    "ProfitLock",
    "SignalHardCapExit",
    "V22FastExit",
    "V22HardCap",
    "ZombieExit",
]
