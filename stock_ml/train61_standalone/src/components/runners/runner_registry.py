from __future__ import annotations

import copy
import importlib
from collections.abc import Callable
from functools import cache
from typing import Any

from src.components.runners._lineage_v34 import RunnerDef

V32_TARGET: dict[str, Any] = {
    "type": "early_wave",
    "forward_window": 8,
    "short_window": 8,
    "long_window": 20,
    "gain_threshold": 0.05,
    "loss_threshold": 0.04,
    "classes": 3,
}


@cache
def _load_runner_fn(module: str, fn: str) -> Callable[..., Any]:
    return getattr(importlib.import_module(module), fn)


def _lazy(module: str, fn: str) -> Callable[..., Any]:
    def _call(*args: Any, **kwargs: Any) -> Any:
        return _load_runner_fn(module, fn)(*args, **kwargs)

    return _call


RUNNER_DEFS: dict[str, RunnerDef] = {
    "v32": RunnerDef(
        version_key="v32",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v32"),
        entry_reason="v32",
        feature_set_default="leading_v3",
        target_default=copy.deepcopy(V32_TARGET),
    ),
    "v35b": RunnerDef(
        version_key="v35b",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v35b"),
        entry_reason="v35b",
    ),
    "v37a": RunnerDef(
        version_key="v37a",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v37a"),
        entry_reason="v37a",
    ),
    "v37a_exit": RunnerDef(
        version_key="v37a_exit",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v37a"),
        entry_reason="v37a_exit",
    ),
    "v37d": RunnerDef(
        version_key="v37d",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v37d"),
        entry_reason="v37d",
    ),
    "v39d": RunnerDef(
        version_key="v39d",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v39d"),
        entry_reason="v39d",
    ),
    "v42_a": RunnerDef(
        version_key="v42_a",
        backtest_fn=_lazy("src.components.runners.lineage_backtests", "backtest_v42"),
        entry_reason="v42_a",
    ),
}


def list_runners() -> dict[str, str]:
    from src.components.runners.generic_fusion import FUSION_RUNNER_DEFS
    from src.pipeline.orchestrator import CHAMPION_RUNNER_MAP

    runners = {name: "lineage" for name in RUNNER_DEFS}
    runners.update({name: "fusion" for name in FUSION_RUNNER_DEFS})
    runners.update({name: "champion" for name in CHAMPION_RUNNER_MAP})
    return runners


__all__ = ["RUNNER_DEFS", "list_runners"]
