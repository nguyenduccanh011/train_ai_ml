from functools import partial

from src.components.runners._lineage_v34 import run_lineage
from src.components.runners.generic_fusion import (
    FUSION_RUNNER_DEFS,
    run_fusion,
    trades_to_v19_3_dataframe,
    trades_to_v22_dataframe,
)
from src.components.runners.rule_runner import run_rule_baseline, trades_to_dataframe
from src.components.runners.runner_registry import RUNNER_DEFS
from src.components.runners.v34_runner import run_v34, trades_to_v34_dataframe

run_v19_3 = partial(run_fusion, FUSION_RUNNER_DEFS["v19_3"])
run_v22 = partial(run_fusion, FUSION_RUNNER_DEFS["v22"])
run_v32 = partial(run_lineage, RUNNER_DEFS["v32"])
run_v35b = partial(run_lineage, RUNNER_DEFS["v35b"])
run_v37a = partial(run_lineage, RUNNER_DEFS["v37a"])
run_v37a_exit = partial(run_lineage, RUNNER_DEFS["v37a_exit"])
run_v37d = partial(run_lineage, RUNNER_DEFS["v37d"])
run_v39d = partial(run_lineage, RUNNER_DEFS["v39d"])
run_v42_a = partial(run_lineage, RUNNER_DEFS["v42_a"])

trades_to_v32_dataframe = trades_to_v34_dataframe
trades_to_v35b_dataframe = trades_to_v34_dataframe
trades_to_v37a_dataframe = trades_to_v34_dataframe
trades_to_v37a_exit_dataframe = trades_to_v34_dataframe
trades_to_v37d_dataframe = trades_to_v34_dataframe
trades_to_v39d_dataframe = trades_to_v34_dataframe
trades_to_v42_a_dataframe = trades_to_v34_dataframe

__all__ = [
    "run_rule_baseline",
    "run_v19_3",
    "run_v22",
    "run_v32",
    "run_v34",
    "run_v35b",
    "run_v37a",
    "run_v37a_exit",
    "run_v37d",
    "run_v39d",
    "run_v42_a",
    "trades_to_dataframe",
    "trades_to_v19_3_dataframe",
    "trades_to_v22_dataframe",
    "trades_to_v32_dataframe",
    "trades_to_v34_dataframe",
    "trades_to_v35b_dataframe",
    "trades_to_v37a_dataframe",
    "trades_to_v37a_exit_dataframe",
    "trades_to_v37d_dataframe",
    "trades_to_v39d_dataframe",
    "trades_to_v42_a_dataframe",
]
