"""Entry point for running the experiment as a CHIA loop.

  python -m loop.run_chia local  traces/...xz   # ray started in-process on this Mac
  python -m loop.run_chia auto   traces/...xz   # attach to a cluster started by `chia up`

Simulations are dispatched as CHIA tasks (parallel per round), the analyst
runs through chia.models.vertex, and CHIA's profiler records the task graph
and per-call LLM token counts for `chia viz-profile`.
"""

import sys
import time

import ray
from chia.trace.profiler import start_collector

from loop import analyst, experiment
from loop.chia_nodes import AnalystNode
from loop import champsim_problem


def main(mode, traces):
    if mode == "local":
        ray.init(resources={"champsim": 4, "champsim_build": 1, "vertex_creds": 1},
                 include_dashboard=False, logging_level="ERROR")
    else:
        ray.init(address="auto")
    start_collector(log_dir="results/chia_profiles")

    analyst.use_chia_node(AnalystNode(model=analyst.MODEL, project=analyst.GCP_PROJECT))
    champsim_problem.DEFAULT_DISPATCH = "chia"

    stamp = time.strftime("%Y%m%d_%H%M")
    experiment.run_experiment(train_socs=["A_mobile", "B_midrange"], test_soc="C_server",
                              traces=traces, rounds=10, per_round=2, seeds=3,
                              output_path="results/experiment_{}.json".format(stamp))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
