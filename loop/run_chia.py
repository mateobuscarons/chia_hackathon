"""Run a cell as a CHIA loop.

  LOOP_DISPATCH is set here, so every build and simulation is a CHIA task, the
  analyst goes through chia.models.vertex, arm runs are Ray tasks, and CHIA's
  profiler records the task graph and per-call token counts for `chia viz-profile`.

  python -m loop.run_chia local <cell> <tag> [playbook.json|-] [arm,arm,...]   # Ray in-process on this machine
  python -m loop.run_chia auto  <cell> <tag> [playbook.json|-] [arm,arm,...]   # attach to a cluster started by `chia up`

Cells are the ones in loop.run. Profiles land in results/chia_profiles/.
"""

import os
import sys

os.environ["LOOP_DISPATCH"] = "chia"       # before loop modules read it

import ray
from chia.trace.profiler import start_collector

from loop import run
from loop.champsim_problem import CHAMPSIM_ROOT
from loop.simulate import tree_paths


def main(mode, cell_name, tag, playbook_path, arms):
    if mode == "local":
        cores = os.cpu_count()
        # One simulation slot per core; one build slot per ChampSim tree copy. Ray
        # also charges each task one CPU, so the CPU pool is cores + builds: a
        # build never waits for a simulation to free a CPU.
        builds = len(tree_paths(CHAMPSIM_ROOT))
        ray.init(num_cpus=cores + builds,
                 resources={"champsim": cores, "champsim_build": builds, "vertex_creds": 1},
                 include_dashboard=False, logging_level="ERROR")
    else:
        ray.init(address="auto")
    start_collector(log_dir="results/chia_profiles")
    run.run_cell(cell_name, tag, playbook_path, arms)


if __name__ == "__main__":
    playbook_path = None
    if len(sys.argv) > 4 and sys.argv[4] != "-":
        playbook_path = sys.argv[4]
    arms = None
    if len(sys.argv) > 5:
        arms = sys.argv[5].split(",")
    main(sys.argv[1], sys.argv[2], sys.argv[3], playbook_path, arms)
