"""Dense reference sweep: simulate EVERY Tier-A config for one SoC + trace.

The sweep is the ground truth: it tells us the true optimum (for
"% of optimum" claims) and lets the loop replay experiments by table
lookup instead of re-simulating. Builds run one at a time (they share
the champsim/ tree); simulations run in parallel.

Usage: python -m loop.sweep <soc_name> <trace_path>
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from loop.configs import all_configurations, config_name, make_config
from loop.simulate import build_binary, run_simulation

CHAMPSIM_ROOT = "champsim"
BASE_CONFIG = "champsim/champsim_config.json"
GENERATED_DIR = "configs/generated"
WARMUP_INSTRUCTIONS = 5_000_000
SIMULATION_INSTRUCTIONS = 10_000_000
PARALLEL_SIMULATIONS = 4


def sweep_path(soc_name, trace_path):
    trace_name = os.path.basename(trace_path).split(".champsimtrace")[0]
    return "results/sweep_{}_{}.json".format(soc_name, trace_name)


def load_sweep(path):
    if not os.path.exists(path):
        return {}
    with open(path) as sweep_file:
        return json.load(sweep_file)


def save_sweep(table, path):
    with open(path, "w") as sweep_file:
        json.dump(table, sweep_file, indent=2)


def run_sweep(soc_name, trace_path):
    path = sweep_path(soc_name, trace_path)
    table = load_sweep(path)
    pool = ThreadPoolExecutor(max_workers=PARALLEL_SIMULATIONS)
    pending = []

    for knobs in all_configurations():
        name = config_name(knobs, soc_name)
        if name in table:
            continue
        config_path = make_config(knobs, soc_name, BASE_CONFIG, GENERATED_DIR)
        binary_path = build_binary(config_path, CHAMPSIM_ROOT)
        future = pool.submit(run_simulation, binary_path, trace_path,
                             WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)
        pending.append((name, knobs, future))
        print("built", name, flush=True)

        # Bank any simulation that finished while we were building.
        pending = collect_finished(pending, table, path, wait=False)

    collect_finished(pending, table, path, wait=True)
    pool.shutdown()
    print("sweep complete:", len(table), "configs ->", path)


def collect_finished(pending, table, path, wait):
    still_pending = []
    for name, knobs, future in pending:
        if wait or future.done():
            table[name] = {"knobs": knobs, "metrics": future.result()}
            print("simulated", name, "ipc={:.4f}".format(table[name]["metrics"]["ipc"]), flush=True)
        else:
            still_pending.append((name, knobs, future))
    save_sweep(table, path)
    return still_pending


if __name__ == "__main__":
    run_sweep(sys.argv[1], sys.argv[2])
