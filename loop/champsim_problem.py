"""ChampSim glue: turns (SoC, trace) into the generic `problem` dict loop.py needs.

This is the only place where the loop learns it is tuning caches.
"""

import os

from loop.configs import SEARCH_SPACE, all_configurations, config_name, make_config, within_budget
from loop.simulate import build_binary, run_simulation
from loop.sweep import sweep_path, load_sweep

CHAMPSIM_ROOT = "champsim"
BASE_CONFIG = "champsim/champsim_config.json"
GENERATED_DIR = "configs/generated"
WARMUP_INSTRUCTIONS = 5_000_000
SIMULATION_INSTRUCTIONS = 10_000_000

# The untouched chip: every SoC/trace loop starts by measuring this.
BASELINE_KNOBS = {"l2_sets": 1024, "llc_sets": 2048,
                  "l2_prefetcher": "no", "llc_replacement": "lru"}


class ChampSimProblem:
    """Holds the sweep table so evaluate() can answer from it without re-simulating."""

    def __init__(self, soc_name, trace_path, allow_simulation):
        self.soc_name = soc_name
        self.trace_path = trace_path
        self.allow_simulation = allow_simulation
        self.sweep_table = load_sweep(sweep_path(soc_name, trace_path))
        self.simulations_run = 0

    def evaluate(self, knobs):
        """Metrics for one config: from the dense sweep if we have it, else simulate."""
        self.simulations_run += 1
        name = config_name(knobs, self.soc_name)
        if name in self.sweep_table:
            return self.sweep_table[name]["metrics"]
        if not self.allow_simulation:
            raise KeyError("not in sweep table and simulation disabled: " + name)
        config_path = make_config(knobs, self.soc_name, BASE_CONFIG, GENERATED_DIR)
        binary_path = build_binary(config_path, CHAMPSIM_ROOT)
        return run_simulation(binary_path, self.trace_path,
                              WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)


def make_problem(soc_name, trace_path, allow_simulation=True):
    trace_name = os.path.basename(trace_path).split(".")[1].split("_")[0]
    holder = ChampSimProblem(soc_name, trace_path, allow_simulation)

    candidates = {}
    for knobs in all_configurations():
        if within_budget(knobs, soc_name, BASE_CONFIG):
            candidates[config_name(knobs, soc_name)] = knobs

    return {
        "name": soc_name + "/" + trace_name,
        "search_space": SEARCH_SPACE,
        "candidates": candidates,
        "baseline": BASELINE_KNOBS,
        "evaluate": holder.evaluate,
        "objective": "ipc",
        "table_metrics": ["ipc", "L2C_mpki", "LLC_mpki"],
        "holder": holder,
    }
