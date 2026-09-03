"""ChampSim glue: turns (SoC, trace) into the generic `problem` dict loop.py needs.

This is the only place where the loop learns it is tuning caches.
"""

import os

from loop.configs import (SEARCH_SPACE, all_configurations, build_config, config_name,
                          make_config, within_budget)
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

# "local" or "chia"; run_chia.py flips this so experiment.py needs no changes.
DEFAULT_DISPATCH = "local"


class ChampSimProblem:
    """Answers evaluate() from the dense sweep table when it can; otherwise simulates.

    dispatch="local": build + run in this process, one config at a time.
    dispatch="chia":  build_from_config and simulate are dispatched as CHIA
                      tasks (ray must be initialised); a round's configs run
                      in parallel on whatever workers advertise "champsim".
    """

    def __init__(self, soc_name, trace_path, allow_simulation, dispatch):
        self.soc_name = soc_name
        self.trace_path = trace_path
        self.allow_simulation = allow_simulation
        self.dispatch = dispatch
        self.sweep_table = load_sweep(sweep_path(soc_name, trace_path))
        self.simulations_run = 0

    def evaluate(self, knobs):
        return self.evaluate_many([knobs])[0]

    def evaluate_many(self, knobs_list):
        results = []
        pending = []                       # (index, knobs, future) for configs not in the table
        for index, knobs in enumerate(knobs_list):
            self.simulations_run += 1
            name = config_name(knobs, self.soc_name)
            if name in self.sweep_table:
                if self.sweep_table[name]["metrics"] is None:
                    raise RuntimeError("config crashed in the sweep: " + name)
                results.append(self.sweep_table[name]["metrics"])
                continue
            if not self.allow_simulation:
                raise KeyError("not in sweep table and simulation disabled: " + name)
            results.append(None)
            if self.dispatch == "chia":
                pending.append((index, self.dispatch_chia(knobs)))
            else:
                results[index] = self.simulate_here(knobs)

        if len(pending) > 0:
            from chia.base.ChiaFunction import get
            for index, future in pending:
                results[index] = get(future)
        return results

    def simulate_here(self, knobs):
        config_path = make_config(knobs, self.soc_name, BASE_CONFIG, GENERATED_DIR)
        binary_path = build_binary(config_path, CHAMPSIM_ROOT)
        return run_simulation(binary_path, self.trace_path,
                              WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)

    def dispatch_chia(self, knobs):
        """Two chained CHIA tasks: build -> simulate. Returns the simulate future."""
        from loop.chia_nodes import build_from_config, simulate
        config = build_config(knobs, self.soc_name, BASE_CONFIG)
        binary_future = build_from_config.chia_remote(config, os.path.abspath(CHAMPSIM_ROOT))
        return simulate.chia_remote(binary_future, os.path.abspath(self.trace_path),
                                    WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS,
                                    _chia_tag=config["executable_name"])


def make_problem(soc_name, trace_path, allow_simulation=True, dispatch=None):
    if dispatch is None:
        dispatch = DEFAULT_DISPATCH
    trace_name = os.path.basename(trace_path).split(".")[1].split("_")[0]
    holder = ChampSimProblem(soc_name, trace_path, allow_simulation, dispatch)

    candidates = {}
    for knobs in all_configurations():
        name = config_name(knobs, soc_name)
        crashed = name in holder.sweep_table and holder.sweep_table[name]["metrics"] is None
        if within_budget(knobs, soc_name, BASE_CONFIG) and not crashed:
            candidates[name] = knobs

    return {
        "name": soc_name + "/" + trace_name,
        "search_space": SEARCH_SPACE,
        "candidates": candidates,
        "baseline": BASELINE_KNOBS,
        "evaluate": holder.evaluate,
        "evaluate_many": holder.evaluate_many,
        "objective": "ipc",
        "table_metrics": ["ipc", "L2C_mpki", "LLC_mpki"],
        # Rule conditions must describe the WORKLOAD, not the chip: miss rates
        # travel across SoCs, IPC does not (P1, invariance).
        "condition_metrics": ["L1D_mpki", "L2C_mpki", "LLC_mpki"],
        "holder": holder,
    }
