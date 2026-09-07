"""ChampSim glue: turns (SoC, trace) into the generic `problem` dict loop.py needs.

This is the only place where the loop learns it is tuning caches.
"""

import fcntl
import json
import os
from concurrent.futures import ThreadPoolExecutor

from loop.configs import (KNOB_LOCATION, SPACES, all_configurations, build_config, config_name,
                          make_config, within_budget)
from loop.simulate import build_binary, run_simulation
from loop.sweep import sweep_path, load_sweep, save_sweep

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

# Workload fingerprint (Sep 7): rule conditions used to be read off the target
# chip's own baseline run, so a threshold learned on a small-cache chip ("L2 MPKI
# < 18") silently failed on a big-cache chip where the same effect existed. The
# fingerprint measures every workload ONCE on the SAME reference machine (stock
# ChampSim core, no prefetchers, LRU) at three L2/LLC capacities, so its numbers
# describe the program, not the chip. Conditions may only use these metrics.
FINGERPRINT_SOC = "B_midrange"
FINGERPRINT_PROBES = {"small": {"l2_sets": 512, "llc_sets": 1024},
                      "medium": {"l2_sets": 1024, "llc_sets": 2048},
                      "large": {"l2_sets": 2048, "llc_sets": 4096}}
FINGERPRINT_METRICS = ["L1D_mpki", "L2C_mpki", "LLC_mpki", "L2C_hit_ratio", "LLC_hit_ratio",
                       "LLC_over_L2C_mpki", "L2_capacity_sensitivity", "LLC_capacity_sensitivity"]


def enrich_metrics(metrics):
    """Add derived descriptors to a result row (in place; safe to call twice).

    Hit ratios say how much of the traffic each level absorbs; the LLC/L2 miss
    ratio says whether L2 misses find reuse in the LLC. Rules may condition on
    them; they describe the workload's access pattern better than MPKI alone.
    """
    if metrics is None:
        return None
    for level in ["L1D", "L2C", "LLC"]:
        hits = metrics.get(level + "_hits")
        misses = metrics.get(level + "_misses")
        if hits is None or misses is None:
            continue
        total = hits + misses
        if total > 0:
            metrics[level + "_hit_ratio"] = hits / total
        else:
            metrics[level + "_hit_ratio"] = 0.0
    if metrics.get("L2C_mpki") and metrics.get("LLC_mpki") is not None:
        if metrics["L2C_mpki"] > 0:
            metrics["LLC_over_L2C_mpki"] = metrics["LLC_mpki"] / metrics["L2C_mpki"]
    return metrics


class ChampSimProblem:
    """Answers evaluate() from the dense sweep table when it can; otherwise simulates.

    dispatch="local": build + run in this process, one config at a time.
    dispatch="chia":  build_from_config and simulate are dispatched as CHIA
                      tasks (ray must be initialised); a round's configs run
                      in parallel on whatever workers advertise "champsim".
    """

    def __init__(self, soc_name, trace_path, allow_simulation, dispatch, space_name):
        self.soc_name = soc_name
        self.trace_path = trace_path
        self.allow_simulation = allow_simulation
        self.dispatch = dispatch
        self.space_name = space_name
        # Tier A reads the dense sweep; Tier B grows a cache shared by all processes.
        self.table_path = sweep_path(soc_name, trace_path)
        if space_name != "A":
            self.table_path = self.table_path.replace("results/sweep_", "results/tier{}_".format(space_name))
        self.sweep_table = load_sweep(self.table_path)
        self.simulations_run = 0
        # A round's configs simulate at the same time (threads; each is a subprocess).
        self.local_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SIM_THREADS", "4")))

    def evaluate(self, knobs):
        return self.evaluate_many([knobs])[0]

    def evaluate_many(self, knobs_list):
        waiters = self.evaluate_many_async(knobs_list)
        results = []
        for waiter in waiters:
            results.append(waiter())
        return results

    def evaluate_many_async(self, knobs_list):
        """Start every missing simulation now; return one callable per design that
        blocks for its result (so a suite can fan out across workloads first)."""
        waiters = []
        self.sweep_table = load_sweep(self.table_path)      # pick up other processes' results
        for knobs in knobs_list:
            self.simulations_run += 1
            name = config_name(knobs, self.soc_name)
            if name in self.sweep_table:
                if self.sweep_table[name]["metrics"] is None:
                    raise RuntimeError("config crashed in the sweep: " + name)
                waiters.append(self.ready(enrich_metrics(dict(self.sweep_table[name]["metrics"]))))
                continue
            if not self.allow_simulation:
                raise KeyError("not in sweep table and simulation disabled: " + name)
            if self.dispatch == "chia":
                waiters.append(self.wait_chia(knobs, self.dispatch_chia(knobs)))
            else:
                waiters.append(self.wait_local(knobs, self.local_pool.submit(self.simulate_here, knobs)))
        return waiters

    def ready(self, metrics):
        def waiter():
            return metrics
        return waiter

    def wait_local(self, knobs, future):
        def waiter():
            metrics = future.result()
            self.remember(knobs, metrics)
            return enrich_metrics(dict(metrics))
        return waiter

    def wait_chia(self, knobs, future):
        def waiter():
            from chia.base.ChiaFunction import get
            metrics = get(future)
            self.remember(knobs, metrics)
            return enrich_metrics(dict(metrics))
        return waiter

    def remember(self, knobs, metrics):
        """Append one result to the shared table under a lock (many processes write it)."""
        name = config_name(knobs, self.soc_name)
        self.sweep_table[name] = {"knobs": knobs, "metrics": metrics}
        with open(self.table_path + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            table = load_sweep(self.table_path)
            table[name] = {"knobs": knobs, "metrics": metrics}
            save_sweep(table, self.table_path)          # atomic rename: readers never see a partial file
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    def simulate_here(self, knobs):
        config_path = make_config(knobs, self.soc_name, BASE_CONFIG, GENERATED_DIR, self.space_name)
        binary_path = build_binary(config_path, CHAMPSIM_ROOT)
        return run_simulation(binary_path, self.trace_path,
                              WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)

    def dispatch_chia(self, knobs):
        """Two chained CHIA tasks: build -> simulate. Returns the simulate future."""
        from loop.chia_nodes import build_from_config, simulate
        config = build_config(knobs, self.soc_name, BASE_CONFIG, self.space_name)
        binary_future = build_from_config.chia_remote(config, os.path.abspath(CHAMPSIM_ROOT))
        return simulate.chia_remote(binary_future, os.path.abspath(self.trace_path),
                                    WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS,
                                    _chia_tag=config["executable_name"])


def profiled_baseline(soc_name, space_name):
    """The chip as profiled: BASELINE_KNOBS plus every other knob of the space at
    the SoC profile's own value, read back from the built config."""
    profiled = build_config(BASELINE_KNOBS, soc_name, BASE_CONFIG, space_name)
    baseline = dict(BASELINE_KNOBS)
    for knob in SPACES[space_name]:
        if knob not in baseline:
            section, field = KNOB_LOCATION[knob]
            baseline[knob] = profiled[section][field]
    return baseline


def workload_fingerprint(trace_path, allow_simulation, dispatch, space_name):
    """Chip-independent descriptors of one workload: the reference machine's
    metrics at the medium probe, plus how much of the L2 / LLC misses a 4x
    capacity step (small -> large) removes. Three simulations per workload,
    cached in the reference SoC's result table like any other design."""
    holder = ChampSimProblem(FINGERPRINT_SOC, trace_path, allow_simulation, dispatch, space_name)
    reference = profiled_baseline(FINGERPRINT_SOC, space_name)
    for knob in ["l1d_prefetcher", "l2_prefetcher", "llc_prefetcher"]:
        if knob in reference:
            reference[knob] = "no"
    if "llc_replacement" in reference:
        reference["llc_replacement"] = "lru"
    probe_names = []
    probe_knobs = []
    for probe_name, sizes in FINGERPRINT_PROBES.items():
        knobs = dict(reference)
        for knob, value in sizes.items():
            knobs[knob] = value
        probe_names.append(probe_name)
        probe_knobs.append(knobs)
    metrics_list = holder.evaluate_many(probe_knobs)
    probes = dict(zip(probe_names, metrics_list))

    fingerprint = {}
    medium = probes["medium"]
    for metric in ["L1D_mpki", "L2C_mpki", "LLC_mpki", "L2C_hit_ratio", "LLC_hit_ratio", "LLC_over_L2C_mpki"]:
        fingerprint[metric] = medium.get(metric, 0.0)
    for level, metric in [("L2", "L2C_mpki"), ("LLC", "LLC_mpki")]:
        small_value = probes["small"].get(metric, 0.0)
        large_value = probes["large"].get(metric, 0.0)
        sensitivity = 0.0
        if small_value > 0:
            sensitivity = (small_value - large_value) / small_value
        fingerprint[level + "_capacity_sensitivity"] = sensitivity
    return fingerprint


def make_problem(soc_name, trace_path, allow_simulation=True, dispatch=None, space_name="A"):
    if dispatch is None:
        dispatch = DEFAULT_DISPATCH
    trace_name = os.path.basename(trace_path).split(".")[1].split("_")[0]
    holder = ChampSimProblem(soc_name, trace_path, allow_simulation, dispatch, space_name)
    space = SPACES[space_name]
    baseline = profiled_baseline(soc_name, space_name)

    # Conditions live on the workload fingerprint. Tier A reads dense sweep tables
    # with simulation disabled and has no probes: it keeps the old chip-relative
    # descriptors (the baseline run's own metrics, filled in by the loop).
    descriptors = None
    condition_metrics = ["L1D_mpki", "L2C_mpki", "LLC_mpki",
                         "L2C_hit_ratio", "LLC_hit_ratio", "LLC_over_L2C_mpki"]
    try:
        descriptors = workload_fingerprint(trace_path, allow_simulation, dispatch, space_name)
        condition_metrics = list(FINGERPRINT_METRICS)
    except KeyError:
        pass

    candidates = {}
    for knobs in all_configurations(space):
        name = config_name(knobs, soc_name)
        crashed = name in holder.sweep_table and holder.sweep_table[name]["metrics"] is None
        if within_budget(knobs, soc_name, BASE_CONFIG, space_name) and not crashed:
            candidates[name] = knobs

    return {
        "name": soc_name + "/" + trace_name,
        "search_space": space,
        "candidates": candidates,
        "baseline": baseline,
        "evaluate": holder.evaluate,
        "evaluate_many": holder.evaluate_many,
        "objective": "ipc",
        "table_metrics": ["ipc", "L1D_mpki", "L2C_mpki", "LLC_mpki", "LLC_hit_ratio"],
        # Rule conditions must describe the WORKLOAD, not the chip: they are checked
        # on `descriptors` (the fingerprint), never on the chip's own runs.
        "condition_metrics": condition_metrics,
        "descriptors": descriptors,
        "holder": holder,
    }


SUITE_AGGREGATE_METRICS = ["L1D_mpki", "L2C_mpki", "LLC_mpki", "L2C_hit_ratio", "LLC_hit_ratio", "LLC_over_L2C_mpki"]


def trace_short_name(trace_path):
    return os.path.basename(trace_path).split(".")[1].split("_")[0]


def make_suite_problem(soc_name, trace_paths, allow_simulation=True, dispatch=None, space_name="C"):
    """One design is scored on a SUITE of workloads: the objective is the geometric
    mean of per-workload IPC (one simulation per workload per design). This is how
    a design team scores a hierarchy; no chip is built for one program.

    Rule conditions see the suite through aggregates: the mean of each workload
    descriptor across the suite (plain name) and its max ("max_" prefix)."""
    if len(trace_paths) == 1:
        return make_problem(soc_name, trace_paths[0], allow_simulation, dispatch, space_name)
    single_problems = []
    for trace_path in trace_paths:
        single_problems.append(make_problem(soc_name, trace_path, allow_simulation, dispatch, space_name))
    first = single_problems[0]
    short_names = []
    for trace_path in trace_paths:
        short_names.append(trace_short_name(trace_path))

    def evaluate_many(knobs_list):
        # Fan out: every (design, workload) pair runs at once through its own holder.
        per_trace_results = []
        for problem in single_problems:
            per_trace_results.append(problem["holder"].evaluate_many_async(knobs_list))
        results = []
        for index in range(len(knobs_list)):
            per_trace = []
            for trace_index in range(len(single_problems)):
                per_trace.append(per_trace_results[trace_index][index]())
            results.append(aggregate_suite(per_trace, short_names))
        return results

    def evaluate(knobs):
        return evaluate_many([knobs])[0]

    # Candidates: a design must fit the budget (same for every workload); crashed
    # designs on any workload are excluded.
    candidates = dict(first["candidates"])
    for problem in single_problems[1:]:
        for name in list(candidates.keys()):
            if name not in problem["candidates"]:
                del candidates[name]

    table_metrics = ["ipc"]
    for short in short_names:
        table_metrics.append(short + ":ipc")
    table_metrics = table_metrics + ["L2C_mpki", "LLC_mpki", "LLC_hit_ratio"]
    # Suite descriptors: mean and max of each workload's fingerprint metric.
    base_metrics = list(first["condition_metrics"])
    condition_metrics = list(base_metrics)
    for metric in base_metrics:
        condition_metrics.append("max_" + metric)
    descriptors = None
    if first["descriptors"] is not None:
        descriptors = {}
        for metric in base_metrics:
            values = []
            for problem in single_problems:
                values.append(problem["descriptors"][metric])
            descriptors[metric] = sum(values) / len(values)
            descriptors["max_" + metric] = max(values)

    return {
        "name": soc_name + "/suite-" + "+".join(short_names),
        "search_space": first["search_space"],
        "candidates": candidates,
        "baseline": first["baseline"],
        "evaluate": evaluate,
        "evaluate_many": evaluate_many,
        "objective": "ipc",
        "table_metrics": table_metrics,
        "condition_metrics": condition_metrics,
        "descriptors": descriptors,
        "holder": first["holder"],
        "holders": [problem["holder"] for problem in single_problems],
    }


def aggregate_suite(per_trace_metrics, short_names):
    """Geometric-mean IPC plus mean/max of every workload descriptor; per-workload
    values kept under "<trace>:<metric>" for the analyst's table."""
    import math
    combined = {}
    log_sum = 0.0
    for short, metrics in zip(short_names, per_trace_metrics):
        log_sum += math.log(max(metrics["ipc"], 1e-9))
        for metric in metrics:
            combined[short + ":" + metric] = metrics[metric]
    combined["ipc"] = math.exp(log_sum / len(per_trace_metrics))
    for metric in SUITE_AGGREGATE_METRICS:
        values = []
        for metrics in per_trace_metrics:
            if metric in metrics:
                values.append(metrics[metric])
        if len(values) > 0:
            combined[metric] = sum(values) / len(values)
            combined["max_" + metric] = max(values)
    return combined
