"""ChampSim glue: turns (SoC, trace) into the generic `problem` dict loop.py needs.

This is the only place where the loop learns it is tuning caches.
"""

import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from loop.configs import (KNOB_LOCATION, SPACES, build_config, config_name, in_space, make_config,
                          random_feasible_designs, typed_knobs, within_budget)
from loop.socs import AREA_BUDGET_KB, describe, num_cores
from loop import trace_profile
from loop.simulate import build_binary, run_simulation

CHAMPSIM_ROOT = "champsim"
BASE_CONFIG = "champsim/champsim_config.json"
GENERATED_DIR = "configs/generated"
# The search runs short simulations; the fidelity check and the final validation
# run the same designs longer (LOOP_WARMUP / LOOP_SIM, in instructions) and land
# in their own tables, so long and short rows never mix.
DEFAULT_WARMUP = 5_000_000
DEFAULT_SIMULATION = 10_000_000
WARMUP_INSTRUCTIONS = int(os.environ.get("LOOP_WARMUP", DEFAULT_WARMUP))
SIMULATION_INSTRUCTIONS = int(os.environ.get("LOOP_SIM", DEFAULT_SIMULATION))

# The untouched chip: every SoC/trace loop starts by measuring this. (ChampSim's
# stock config names no L2 replacement policy and uses LRU; the knob says so.)
BASELINE_KNOBS = {"l2_sets": 1024, "llc_sets": 2048,
                  "l2_prefetcher": "no", "llc_replacement": "lru", "l2_replacement": "lru"}

# The GP and random arms score a fixed random sample of the feasible designs (the
# space has millions); the agent arms may name ANY feasible design (is_candidate).
CANDIDATE_POOL = 20000

# "local" or "chia" (LOOP_DISPATCH=chia is set by run_chia): with "chia" every
# build and simulation is a CHIA task, so experiment.py needs no changes.
DEFAULT_DISPATCH = os.environ.get("LOOP_DISPATCH", "local")

# Workload descriptors: a rule condition sees (a) the PROGRAM, profiled once from
# its trace with no simulator (loop.trace_profile), and (b) that program against
# THIS chip's public geometry: working set over cache size, predicted LRU miss
# ratio at the chip's L2 / LLC, and the misses a bigger cache could still remove
# within the chip's area budget. The chip's own measured speed is never a
# condition, so a threshold means the same thing on every chip.
DESCRIPTOR_METRICS = ["mem_accesses_per_kinstr", "write_fraction", "footprint_kb",
                      "stride_regular_fraction", "reuse_local_fraction",
                      "l2_footprint_ratio", "llc_footprint_ratio",
                      "pred_l1d_miss_ratio", "pred_l2_miss_ratio", "pred_llc_miss_ratio",
                      "movable_l2_mpki", "movable_llc_mpki"]


def largest_affordable_kb(search_space, soc_name, llc_share=1.0):
    """The biggest L2 and the biggest L2 + (own share of the) LLC capacity, in KB,
    any design in the space can have on this chip within its area budget."""
    largest_l2_kb = 0.0
    largest_total_kb = 0.0
    for l2_sets in search_space["l2_sets"]:
        for l2_ways in search_space["l2_ways"]:
            for llc_sets in search_space["llc_sets"]:
                for llc_ways in search_space["llc_ways"]:
                    knobs = {"l2_sets": l2_sets, "l2_ways": l2_ways,
                             "llc_sets": llc_sets, "llc_ways": llc_ways}
                    if not within_budget(knobs, soc_name, BASE_CONFIG):
                        continue
                    l2_kb = l2_sets * l2_ways * 64 / 1024.0
                    total_kb = l2_kb + llc_sets * llc_ways * 64 / 1024.0 * llc_share
                    if l2_kb > largest_l2_kb:
                        largest_l2_kb = l2_kb
                    if total_kb > largest_total_kb:
                        largest_total_kb = total_kb
    return largest_l2_kb, largest_total_kb


def chip_descriptors(profile, baseline_knobs, search_space, soc_name, llc_share=1.0):
    """The workload profile read against one chip's cache geometry (its baseline
    knobs: sets x ways x 64 B per level) and area budget. No simulation involved.
    llc_share: the fraction of the LLC this program can count on (1 on a single
    core; 1/N when N programs share it - an approximation, since footprint theory
    predicts a dedicated cache's miss ratio)."""
    l1d_kb = baseline_knobs["l1d_sets"] * baseline_knobs["l1d_ways"] * 64 / 1024.0
    l2_kb = baseline_knobs["l2_sets"] * baseline_knobs["l2_ways"] * 64 / 1024.0
    llc_kb = baseline_knobs["llc_sets"] * baseline_knobs["llc_ways"] * 64 / 1024.0 * llc_share
    largest_l2_kb, largest_total_kb = largest_affordable_kb(search_space, soc_name, llc_share)
    descriptors = {}
    for metric in ["mem_accesses_per_kinstr", "write_fraction", "footprint_kb",
                   "stride_regular_fraction", "reuse_local_fraction"]:
        descriptors[metric] = profile[metric]
    descriptors["l2_footprint_ratio"] = profile["footprint_kb"] / l2_kb
    descriptors["llc_footprint_ratio"] = profile["footprint_kb"] / (l2_kb + llc_kb)
    descriptors["pred_l1d_miss_ratio"] = trace_profile.predicted_miss_ratio(profile, l1d_kb)
    descriptors["pred_l2_miss_ratio"] = trace_profile.predicted_miss_ratio(profile, l2_kb)
    descriptors["pred_llc_miss_ratio"] = trace_profile.predicted_miss_ratio(profile, l2_kb + llc_kb)
    # Movable MPKI: the misses per 1000 instructions a bigger cache would remove,
    # from this chip's baseline capacity to the largest its budget affords. A huge
    # footprint read once front to back has a large footprint ratio and movable
    # MPKI near zero: capacity cannot help it, and this is the descriptor that says so.
    accesses = profile["mem_accesses_per_kinstr"]
    l2_drop = descriptors["pred_l2_miss_ratio"] - trace_profile.predicted_miss_ratio(profile, largest_l2_kb)
    llc_drop = descriptors["pred_llc_miss_ratio"] - trace_profile.predicted_miss_ratio(profile, largest_total_kb)
    descriptors["movable_l2_mpki"] = accesses * l2_drop
    descriptors["movable_llc_mpki"] = accesses * llc_drop
    return descriptors


# Descriptors that add up across the programs of a mix (N cores sharing an LLC
# collectively need the sum of their working sets); every other descriptor is a
# rate and is averaged, weighted by each program's memory accesses.
MIX_SUMMED = ["footprint_kb"]
MIX_PLAIN_MEAN = ["mem_accesses_per_kinstr"]


def mix_descriptors(profiles, baseline_knobs, search_space, soc_name):
    """Descriptors of a mix of programs, one per core, on a chip with private L2s
    and one shared LLC: each program is read against its own L2 and its 1/N share
    of the LLC, then the per-program descriptors are combined."""
    cores = len(profiles)
    per_program = []
    for profile in profiles:
        per_program.append(chip_descriptors(profile, baseline_knobs, search_space, soc_name,
                                            llc_share=1.0 / cores))
    weights = []
    for descriptors in per_program:
        weights.append(descriptors["mem_accesses_per_kinstr"])
    total_weight = sum(weights)
    combined = {}
    for metric in DESCRIPTOR_METRICS:
        values = []
        for descriptors in per_program:
            values.append(descriptors[metric])
        if metric in MIX_SUMMED:
            combined[metric] = sum(values)
        elif metric in MIX_PLAIN_MEAN:
            combined[metric] = sum(values) / cores
        else:
            weighted = 0.0
            for value, weight in zip(values, weights):
                weighted += value * weight
            combined[metric] = weighted / total_weight
    # The mix's footprint over the capacity all its programs see together.
    l2_kb = baseline_knobs["l2_sets"] * baseline_knobs["l2_ways"] * 64 / 1024.0
    llc_kb = baseline_knobs["llc_sets"] * baseline_knobs["llc_ways"] * 64 / 1024.0
    combined["llc_footprint_ratio"] = combined["footprint_kb"] / (cores * l2_kb + llc_kb)
    return combined


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


def table_path(soc_name, trace_path):
    """The shared result table for one chip and one trace (or one mix of traces):
    the simulation cache every process reads and appends to. Non-default
    simulation lengths get their own table."""
    if isinstance(trace_path, list):
        trace_name = "mix-" + trace_short_name(trace_path)
    else:
        trace_name = os.path.basename(trace_path).split(".champsimtrace")[0]
    suffix = ""
    if WARMUP_INSTRUCTIONS != DEFAULT_WARMUP or SIMULATION_INSTRUCTIONS != DEFAULT_SIMULATION:
        suffix = "_w{}M_s{}M".format(WARMUP_INSTRUCTIONS // 1_000_000, SIMULATION_INSTRUCTIONS // 1_000_000)
    return "results/tierC_{}_{}{}.json".format(soc_name, trace_name, suffix)


def trace_short_name(trace_path):
    """The trace's whole name, without directory and without the compression
    suffix: "605.mcf_s-665B", "bfs.urand-36B", "llama2.c-stories15M.2". The
    names joined with "+" for a mix (a list of traces). No parsing: a shortened
    name collides as soon as two traces share a prefix (the four llama2 model
    sizes all begin "llama2.c"), and the agent reads this name in its prompt."""
    if isinstance(trace_path, list):
        parts = []
        for member in trace_path:
            parts.append(trace_short_name(member))
        return "+".join(parts)
    return os.path.basename(trace_path).split(".champsimtrace")[0]


def load_table(path):
    """Many processes read this table while one rewrites it; a reader may catch a
    half-written file. Retry briefly instead of crashing the arm."""
    if not os.path.exists(path):
        return {}
    for attempt in range(10):
        try:
            with open(path) as table_file:
                return json.load(table_file)
        except json.JSONDecodeError:
            time.sleep(0.2)
    with open(path) as table_file:
        return json.load(table_file)


def save_table(table, path):
    """Atomic: write next to the target, then rename, so readers never see a partial file."""
    temporary_path = path + ".tmp.{}".format(os.getpid())
    with open(temporary_path, "w") as table_file:
        json.dump(table, table_file, indent=2)
    os.replace(temporary_path, path)


class ChampSimProblem:
    """Answers evaluate() from the result table when it can; otherwise simulates.

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
        self.table_path = table_path(soc_name, trace_path)
        self.sweep_table = load_table(self.table_path)
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
        self.sweep_table = load_table(self.table_path)      # pick up other processes' results
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
            table = load_table(self.table_path)
            table[name] = {"knobs": knobs, "metrics": metrics}
            save_table(table, self.table_path)          # atomic rename: readers never see a partial file
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
        trace_paths = self.trace_path
        if isinstance(trace_paths, str):
            trace_paths = [trace_paths]
        absolute_paths = []
        for trace_path in trace_paths:
            absolute_paths.append(os.path.abspath(trace_path))
        return simulate.chia_remote(binary_future, absolute_paths,
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


def make_problem(soc_name, trace_path, allow_simulation=True, dispatch=None, space_name="C", start_knobs=None):
    """One workload on one chip. `trace_path` is one trace, or a list with one trace
    per core on a multi-core chip (a mix). `start_knobs`: the design every arm starts
    from (default: the untouched chip); descriptors are read against its geometry."""
    if dispatch is None:
        dispatch = DEFAULT_DISPATCH
    if isinstance(trace_path, list) and len(trace_path) != num_cores(soc_name):
        raise ValueError("{} has {} cores; a mix needs one trace per core".format(soc_name, num_cores(soc_name)))
    trace_name = trace_short_name(trace_path)
    holder = ChampSimProblem(soc_name, trace_path, allow_simulation, dispatch, space_name)
    space = SPACES[space_name]
    untouched = profiled_baseline(soc_name, space_name)
    baseline = dict(untouched)
    if start_knobs is not None:
        baseline = dict(start_knobs)
        if not within_budget(baseline, soc_name, BASE_CONFIG):
            raise ValueError("start design does not fit {}'s budget: {}".format(soc_name, json.dumps(baseline)))

    # Conditions live on the workload descriptors: the trace profile (built once
    # per trace, cached in results/profile_*.json) read against this chip's geometry.
    # Always the UNTOUCHED chip's geometry: a descriptor is a property of the
    # workload on this chip, and must not change with the design a search starts from.
    if isinstance(trace_path, list):
        profiles = []
        for member in trace_path:
            profiles.append(trace_profile.load_or_build(member, WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS))
        descriptors = mix_descriptors(profiles, untouched, space, soc_name)
    else:
        profile = trace_profile.load_or_build(trace_path, WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)
        descriptors = chip_descriptors(profile, untouched, space, soc_name)
    condition_metrics = list(DESCRIPTOR_METRICS)

    def is_candidate(knobs):
        """Any design the loop may run: in the space, inside the budget, not crashed here."""
        if not in_space(knobs, space):
            return False
        typed = typed_knobs(knobs, space)
        if not within_budget(typed, soc_name, BASE_CONFIG):
            return False
        name = config_name(typed, soc_name)
        crashed = name in holder.sweep_table and holder.sweep_table[name]["metrics"] is None
        return not crashed

    def name_of(knobs):
        return config_name(typed_knobs(knobs, space), soc_name)

    candidates = {}
    for knobs in [baseline] + random_feasible_designs(soc_name, CANDIDATE_POOL, space=space):
        if is_candidate(knobs):
            candidates[name_of(knobs)] = knobs

    return {
        "name": soc_name + "/" + trace_name,
        "soc_name": soc_name,
        "search_space": space,
        "candidates": candidates,
        "is_candidate": is_candidate,
        "name_of": name_of,
        "baseline": baseline,
        # The chip as shipped: what "bigger than the chip's own cache" means for a
        # rule, whatever design the search starts from.
        "untouched": untouched,
        "chip_text": describe(soc_name),
        "evaluate": holder.evaluate,
        "evaluate_many": holder.evaluate_many,
        "objective": "ipc",
        "table_metrics": ["ipc", "L1D_mpki", "L2C_mpki", "LLC_mpki", "LLC_hit_ratio"],
        # Rule conditions must describe the WORKLOAD, not the chip: they are checked
        # on `descriptors`, never on the chip's own runs.
        "condition_metrics": condition_metrics,
        "descriptors": descriptors,
        # Per-workload descriptors (one workload here): used to FIT condition
        # thresholds from where a claim's effect appears and where it does not.
        "workload_descriptors": {trace_name: descriptors},
        "area_budget_kb": AREA_BUDGET_KB[soc_name],
        "holder": holder,
    }


SUITE_AGGREGATE_METRICS = ["L1D_mpki", "L2C_mpki", "LLC_mpki", "L2C_hit_ratio", "LLC_hit_ratio", "LLC_over_L2C_mpki"]


def make_suite_problem(soc_name, trace_paths, allow_simulation=True, dispatch=None, space_name="C", start_knobs=None):
    """One design is scored on a SUITE of workloads: the objective is the geometric
    mean of per-workload IPC (one simulation per workload per design). This is how
    a design team scores a hierarchy; no chip is built for one program.

    Rule conditions see the suite through aggregates: the mean of each workload
    descriptor across the suite (plain name) and its max ("max_" prefix).
    On a multi-core chip each `trace_paths` entry is itself a list: a mix."""
    if len(trace_paths) == 1:
        return make_problem(soc_name, trace_paths[0], allow_simulation, dispatch, space_name, start_knobs)
    single_problems = []
    for trace_path in trace_paths:
        single_problems.append(make_problem(soc_name, trace_path, allow_simulation, dispatch, space_name, start_knobs))
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
    def is_candidate(knobs):
        for problem in single_problems:
            if not problem["is_candidate"](knobs):
                return False
        return True

    candidates = {}
    for name, knobs in first["candidates"].items():
        if is_candidate(knobs):
            candidates[name] = knobs

    # The table shows the suite objective and, per workload, the objective and the
    # cache diagnostics the agent reasons with (a suite-mean MPKI mixes workloads).
    table_metrics = ["ipc"]
    for short in short_names:
        for metric in ["ipc", "L2C_mpki", "LLC_mpki", "LLC_hit_ratio"]:
            table_metrics.append(short + ":" + metric)
    # Suite descriptors: mean and max of each workload's descriptor.
    base_metrics = list(first["condition_metrics"])
    condition_metrics = list(base_metrics)
    for metric in base_metrics:
        condition_metrics.append("max_" + metric)
    workload_descriptors = {}
    for short, problem in zip(short_names, single_problems):
        workload_descriptors[short] = problem["descriptors"]
    descriptors = {}
    for metric in base_metrics:
        values = []
        for problem in single_problems:
            values.append(problem["descriptors"][metric])
        descriptors[metric] = sum(values) / len(values)
        descriptors["max_" + metric] = max(values)

    return {
        "name": soc_name + "/suite-" + "+".join(short_names),
        "soc_name": soc_name,
        "search_space": first["search_space"],
        "candidates": candidates,
        "is_candidate": is_candidate,
        "name_of": first["name_of"],
        "baseline": first["baseline"],
        "untouched": first["untouched"],
        "chip_text": first["chip_text"],
        "evaluate": evaluate,
        "evaluate_many": evaluate_many,
        "objective": "ipc",
        "table_metrics": table_metrics,
        "condition_metrics": condition_metrics,
        "descriptors": descriptors,
        "workload_descriptors": workload_descriptors,
        "area_budget_kb": first["area_budget_kb"],
        "holder": first["holder"],
        "holders": [problem["holder"] for problem in single_problems],
    }


def candidate_pool(problem, seed):
    """The sample of feasible designs the GP and random arms score, drawn with the
    run's own seed: every seed scores a different sample, as the agents' seeds
    give different searches. Always holds the start design."""
    baseline = problem["baseline"]
    pool = {problem["name_of"](baseline): baseline}
    drawn = random_feasible_designs(problem["soc_name"], CANDIDATE_POOL, seed=seed, space=problem["search_space"])
    for knobs in drawn:
        if problem["is_candidate"](knobs):
            pool[problem["name_of"](knobs)] = knobs
    return pool


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
