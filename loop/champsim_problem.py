"""ChampSim glue: turns a list of traces into the `problem` dict the arms search.

This is the only place where the loop learns it is tuning caches. A problem is:
  name, search_space, stock (the untouched chip's knobs), is_candidate(knobs),
  name_of(knobs), evaluate(knobs) -> metrics, evaluate_many([knobs]) -> [metrics],
  objective ("ipc": the geometric mean over the suite), table_metrics (what the
  agent's table shows), workloads (short names), descriptors per workload,
  chip_text, area_budget_kb, holders (one result table per workload).
"""

import fcntl
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

from loop import trace_profile
from loop.configs import (AREA_BUDGET_KB, KNOB_LOCATION, SEARCH_SPACE, build_config, config_name, describe, in_space,
                          make_config, random_feasible_designs, typed_knobs, within_budget)
from loop.simulate import build_binary, run_simulation

CHAMPSIM_ROOT = "champsim"
BASE_CONFIG = "champsim/champsim_config.json"
GENERATED_DIR = "configs/generated"
# The search runs short simulations (validated: Spearman 0.919 against 50M/50M
# over 26 designs). LOOP_WARMUP / LOOP_SIM run the same designs longer into their
# own tables, so long and short rows never mix.
DEFAULT_WARMUP = 5_000_000
DEFAULT_SIMULATION = 10_000_000
WARMUP_INSTRUCTIONS = int(os.environ.get("LOOP_WARMUP", DEFAULT_WARMUP))
SIMULATION_INSTRUCTIONS = int(os.environ.get("LOOP_SIM", DEFAULT_SIMULATION))

# The untouched chip: what ChampSim's stock config says for the knobs it names
# (no L2 replacement policy is named there, which means LRU).
STOCK_KNOBS = {"l2_sets": 1024, "llc_sets": 2048, "l2_prefetcher": "no", "llc_replacement": "lru", "l2_replacement": "lru"}

# The GP arm scores a fixed random sample of the feasible designs (the space has
# millions); the agent may name ANY feasible design (is_candidate).
CANDIDATE_POOL = 20000

# "local" or "chia" (LOOP_DISPATCH=chia, set in the environment): with "chia" every
# build and simulation is a CHIA task.
DEFAULT_DISPATCH = os.environ.get("LOOP_DISPATCH", "local")

# Workload descriptors: the PROGRAM, profiled once from its trace with no
# simulator (loop.trace_profile), and that program against THIS chip's geometry:
# working set over cache size, predicted LRU miss ratio at the chip's caches, and
# the misses a bigger cache could still remove within the area budget. The
# chip's measured speed is never a descriptor.
DESCRIPTOR_METRICS = ["mem_accesses_per_kinstr", "write_fraction", "footprint_kb",
                      "stride_regular_fraction", "reuse_local_fraction",
                      "l2_footprint_ratio", "llc_footprint_ratio",
                      "pred_l1d_miss_ratio", "pred_l2_miss_ratio", "pred_llc_miss_ratio",
                      "movable_l2_mpki", "movable_llc_mpki"]

DESCRIPTOR_LEGEND = """Two kinds of descriptor. Program-only, profiled from the trace with no simulator:
memory accesses per 1000 instructions, write fraction, working set (footprint_kb),
fraction of stride-regular accesses (what a stride prefetcher catches), fraction of
reuses within 1024 accesses (temporal locality). Program against THIS chip's cache
sizes: working set over L2 size and over L2+LLC size, the predicted LRU miss ratio
at this chip's L1D, L2 and LLC capacity (footprint theory, from the trace), and
movable_l2_mpki / movable_llc_mpki: the misses per 1000 instructions that a bigger
L2 / LLC would remove, from this chip's size up to the largest its area budget
affords. A program with a huge footprint but movable_llc_mpki near zero reads its
data once and no cache size helps it; a large movable MPKI means capacity pays."""


def largest_affordable_kb():
    """The biggest L2 and the biggest L2 + LLC capacity, in KB, any design in the
    space can have within the area budget."""
    largest_l2_kb = 0.0
    largest_total_kb = 0.0
    for l2_sets in SEARCH_SPACE["l2_sets"]:
        for l2_ways in SEARCH_SPACE["l2_ways"]:
            for llc_sets in SEARCH_SPACE["llc_sets"]:
                for llc_ways in SEARCH_SPACE["llc_ways"]:
                    knobs = {"l2_sets": l2_sets, "l2_ways": l2_ways, "llc_sets": llc_sets, "llc_ways": llc_ways}
                    if not within_budget(knobs):
                        continue
                    l2_kb = l2_sets * l2_ways * 64 / 1024.0
                    total_kb = l2_kb + llc_sets * llc_ways * 64 / 1024.0
                    if l2_kb > largest_l2_kb:
                        largest_l2_kb = l2_kb
                    if total_kb > largest_total_kb:
                        largest_total_kb = total_kb
    return largest_l2_kb, largest_total_kb


def chip_descriptors(profile, stock):
    """The workload profile read against the untouched chip's cache geometry
    (sets x ways x 64 B per level) and the area budget. No simulation involved."""
    l1d_kb = stock["l1d_sets"] * stock["l1d_ways"] * 64 / 1024.0
    l2_kb = stock["l2_sets"] * stock["l2_ways"] * 64 / 1024.0
    llc_kb = stock["llc_sets"] * stock["llc_ways"] * 64 / 1024.0
    largest_l2_kb, largest_total_kb = largest_affordable_kb()
    descriptors = {}
    for metric in trace_profile.PROFILE_METRICS:
        descriptors[metric] = profile[metric]
    descriptors["l2_footprint_ratio"] = profile["footprint_kb"] / l2_kb
    descriptors["llc_footprint_ratio"] = profile["footprint_kb"] / (l2_kb + llc_kb)
    descriptors["pred_l1d_miss_ratio"] = trace_profile.predicted_miss_ratio(profile, l1d_kb)
    descriptors["pred_l2_miss_ratio"] = trace_profile.predicted_miss_ratio(profile, l2_kb)
    descriptors["pred_llc_miss_ratio"] = trace_profile.predicted_miss_ratio(profile, l2_kb + llc_kb)
    accesses = profile["mem_accesses_per_kinstr"]
    l2_drop = descriptors["pred_l2_miss_ratio"] - trace_profile.predicted_miss_ratio(profile, largest_l2_kb)
    llc_drop = descriptors["pred_llc_miss_ratio"] - trace_profile.predicted_miss_ratio(profile, largest_total_kb)
    descriptors["movable_l2_mpki"] = accesses * l2_drop
    descriptors["movable_llc_mpki"] = accesses * llc_drop
    return descriptors


def enrich_metrics(metrics):
    """Hit ratios per level, derived from the simulator's hit and miss counts."""
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
    return metrics


# ---------------------------------------------------------------- names and tables ----

def trace_short_name(trace_path):
    """The trace's name without directory or compression suffix (trace_profile.short_name)."""
    return trace_profile.short_name(trace_path)


def table_path(trace_name):
    """The shared result table for one workload: the simulation cache every
    process reads and appends to. Non-default simulation lengths get their own."""
    suffix = ""
    if WARMUP_INSTRUCTIONS != DEFAULT_WARMUP or SIMULATION_INSTRUCTIONS != DEFAULT_SIMULATION:
        suffix = "_w{}M_s{}M".format(WARMUP_INSTRUCTIONS // 1_000_000, SIMULATION_INSTRUCTIONS // 1_000_000)
    return "results/table_{}{}.json".format(trace_name, suffix)


def load_table(path):
    """Many processes read this table while one rewrites it; a reader may catch a
    half-written file. Retry briefly instead of crashing the run."""
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


def measured_rows(table):
    """Every complete row of a table as {"name", "knobs", "ipc", "metrics"}."""
    rows = []
    for name in table:
        entry = table[name]
        if entry["metrics"] is None or "ipc" not in entry["metrics"]:
            continue
        rows.append({"name": name, "knobs": entry["knobs"], "ipc": entry["metrics"]["ipc"], "metrics": entry["metrics"]})
    return rows


# ---------------------------------------------------------------- one workload's simulator ----

class ChampSimProblem:
    """Answers evaluate() from the result table when it can; otherwise simulates.

    dispatch="local": build + run in this process, one thread per design.
    dispatch="chia":  build_from_config and simulate are dispatched as CHIA
                      tasks (ray must be initialised)."""

    def __init__(self, trace_path, allow_simulation, dispatch):
        self.trace_path = trace_path
        self.trace_name = trace_short_name(trace_path)
        self.allow_simulation = allow_simulation
        self.dispatch = dispatch
        self.table_path = table_path(self.trace_name)
        self.sweep_table = load_table(self.table_path)
        self.local_pool = ThreadPoolExecutor(max_workers=int(os.environ.get("SIM_THREADS", "4")))

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
            name = config_name(knobs)
            if name in self.sweep_table:
                if self.sweep_table[name]["metrics"] is None:
                    raise RuntimeError("design crashed earlier on this workload: " + name)
                waiters.append(self.ready(enrich_metrics(dict(self.sweep_table[name]["metrics"]))))
                continue
            if not self.allow_simulation:
                raise KeyError("not in the table and simulation disabled: " + name)
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
        name = config_name(knobs)
        self.sweep_table[name] = {"knobs": knobs, "metrics": metrics}
        with open(self.table_path + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            table = load_table(self.table_path)
            table[name] = {"knobs": knobs, "metrics": metrics}
            save_table(table, self.table_path)
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    def simulate_here(self, knobs):
        config_path = make_config(knobs, BASE_CONFIG, GENERATED_DIR)
        binary_path = build_binary(config_path, CHAMPSIM_ROOT)
        return run_simulation(binary_path, self.trace_path, WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)

    def dispatch_chia(self, knobs):
        """Two chained CHIA tasks: build -> simulate. Returns the simulate future."""
        from loop.chia_nodes import build_from_config, simulate
        config = build_config(knobs, BASE_CONFIG)
        binary_future = build_from_config.chia_remote(config, os.path.abspath(CHAMPSIM_ROOT))
        return simulate.chia_remote(binary_future, [os.path.abspath(self.trace_path)],
                                    WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS, _chia_tag=config["executable_name"])


# ---------------------------------------------------------------- the problem ----

def stock_design():
    """The untouched chip: STOCK_KNOBS plus every other knob at the value the
    chip's own config carries, read back from the built config."""
    profiled = build_config(STOCK_KNOBS, BASE_CONFIG)
    stock = dict(STOCK_KNOBS)
    for knob in SEARCH_SPACE:
        if knob not in stock:
            section, field = KNOB_LOCATION[knob]
            stock[knob] = profiled[section][field]
    return stock


def workload_descriptors(trace_path):
    """One workload's descriptors, from its cached profile against the stock chip."""
    profile = trace_profile.load_or_build(trace_path, WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS)
    return chip_descriptors(profile, stock_design())


def make_suite_problem(trace_paths, allow_simulation=True, dispatch=None):
    """One design is scored on a SUITE of workloads: the objective is the geometric
    mean of the per-workload IPC (one simulation per workload per design). This is
    how a design team scores a hierarchy; no chip is built for one program."""
    if dispatch is None:
        dispatch = DEFAULT_DISPATCH
    holders = []
    names = []
    descriptors = {}
    for trace_path in trace_paths:
        holder = ChampSimProblem(trace_path, allow_simulation, dispatch)
        holders.append(holder)
        names.append(holder.trace_name)
        descriptors[holder.trace_name] = workload_descriptors(trace_path)
    stock = stock_design()

    def is_candidate(knobs):
        """Any design the loop may run: in the space, inside the budget, not crashed anywhere."""
        if not in_space(knobs):
            return False
        typed = typed_knobs(knobs)
        if not within_budget(typed):
            return False
        name = config_name(typed)
        for holder in holders:
            if name in holder.sweep_table and holder.sweep_table[name]["metrics"] is None:
                return False
        return True

    def name_of(knobs):
        return config_name(typed_knobs(knobs))

    def evaluate_many(knobs_list):
        # Fan out: every (design, workload) pair runs at once through its own holder.
        per_workload_waiters = []
        for holder in holders:
            per_workload_waiters.append(holder.evaluate_many_async(knobs_list))
        results = []
        for index in range(len(knobs_list)):
            per_workload = []
            for holder_index in range(len(holders)):
                per_workload.append(per_workload_waiters[holder_index][index]())
            results.append(aggregate_suite(per_workload, names))
        return results

    def evaluate(knobs):
        return evaluate_many([knobs])[0]

    # The table shows the suite objective and, per workload, IPC and the LLC diagnostics.
    table_metrics = ["ipc"]
    for name in names:
        for metric in ["ipc", "LLC_mpki", "LLC_hit_ratio"]:
            table_metrics.append(name + ":" + metric)

    return {
        "name": "suite-" + "+".join(names),
        "search_space": SEARCH_SPACE,
        "stock": stock,
        "is_candidate": is_candidate,
        "name_of": name_of,
        "evaluate": evaluate,
        "evaluate_many": evaluate_many,
        "objective": "ipc",
        "table_metrics": table_metrics,
        "workloads": names,
        "descriptors": descriptors,
        "chip_text": describe(),
        "area_budget_kb": AREA_BUDGET_KB,
        "holders": holders,
    }


def candidate_pool(problem, seed):
    """The sample of feasible designs the GP arm scores, drawn with the run's own
    seed, so every seed scores a different sample. Always holds the stock design."""
    pool = {problem["name_of"](problem["stock"]): problem["stock"]}
    for knobs in random_feasible_designs(CANDIDATE_POOL, seed=seed):
        if problem["is_candidate"](knobs):
            pool[problem["name_of"](knobs)] = knobs
    return pool


def aggregate_suite(per_workload_metrics, names):
    """Geometric-mean IPC over the suite; per-workload values kept under "<workload>:<metric>"."""
    combined = {}
    log_sum = 0.0
    for name, metrics in zip(names, per_workload_metrics):
        log_sum += math.log(max(metrics["ipc"], 1e-9))
        for metric in metrics:
            combined[name + ":" + metric] = metrics[metric]
    combined["ipc"] = math.exp(log_sum / len(per_workload_metrics))
    return combined


def measured_designs(problem):
    """Every design measured on EVERY workload of the suite, with suite metrics
    (the best of them is the cell's best known design)."""
    first_table = problem["holders"][0].sweep_table
    designs = []
    for name in first_table:
        per_workload = []
        complete = True
        for holder in problem["holders"]:
            entry = holder.sweep_table.get(name)
            if entry is None or entry["metrics"] is None:
                complete = False
                break
            per_workload.append(enrich_metrics(dict(entry["metrics"])))
        if not complete:
            continue
        designs.append({"name": name, "knobs": first_table[name]["knobs"],
                        "metrics": aggregate_suite(per_workload, problem["workloads"])})
    return designs
