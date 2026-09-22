"""The suite being tuned: what a design measures on it, and what already has been.

`make_suite_problem(traces)` returns the `problem` dict the arms search:
  stock, search_space, is_candidate(knobs), name_of(knobs),
  evaluate(knobs) -> metrics, evaluate_many([knobs]) -> [metrics],
  objective ("ipc": the geometric mean over the suite), workloads, chip_text, chip,
  area_budget_kb, holders (one result table per workload).

A design is scored on a SUITE, not one program: that is how a design team scores a
hierarchy, and no chip is built for one program. Answers come from the cached
result table when they can and from ChampSim (loop.simulate) when they cannot. The
table is the dataset - shared by every process, appended to under a lock, and the
place `measured_designs` reads the best design known from.

Nothing here opens a trace. A workload is a path handed to the simulator and a
name to key its table by; everything the loop learns about it, it measures.
"""

import fcntl
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

from loop import chip                         # the chip this run tunes, as numbers
from loop.simulate import (ALL_CACHES, KNOB_LOCATION, METRICS_VERSION, SimulatorInterrupted, build_binary,
                           build_config, make_config, run_simulation)
from loop.socs import CORES
from loop.space import AREA_BUDGET_KB, POWER_BUDGET_W, SEARCH_SPACE, budget_text, config_name, in_space, silicon_mm2, typed_knobs, watts, within_budget

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


# ---------------------------------------------------------------- naming ----

def short_name(trace_path):
    """The trace's name without directory or compression suffix: "605.mcf_s-665B",
    "bfs.urand-36B", "merced_0000"."""
    name = os.path.basename(trace_path)
    for suffix in [".champsimtrace.xz", ".champsimtrace.gz", ".champsim.gz", ".champsim.xz", ".xz", ".gz"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def workload_of(entry):
    """What one workload of a cell is: a trace on a single-core chip, or ("name", [trace per
    core]) on a multi-core one, where the mix is the workload and its name keys the table."""
    if isinstance(entry, str):
        return short_name(entry), [entry]
    return entry[0], list(entry[1])


def workloads_for(programs):
    """What a design is measured on, given how many cores the chip has - the cell names
    the programs, the chip decides how they are run.

    One core: each program is a workload. Several cores: a workload is one trace per
    core, and there are two of them - the cell's programs cycled onto the cores, and the
    cell's first program on every core, which is the worst case for a shared level. A
    cell lists its heaviest program first."""
    if CORES == 1:
        return list(programs)
    mixes = [("het", [programs[index % len(programs)] for index in range(CORES)])]
    # The homogeneous mix exists to give a worst case for the shared level when the
    # heterogeneous one cannot: with fewer programs than cores some core repeats a
    # program anyway. A cell that fills every core with a different program already is
    # the worst case, and a second mix would only double what every design costs.
    if len(programs) < CORES:
        mixes.append(("hom", [programs[0]] * CORES))
    # The name keys the result table, so it has to say which traces the mix is: two cells
    # on the same chip both have a heterogeneous mix, and a row of one must never be read
    # for the other. A cell of one program makes the two mixes the same traces; the name
    # says so and the duplicate is dropped rather than simulated twice.
    workloads, seen = [], set()
    for kind, traces in mixes:
        if tuple(traces) in seen:
            continue                 # a cell of one program makes both mixes the same traces
        seen.add(tuple(traces))
        workloads.append(("mix_{}_{}".format(kind, digest(traces)), traces))
    return workloads


def digest(traces):
    """A short stable token for a list of traces."""
    value = 0
    for character in "|".join(traces):
        value = (value * 131 + ord(character)) & 0xFFFFFFFF
    return "{:08x}".format(value)[:4]


# ---------------------------------------------------------------- the result tables: what has already been measured ----

TABLE_DIR = os.environ.get("TABLE_DIR", "results/tables")   # a cell measured elsewhere scores against its own machine's tables


def table_path(trace_name, warmup=WARMUP_INSTRUCTIONS, simulation=SIMULATION_INSTRUCTIONS):
    """The shared result table for one workload: the simulation cache every
    process reads and appends to. Non-default simulation lengths get their own."""
    suffix = ""
    if warmup != DEFAULT_WARMUP or simulation != DEFAULT_SIMULATION:
        suffix = "_w{}M_s{}M".format(warmup // 1_000_000, simulation // 1_000_000)
    os.makedirs(TABLE_DIR, exist_ok=True)
    return os.path.join(TABLE_DIR, "{}{}.json".format(trace_name, suffix))


def load_table(path):
    """Many processes read this table while one rewrites it; a reader may catch a
    half-written file. Retry briefly instead of crashing the run."""
    if not os.path.exists(path):
        return {}
    # Another process may be mid-write (or the file may be momentarily empty):
    # back off and try again rather than crash a whole run on a transient read.
    for attempt in range(20):
        try:
            with open(path) as table_file:
                text = table_file.read()
            if text.strip() == "":
                raise json.JSONDecodeError("empty file", text, 0)
            return json.loads(text)
        except json.JSONDecodeError:
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError("could not read the result table after 20 tries: " + path)


def save_table(table, path):
    """Atomic: write next to the target, then rename, so readers never see a partial file."""
    temporary_path = path + ".tmp.{}".format(os.getpid())
    with open(temporary_path, "w") as table_file:
        json.dump(table, table_file, indent=2)
    os.replace(temporary_path, path)


# ---------------------------------------------------------------- measuring one workload ----

def complete(metrics):
    """A row is usable only if the current simulator wrote it. An older row is a
    cache miss, not a hit: it is re-simulated and overwritten, so no design ever
    serves a report with holes in it."""
    return metrics is not None and metrics.get("metrics_version") == METRICS_VERSION


def enrich_metrics(metrics):
    """Per level, what the raw counts imply: the hit ratio, and how well the
    prefetcher did - accuracy (useful prefetches over issued) and coverage (the
    share of the level's misses it removed). Derived here so every reader agrees."""
    if metrics is None:
        return None
    for level in ALL_CACHES:
        hits = metrics.get(level + "_hits")
        misses = metrics.get(level + "_misses")
        if hits is None or misses is None:
            continue
        total = hits + misses
        metrics[level + "_hit_ratio"] = hits / total if total > 0 else 0.0
        issued = metrics.get(level + "_pf_issued")
        useful = metrics.get(level + "_pf_useful")
        if issued is None or useful is None:
            continue
        metrics[level + "_pf_accuracy"] = useful / issued if issued > 0 else 0.0
        metrics[level + "_pf_coverage"] = useful / (useful + misses) if useful + misses > 0 else 0.0
    return metrics


class ChampSimProblem:
    """Answers evaluate() from the result table when it can; otherwise builds and
    runs ChampSim in this process, one thread per design."""

    def __init__(self, entry, allow_simulation, warmup=WARMUP_INSTRUCTIONS, simulation=SIMULATION_INSTRUCTIONS):
        self.trace_name, self.trace_paths = workload_of(entry)
        self.allow_simulation = allow_simulation
        self.warmup = warmup
        self.simulation = simulation
        self.table_path = table_path(self.trace_name, warmup, simulation)
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
                if complete(self.sweep_table[name]["metrics"]):
                    waiters.append(self.ready(enrich_metrics(dict(self.sweep_table[name]["metrics"]))))
                    continue
            if not self.allow_simulation:
                raise KeyError("not in the table and simulation disabled: " + name)
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
        """Build and run one design. A failure is written to the table as a null
        row, so the next run finds it in the cache instead of paying for the same
        failed attempt again - the candidate pool is drawn from the seed, so the
        same bad design comes back otherwise. To retry one deliberately, delete
        its row from the table."""
        try:
            config_path = make_config(knobs, BASE_CONFIG, GENERATED_DIR)
            binary_path = build_binary(config_path, CHAMPSIM_ROOT)
            return run_simulation(binary_path, self.trace_paths, self.warmup, self.simulation)
        except SimulatorInterrupted:
            raise                       # killed from outside: not the design's fault, not cached
        except Exception:
            self.remember(knobs, None)
            raise


# ---------------------------------------------------------------- the suite: the objective the arms search ----

# The one knob ChampSim's configuration does not name: with no replacement policy
# given, a cache is LRU.
UNNAMED_KNOBS = {"l2_replacement": "lru"}


def stock_design():
    """The untouched chip: every knob at the value this SoC's own configuration carries,
    read back from the rendered config, so the search starts from the chip as it is and
    no value is written down twice."""
    profiled = build_config({}, BASE_CONFIG)
    stock = {}
    for knob in SEARCH_SPACE:
        section, field = KNOB_LOCATION[knob]
        stock[knob] = profiled[section].get(field, UNNAMED_KNOBS.get(knob))
        if stock[knob] is None:
            raise RuntimeError("the chip's configuration does not say what {} is".format(knob))
    return stock


def shrink_ladder(stock):
    """The chip's own hierarchy, then the same design with the last level cut a rung at a
    time: sets first, then ways. Capacity is what a last level is for, and halving it
    costs less of what the level does than halving its associativity."""
    ladder = [dict(stock)]
    current = dict(stock)
    for knob in ["llc_sets", "llc_ways"]:
        for value in sorted(SEARCH_SPACE[knob], reverse=True):
            if value >= int(current[knob]):
                continue
            current = dict(current)
            current[knob] = value
            ladder.append(current)
    return ladder


def shrunk_to_fit(stock, drawn=None):
    """The chip's hierarchy when it still fits the caps; otherwise the first rung down
    the ladder that does - the obvious engineering answer, and the one this project
    measures everything against. A cap can move after a chip is under design, and the
    design that was legal under the old one may not be: the reference is then the shrunk
    form of it, never an illegal design.

    Silicon and leakage are decided by the shape, so they are checked here. Power is
    SPENT, not provisioned, and shrinking on leakage alone under-shrinks: a shape that
    leaks little enough can still draw too much once it runs. `drawn` reads what a rung
    measured where the tables already hold it, and None where they do not - an unmeasured
    rung is taken on its shape alone and violates like any other design if it turns out
    to draw too much."""
    for design in shrink_ladder(stock):
        if not within_budget(typed_knobs(design)):
            continue
        if drawn is not None and POWER_BUDGET_W is not None:
            spent = drawn(design)
            if spent is not None and spent > POWER_BUDGET_W:
                continue
        return design
    raise RuntimeError("no shrink of the last level fits " + budget_text(typed_knobs(stock)))


def make_suite_problem(programs, allow_simulation=True, warmup=WARMUP_INSTRUCTIONS, simulation=SIMULATION_INSTRUCTIONS):
    """One design is scored on a SUITE of workloads: the objective is the geometric
    mean of the per-workload IPC (one simulation per workload per design). This is
    how a design team scores a hierarchy; no chip is built for one program.
    `warmup` and `simulation` pick the fidelity, each with its own tables: the
    council's probes run the same problem at a cheaper rung."""
    holders = []
    names = []
    for entry in workloads_for(programs):
        holder = ChampSimProblem(entry, allow_simulation, warmup, simulation)
        holders.append(holder)
        names.append(holder.trace_name)
    def drawn(knobs):
        """What a design measured in watts over this suite, or None where the tables do
        not hold every workload of it yet."""
        name = config_name(typed_knobs(knobs))
        metrics = {}
        for holder in holders:
            row = holder.sweep_table.get(name)
            if row is None or row["metrics"] is None:
                return None
            for key in row["metrics"]:
                metrics["{}:{}".format(holder.trace_name, key)] = row["metrics"][key]
        return watts(typed_knobs(knobs), metrics, names)

    stock = shrunk_to_fit(stock_design(), drawn)

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
            results.append(aggregate_suite(per_workload, names, knobs_list[index]))
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
        "chip_text": chip.chip_text(),
        "chip": chip.card(),
        "area_budget_kb": AREA_BUDGET_KB,
        "holders": holders,
        "fidelity": (warmup, simulation),
    }


def aggregate_suite(per_workload_metrics, names, knobs):
    """Geometric-mean IPC over the suite, the design's silicon and its power; per-workload
    values kept under "<workload>:<metric>"."""
    combined = {}
    log_sum = 0.0
    for name, metrics in zip(names, per_workload_metrics):
        log_sum += math.log(max(metrics["ipc"], 1e-9))
        for metric in metrics:
            combined[name + ":" + metric] = metrics[metric]
    combined["ipc"] = math.exp(log_sum / len(per_workload_metrics))
    combined["mm2"] = silicon_mm2(knobs)
    combined["watts"] = watts(knobs, combined, names)
    return combined


def measured_designs(problem):
    """Every design measured on EVERY workload of the suite, with suite metrics
    (the best of them is the cell's best known design)."""
    first_table = problem["holders"][0].sweep_table
    designs = []
    for name in first_table:
        if not name.startswith(chip.NAME + "_"):
            continue
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
                        "metrics": aggregate_suite(per_workload, problem["workloads"], first_table[name]["knobs"])})
    return designs
