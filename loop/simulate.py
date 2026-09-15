"""One design, all the way to numbers: ChampSim's config, a binary, a run.

Three steps, and this file is the only place any of them lives:
  make_config   a design (loop.space) plus the chip's overrides -> ChampSim's own
                configuration JSON
  build_binary  that JSON -> a compiled binary, cached under champsim_bin/ and
                built in whichever ChampSim tree is free
  run_simulation  that binary on a trace -> a flat metrics dict (ipc, misses, mpki)

The chip itself is here too, as PROFILE: the handful of fields this project
overrides on ChampSim's stock core and memory. Fields that are search knobs are
not chip parameters and are not in it.
"""

import fcntl
import json
import math
import shutil
import subprocess
import threading
import time
import tempfile
import os

from loop.space import AREA_BUDGET_KB, CHIP, config_name

# The cache levels our loop tunes and reports on.
# The levels whose hit and miss counts are kept. Only the LLC's are read anywhere
# downstream - the objective is IPC, and the results table the agent reads shows LLC
# misses and hit ratio - so counting the other two would fill the result tables with
# columns nothing looks at.
CACHE_LEVELS = ["LLC"]


# ---------------------------------------------------------------- the chip ----

PROFILE = {
    "ooo_cpu": {"rob_size": 512, "lq_size": 192, "sq_size": 114, "scheduler_size": 192},
    "L1D": {"mshr_size": 32},
    "L2C": {"mshr_size": 64},
    "LLC": {"ways": 16, "mshr_size": 128},
    "physical_memory": {"data_rate": 3200, "channels": 2},
}

# Caches whose fixed latency is dropped so ChampSim derives it from size: a bigger
# cache is then slower, and capacity is never free.
LATENCY_CACHES = ["L1D", "L2C", "LLC"]

# Where each knob lives in the ChampSim config JSON: (section, field).
KNOB_LOCATION = {
    "l1d_sets": ("L1D", "sets"), "l1d_ways": ("L1D", "ways"), "l1d_prefetcher": ("L1D", "prefetcher"),
    "l2_sets": ("L2C", "sets"), "l2_ways": ("L2C", "ways"), "l2_prefetcher": ("L2C", "prefetcher"),
    "l2_replacement": ("L2C", "replacement"),
    "llc_sets": ("LLC", "sets"), "llc_ways": ("LLC", "ways"), "llc_prefetcher": ("LLC", "prefetcher"),
    "llc_replacement": ("LLC", "replacement"),
    "l2_mshr": ("L2C", "mshr_size"), "llc_mshr": ("LLC", "mshr_size"),
}

_base_config_cache = {}


def describe():
    """The chip in one line for the agent's prompt: its overrides over the stock
    ChampSim core and memory, and its area budget."""
    knob_fields = ["sets", "ways", "prefetcher", "replacement", "mshr_size"]
    parts = []
    for section_name in sorted(PROFILE):
        fields = []
        for field_name in sorted(PROFILE[section_name]):
            if section_name in ["L1D", "L2C", "LLC"] and field_name in knob_fields:
                continue
            fields.append("{} {}".format(field_name, PROFILE[section_name][field_name]))
        if len(fields) > 0:
            parts.append("{}: {}".format(section_name, ", ".join(fields)))
    return "Chip {} (one core; area budget {} KB for L2 + LLC): {}".format(CHIP, AREA_BUDGET_KB, "; ".join(parts))


def apply_profile(config):
    """Overwrite the stock config's fields with the chip's overrides."""
    for section_name in PROFILE:
        if section_name == "ooo_cpu":
            # ChampSim stores cores as a list; the first entry is the template.
            target = config["ooo_cpu"][0]
        else:
            target = config[section_name]
        for field_name in PROFILE[section_name]:
            target[field_name] = PROFILE[section_name][field_name]
    return config


def build_config(knobs, base_config_path):
    """The full ChampSim config dict for one design (nothing written)."""
    if base_config_path not in _base_config_cache:
        with open(base_config_path) as base_file:
            _base_config_cache[base_config_path] = json.load(base_file)
    config = json.loads(json.dumps(_base_config_cache[base_config_path]))   # deep copy
    config = apply_profile(config)
    config["executable_name"] = config_name(knobs)
    for knob in knobs:
        section, field = KNOB_LOCATION[knob]
        config[section][field] = knobs[knob]
    # Drop the fixed latencies: ChampSim then uses round((sets*ways)^0.343 * 0.416).
    for cache_name in LATENCY_CACHES:
        if "latency" in config[cache_name]:
            del config[cache_name]["latency"]
    return config


def make_config(knobs, base_config_path, output_dir):
    """Write the ChampSim config JSON for one design; return its path."""
    config = build_config(knobs, base_config_path)
    name = config["executable_name"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name + ".json")
    # A suite evaluates one design on several workloads at once, so two threads
    # may write this file together: write a private temp file and rename it.
    temp_path = "{}.{}.{}.tmp".format(output_path, os.getpid(), threading.get_ident())
    with open(temp_path, "w") as output_file:
        json.dump(config, output_file, indent=2)
    os.replace(temp_path, output_path)
    return output_path


# ---------------------------------------------------------------- the run ----


def run_simulation(binary_path, trace_paths, warmup_instructions, simulation_instructions):
    """Run one simulation and return a flat metrics dict (ipc, misses, mpki).

    trace_paths: one trace (a string) for a single-core chip, or a list with one
    trace per core for a multi-core chip (ChampSim requires exactly one per core)."""
    if isinstance(trace_paths, str):
        trace_paths = [trace_paths]
    # ChampSim writes machine-readable stats to this file via --json.
    stats_path = tempfile.mktemp(suffix=".json", prefix="champsim_stats_")

    command = [
        binary_path,
        "--warmup-instructions", str(warmup_instructions),
        "--simulation-instructions", str(simulation_instructions),
        "--json", stats_path,
    ] + list(trace_paths)
    subprocess.run(command, check=True, capture_output=True, text=True)

    # A design that makes ChampSim exit cleanly but write no usable stats used to
    # surface as a bare JSONDecodeError with nothing identifying it, which killed a
    # whole run. Name the binary and the trace so the next one can be reproduced.
    try:
        with open(stats_path) as stats_file:
            phases = json.load(stats_file)
    except (json.JSONDecodeError, ValueError) as error:
        size = os.path.getsize(stats_path) if os.path.isfile(stats_path) else -1
        raise RuntimeError("ChampSim wrote no usable stats ({} bytes) for {} on {}: {}".format(
            size, os.path.basename(binary_path), trace_paths[0], error))
    finally:
        if os.path.isfile(stats_path):
            os.remove(stats_path)

    # ChampSim reports one entry per phase; we only want the measured one.
    simulation_phase = None
    for phase in phases:
        if phase.get("name") == "Simulation":
            simulation_phase = phase
    stats = simulation_phase["roi"]

    # One core: its IPC. Several cores: the geometric mean of the per-core IPCs
    # (each core runs one program of the mix), with every core's own IPC kept.
    metrics = {}
    total_instructions = 0
    log_ipc_sum = 0.0
    for index, core in enumerate(stats["cores"]):
        core_ipc = core["instructions"] / core["cycles"]
        total_instructions += core["instructions"]
        log_ipc_sum += math.log(core_ipc)
        if len(stats["cores"]) > 1:
            metrics["core{}:ipc".format(index)] = core_ipc
    metrics["ipc"] = math.exp(log_ipc_sum / len(stats["cores"]))

    for cache_name in CACHE_LEVELS:
        hits = 0
        misses = 0
        # Private caches exist once per core (cpu0_L2C, cpu1_L2C, ...); the LLC once.
        for cache in _find_caches(stats, cache_name):
            # Demand traffic only (real requests, not prefetches).
            for access_type in ["LOAD", "RFO", "WRITE"]:
                hits += sum(cache[access_type]["hit"])
                misses += sum(cache[access_type]["miss"])
        metrics[cache_name + "_hits"] = hits
        metrics[cache_name + "_misses"] = misses
        metrics[cache_name + "_mpki"] = misses * 1000.0 / total_instructions

    return metrics


def tree_paths(champsim_root):
    """The ChampSim trees available for building: <root>, <root>_1, ... as long as
    they exist (copies made at setup; env CHAMPSIM_TREES caps how many are used).
    Builds hold a whole tree, so N trees = N builds at once."""
    trees = [champsim_root]
    limit = int(os.environ.get("CHAMPSIM_TREES", "64"))
    index = 1
    while index < limit and os.path.isdir("{}_{}".format(champsim_root, index)):
        trees.append("{}_{}".format(champsim_root, index))
        index += 1
    return trees


def shared_binary_path(champsim_root, executable_name):
    """Binaries from every tree are collected here, so a design built once is
    never built again whichever tree is free."""
    shared_dir = os.path.join(os.path.dirname(os.path.abspath(champsim_root)), "champsim_bin")
    os.makedirs(shared_dir, exist_ok=True)
    return os.path.join(shared_dir, executable_name)


def build_binary(config_path, champsim_root):
    """Compile ChampSim for one config JSON; return the path to the binary."""
    config_path = os.path.abspath(config_path)

    with open(config_path) as config_file:
        config = json.load(config_file)
    executable_name = config["executable_name"]

    # A build takes ~2 minutes. In this project one executable_name always
    # means one exact config, so an existing binary can be reused as-is.
    shared_path = shared_binary_path(champsim_root, executable_name)
    if os.path.isfile(shared_path):
        return shared_path
    # A suite evaluates one design on several workloads at once, so the same
    # binary is requested by several threads together. One lock per binary makes
    # them queue: the first builds, the rest find the finished file below.
    binary_lock = open(shared_path + ".lock", "w")
    fcntl.flock(binary_lock, fcntl.LOCK_EX)
    try:
        return _build_binary_locked(config_path, champsim_root, executable_name, shared_path)
    finally:
        fcntl.flock(binary_lock, fcntl.LOCK_UN)
        binary_lock.close()


def _build_binary_locked(config_path, champsim_root, executable_name, shared_path):
    """Build in a free tree and publish the binary; the caller holds the per-binary lock."""
    if os.path.isfile(shared_path):     # built by whoever held the lock before us
        return shared_path

    # Parallel processes must never run config.sh/make in the same tree at once:
    # take the first free tree (non-blocking), else wait for one.
    trees = tree_paths(champsim_root)
    lock_files = []
    for tree in trees:
        lock_files.append(open(os.path.join(tree, ".build.lock"), "w"))
    chosen_tree = None
    chosen_lock = None
    while chosen_tree is None:
        for tree, lock_file in zip(trees, lock_files):
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                chosen_tree = tree
                chosen_lock = lock_file
                break
            except BlockingIOError:
                continue
        if chosen_tree is None:
            time.sleep(2)
    try:
        if os.path.isfile(shared_path):     # another process built it while we waited
            return shared_path
        tree_binary = os.path.join(chosen_tree, "bin", executable_name)
        _configure_and_make(config_path, chosen_tree, executable_name, tree_binary)
        temp_path = "{}.{}.tmp".format(shared_path, os.getpid())
        shutil.copy2(tree_binary, temp_path)
        os.replace(temp_path, shared_path)
        return shared_path
    finally:
        fcntl.flock(chosen_lock, fcntl.LOCK_UN)
        for lock_file in lock_files:
            lock_file.close()


def _configure_and_make(config_path, champsim_root, executable_name, binary_path):
    subprocess.run(
        ["./config.sh", config_path],
        cwd=champsim_root, check=True, capture_output=True, text=True,
    )
    # generated_environment.o is shared between configs but its contents
    # depend on the config hash; make sometimes keeps a stale copy and the
    # link fails. Deleting it forces a rebuild (adds ~20 s per build).
    stale_object = os.path.join(champsim_root, ".csconfig", "generated_environment.o")
    if os.path.isfile(stale_object):
        os.remove(stale_object)

    cpu_count = max(2, os.cpu_count() // int(os.environ.get("CHAMPSIM_BUILD_SHARE", "1")))
    for attempt in range(2):
        make_result = subprocess.run(
            ["make", "-j" + str(cpu_count)],
            cwd=champsim_root, capture_output=True, text=True,
        )
        if make_result.returncode == 0:
            return binary_path
        print("make attempt", attempt + 1, "failed:", make_result.stderr[-1500:], flush=True)
    raise RuntimeError("make failed for " + executable_name)


def _find_caches(stats, cache_name):
    """Every instance of one cache level; ChampSim names them 'LLC' but also 'cpu0_L1D'."""
    found = []
    for key in stats:
        if key == cache_name or key.endswith("_" + cache_name):
            found.append(stats[key])
    if len(found) == 0:
        raise KeyError("cache not found in ChampSim output: " + cache_name)
    return found
