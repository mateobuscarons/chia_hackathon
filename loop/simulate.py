"""Node [1] of the loop: run one ChampSim simulation, return clean metrics.

This is the only file that touches the simulator. The analyst agent and
the surrogate model only ever see the flat dict that run_simulation()
returns — so swapping simulators or moving to GCP touches this file only.
"""

import fcntl
import json
import shutil
import subprocess
import time
import tempfile
import os

# The cache levels our loop tunes and reports on.
CACHE_LEVELS = ["L1D", "L2C", "LLC"]


def run_simulation(binary_path, trace_path, warmup_instructions, simulation_instructions):
    """Run one simulation and return a flat metrics dict (ipc, misses, mpki)."""
    # ChampSim writes machine-readable stats to this file via --json.
    stats_path = tempfile.mktemp(suffix=".json", prefix="champsim_stats_")

    command = [
        binary_path,
        "--warmup-instructions", str(warmup_instructions),
        "--simulation-instructions", str(simulation_instructions),
        "--json", stats_path,
        trace_path,
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    with open(stats_path) as stats_file:
        phases = json.load(stats_file)
    os.remove(stats_path)

    # ChampSim reports one entry per phase; we only want the measured one.
    simulation_phase = None
    for phase in phases:
        if phase.get("name") == "Simulation":
            simulation_phase = phase
    stats = simulation_phase["roi"]

    core = stats["cores"][0]
    metrics = {}
    metrics["ipc"] = core["instructions"] / core["cycles"]

    for cache_name in CACHE_LEVELS:
        cache = _find_cache(stats, cache_name)
        hits = 0
        misses = 0
        # Demand traffic only (real requests, not prefetches).
        for access_type in ["LOAD", "RFO", "WRITE"]:
            hits += sum(cache[access_type]["hit"])
            misses += sum(cache[access_type]["miss"])
        metrics[cache_name + "_hits"] = hits
        metrics[cache_name + "_misses"] = misses
        metrics[cache_name + "_mpki"] = misses * 1000.0 / core["instructions"]

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
    legacy_path = os.path.join(champsim_root, "bin", executable_name)
    if os.path.isfile(legacy_path):
        return legacy_path

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


def _find_cache(stats, cache_name):
    """Find a cache's stats; ChampSim names them 'LLC' but also 'cpu0_L1D'."""
    for key in stats:
        if key == cache_name or key.endswith("_" + cache_name):
            return stats[key]
    raise KeyError("cache not found in ChampSim output: " + cache_name)
