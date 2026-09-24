"""One design, all the way to numbers: ChampSim's config, a binary, a run.

Three steps, and this file is the only place any of them lives:
  build_config  a design (council_loop.space) plus the chip's overrides -> ChampSim's own
                configuration, as the dict its config.sh reads
  build_binary  that configuration -> a compiled binary, built by the CHIA block
                `build_champsim_config` in whichever ChampSim tree is free and
                cached under champsim_bin/
  run_simulation  that binary on a trace, run by CHIA's `ChampSimNode.run_champsim`
                -> a flat metrics dict (ipc, misses, mpki) from the run's raw stats

The chip comes from `council_loop.chip`: the SoC's overrides on ChampSim's stock core and
memory, the clock every latency is counted in, and CACTI's hit latency for each level.
"""

import fcntl
import json
import math
import time
import os

from chia.base.ChiaFunction import get
from chia.simulators.champsim import ChampSimNode
from chia.simulators.champsim_config import build_champsim_config

from council_loop import chip
from council_loop.space import config_name

# The cache levels our loop tunes and reports on.
# The levels whose hit and miss counts are kept. Only the LLC's are read anywhere
# downstream - the objective is IPC, and the results table the agent reads shows LLC
# misses and hit ratio - so counting the other two would fill the result tables with
# columns nothing looks at.
# Every cache ChampSim reports, not only the three the design space touches: a
# simulation is expensive and a counter is free, so a later ablation over the
# instruction side or the TLBs does not have to re-run anything.
ALL_CACHES = ["L1I", "L1D", "L2C", "LLC", "ITLB", "DTLB", "STLB"]
# How each kind of traffic is counted: demand requests are the design's real work
# and carry no prefix; prefetch and translation traffic are kept apart from it.
TRAFFIC = {"LOAD": "", "RFO": "", "WRITE": "", "PREFETCH": "pf_", "TRANSLATION": "translation_"}
PREFETCH_STATS = {"pf_requested": "prefetch requested", "pf_issued": "prefetch issued",
                  "pf_useful": "useful prefetch", "pf_useless": "useless prefetch"}

# Bumped whenever this file changes what it records. A row written under an older
# version is a cache miss and is re-simulated, so a design never serves a report
# with holes in it.
METRICS_VERSION = 3


# ---------------------------------------------------------------- the chip ----

# The chip is `council_loop.chip`: this file renders any chip and knows none.

# Caches whose hit latency this loop sets from CACTI rather than leaving to ChampSim's
# size formula, so a level's latency follows the SoC's node and clock.
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


def build_config(knobs, base_config_path):
    """The full ChampSim config dict for one design (nothing written)."""
    if base_config_path not in _base_config_cache:
        with open(base_config_path) as base_file:
            _base_config_cache[base_config_path] = json.load(base_file)
    config = json.loads(json.dumps(_base_config_cache[base_config_path]))   # deep copy
    config = chip.apply_profile(config)
    config["executable_name"] = config_name(knobs)
    for knob in knobs:
        section, field = KNOB_LOCATION[knob]
        config[section][field] = knobs[knob]
    # CACTI's access time at this SoC's node and clock, in place of ChampSim's own
    # size formula: the same capacity costs a different number of cycles on a chip
    # clocked differently, which is what makes two chips want different geometry.
    for cache_name in LATENCY_CACHES:
        config[cache_name]["latency"] = chip.latency(
            cache_name, config[cache_name]["sets"], config[cache_name]["ways"])
    return config


# ---------------------------------------------------------------- the run ----


class SimulatorInterrupted(RuntimeError):
    """The simulator was killed from outside; nothing is known about the design."""


def run_simulation(binary_path, trace_paths, warmup_instructions, simulation_instructions):
    """Run one simulation and return a flat metrics dict (ipc, misses, mpki).

    trace_paths: one trace (a string, or a list of one). CHIA's run node takes one trace, so a
    multi-core mix (one trace per core) is not run through it."""
    if isinstance(trace_paths, str):
        trace_paths = [trace_paths]
    if len(trace_paths) != 1:
        raise NotImplementedError("CHIA's run node takes one trace; a multi-core mix cannot be run")
    with open(binary_path, "rb") as binary_file:
        binary = binary_file.read()
    # CHIA's node writes the binary to a temporary path, runs it with --json and parses a typed
    # subset of the stats; the raw record it also returns (our patch) is what is parsed below,
    # so every counter the loop reads (merges, every DRAM channel) is the simulator's own.
    result = get(ChampSimNode.run_champsim.chia_remote(
        binary, trace_paths[0], warmup_instructions=warmup_instructions,
        simulation_instructions=simulation_instructions, timeout_s=3600))
    if not result.success:
        # A kill from outside or the clock: the design is not to blame and the table must not
        # remember it as a crash.
        message = "ChampSim exited {} for {} on {}: {}".format(
            result.returncode, os.path.basename(binary_path), trace_paths[0], result.stdout_tail[-300:].strip())
        raise (SimulatorInterrupted if result.returncode < 0 or result.timed_out else RuntimeError)(message)
    phases = result.raw_stats
    if not phases:
        raise RuntimeError("ChampSim returned no stats record for {} on {}: the raw-stats patch is not installed".format(
            os.path.basename(binary_path), trace_paths[0]))

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

    metrics["instructions"] = total_instructions
    metrics["cycles"] = sum(core["cycles"] for core in stats["cores"])
    metrics["mispredict"] = sum(sum(core["mispredict"].values()) for core in stats["cores"])
    metrics["rob_occupancy_at_mispredict"] = sum(
        core["Avg ROB occupancy at mispredict"] for core in stats["cores"]) / len(stats["cores"])

    per_core_instructions = [core["instructions"] for core in stats["cores"]]
    for cache_name in ALL_CACHES:
        try:
            found = _find_caches(stats, cache_name)
        except KeyError:
            continue        # a chip need not have every cache
        counts = dict.fromkeys(
            [prefix + field for prefix in set(TRAFFIC.values()) for field in ["hits", "misses"]]
            + ["merges"] + list(PREFETCH_STATS), 0)
        miss_latency = 0.0
        # Private caches exist once per core (cpu0_L2C, cpu1_L2C, ...); the LLC once.
        for cache in found:
            for access_type, prefix in TRAFFIC.items():
                counts[prefix + "hits"] += sum(cache[access_type]["hit"])
                counts[prefix + "misses"] += sum(cache[access_type]["miss"])
                counts["merges"] += sum(cache[access_type]["miss_merge"])
            for key, field in PREFETCH_STATS.items():
                counts[key] += cache[field]
            # A cache with no misses reports its miss latency as null; treat it as none.
            if cache["miss latency"] is not None:
                miss_latency = max(miss_latency, cache["miss latency"])
        for key in counts:
            metrics["{}_{}".format(cache_name, key)] = counts[key]
        metrics[cache_name + "_mpki"] = counts["misses"] * 1000.0 / total_instructions
        metrics[cache_name + "_miss_latency"] = miss_latency
        if len(found) > 1:
            # A private level on a multi-core chip: the mean hides which core is
            # missing. Keep each instance's own rate, in core order.
            metrics[cache_name + "_mpki_cores"] = [
                sum(sum(cache[access_type]["miss"]) for access_type in TRAFFIC
                    if not TRAFFIC[access_type]) * 1000.0 / instructions
                for cache, instructions in zip(found, per_core_instructions)]

    channels = stats.get("DRAM", [])
    for channel in channels:
        for field in channel:
            key = "dram_" + field.lower().replace(" ", "_")
            # Counts add across channels; a field the simulator already averaged
            # has to be averaged again, not summed.
            share = len(channels) if field.lower().startswith("avg") else 1
            if channel[field] is None:
                continue
            metrics[key] = metrics.get(key, 0.0) + channel[field] / share

    metrics["metrics_version"] = METRICS_VERSION
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


def build_binary(config, champsim_root):
    """Compile ChampSim for one configuration dict; return the path to the binary."""
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
        return _build_binary_locked(config, champsim_root, executable_name, shared_path)
    finally:
        fcntl.flock(binary_lock, fcntl.LOCK_UN)
        binary_lock.close()


def _build_binary_locked(config, champsim_root, executable_name, shared_path):
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
        # The CHIA block builds in the chosen tree (config.sh + make) and returns the binary's
        # bytes; they go to the shared cache under a temporary name, and one rename publishes them.
        built = get(build_champsim_config.chia_remote(os.path.abspath(chosen_tree), config, timeout_s=1800))
        if not built.success:
            raise RuntimeError("build failed for {}: {}".format(
                executable_name, (built.build_diagnostics or built.stdout_tail)[-1500:]))
        temp_path = "{}.{}.tmp".format(shared_path, os.getpid())
        with open(temp_path, "wb") as binary_file:
            binary_file.write(built.binary)
        os.chmod(temp_path, 0o755)
        os.replace(temp_path, shared_path)
        return shared_path
    finally:
        fcntl.flock(chosen_lock, fcntl.LOCK_UN)
        for lock_file in lock_files:
            lock_file.close()


def _find_caches(stats, cache_name):
    """Every instance of one cache level, in core order; ChampSim names them 'LLC' but
    also 'cpu0_L1D'. A private level has one instance per core, a shared level one."""
    found = []
    for key in stats:
        if key == cache_name or key.endswith("_" + cache_name):
            core = int(key[3:key.index("_")]) if key.startswith("cpu") else -1
            found.append((core, stats[key]))
    if len(found) == 0:
        raise KeyError("cache not found in ChampSim output: " + cache_name)
    found.sort()
    return [cache for _, cache in found]
