"""Node [1] of the loop: run one ChampSim simulation, return clean metrics.

This is the only file that touches the simulator. The analyst agent and
the surrogate model only ever see the flat dict that run_simulation()
returns — so swapping simulators or moving to GCP touches this file only.
"""

import json
import subprocess
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


def build_binary(config_path, champsim_root):
    """Compile ChampSim for one config JSON; return the path to the binary."""
    config_path = os.path.abspath(config_path)

    with open(config_path) as config_file:
        config = json.load(config_file)
    executable_name = config["executable_name"]

    binary_path = os.path.join(champsim_root, "bin", executable_name)

    # A build takes ~2 minutes. In this project one executable_name always
    # means one exact config, so an existing binary can be reused as-is.
    if os.path.isfile(binary_path):
        return binary_path

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

    cpu_count = os.cpu_count()
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
