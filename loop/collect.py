"""Collect per-workload rows for the physics surrogate: simulate a fixed set of
designs, per chip, on every trace given.

  python -m loop.collect <designs_per_chip> <trace.xz> [more traces]

Designs come from the chip's own Tier C table on the reference workload (mcf):
the baseline first, then every design one knob away from it, then the rest in a
fixed random order. The same designs then run on each new trace, so every chip
gets aligned rows across workloads. Results land in the usual Tier C tables
(results/tierC_<soc>_<trace>.json), which every other command reads as a cache.
Env: PARALLEL_PAIRS (soc x trace pairs at once, default 4), SIM_THREADS per pair,
COLLECT_SOCS (comma-separated chip names; default all three).
"""

import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from loop import champsim_problem
from loop.sweep import load_sweep

ALL_SOCS = ["A_mobile", "B_midrange", "C_server"]
REFERENCE_TRACE = "605.mcf_s-665B"
SPACE = "C"


def designs_for_chip(soc_name, how_many):
    table = load_sweep("results/tierC_{}_{}.json".format(soc_name, REFERENCE_TRACE))
    baseline = champsim_problem.profiled_baseline(soc_name, SPACE)
    one_knob_away = []
    others = []
    for name in table:
        entry = table[name]
        if entry["metrics"] is None:
            continue
        knobs = entry["knobs"]
        if "soc" in knobs:
            continue
        differences = 0
        for knob in baseline:
            if str(knobs.get(knob)) != str(baseline[knob]):
                differences += 1
        if differences == 0:
            continue
        if differences == 1:
            one_knob_away.append(knobs)
        else:
            others.append(knobs)
    random.Random(0).shuffle(others)
    ordered = [baseline] + one_knob_away + others
    return ordered[:how_many]


def collect_pair(soc_name, trace_path, designs):
    problem = champsim_problem.make_problem(soc_name, trace_path, allow_simulation=True, space_name=SPACE)
    holder = problem["holder"]
    missing = 0
    for knobs in designs:
        name = champsim_problem.config_name(knobs, soc_name)
        if name not in holder.sweep_table:
            missing += 1
    started = time.time()
    print("{} {}: {} designs, {} to simulate".format(soc_name, os.path.basename(trace_path),
                                                      len(designs), missing), flush=True)
    holder.evaluate_many(designs)
    print("{} {}: done in {:.0f} min".format(soc_name, os.path.basename(trace_path),
                                             (time.time() - started) / 60), flush=True)


def chosen_socs():
    """Which chips to collect on. A headroom check only needs the target chip."""
    setting = os.environ.get("COLLECT_SOCS", "")
    if setting == "":
        return ALL_SOCS
    names = []
    for name in setting.split(","):
        names.append(name.strip())
    return names


def main():
    how_many = int(sys.argv[1])
    trace_paths = sys.argv[2:]
    socs = chosen_socs()
    designs_by_soc = {}
    for soc_name in socs:
        designs_by_soc[soc_name] = designs_for_chip(soc_name, how_many)
    pairs = []
    for trace_path in trace_paths:
        for soc_name in socs:
            pairs.append((soc_name, trace_path))
    pool = ThreadPoolExecutor(max_workers=int(os.environ.get("PARALLEL_PAIRS", "4")))
    futures = []
    for soc_name, trace_path in pairs:
        futures.append(pool.submit(collect_pair, soc_name, trace_path, designs_by_soc[soc_name]))
    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
