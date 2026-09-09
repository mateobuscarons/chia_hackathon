"""The search space and ChampSim config generation.

Twelve knobs over the three cache levels: sizes, associativity, prefetchers,
replacement and miss-handling depth (MSHRs gate memory-level parallelism and
interact with prefetchers). Cache latency is DERIVED FROM SIZE by ChampSim's own
formula, so a bigger cache is slower and capacity is never free. 207,360 designs
before the area budget. PLACEHOLDER values, team to review.
"""

import itertools
import json
import os
import threading

from loop.socs import apply_profile, num_cores, AREA_BUDGET_KB

SEARCH_SPACE = {
    "l1d_sets": [32, 64, 128],
    "l1d_ways": [8, 12],
    "l1d_prefetcher": ["no", "next_line"],
    "l2_sets": [256, 512, 1024, 2048],
    "l2_ways": [4, 8, 16],
    "l2_prefetcher": ["no", "spp_dev", "ip_stride", "next_line", "va_ampm_lite"],
    "llc_sets": [1024, 2048, 4096, 8192],
    "llc_ways": [8, 16],
    "llc_prefetcher": ["no", "next_line"],
    "llc_replacement": ["lru", "srrip", "drrip", "ship"],
    "l2_mshr": [16, 32, 64],
    "llc_mshr": [32, 64, 128],
}

# Named spaces, so a chip profile can point at its own (chip D will).
SPACES = {"C": SEARCH_SPACE}

# Caches whose fixed latency is dropped so ChampSim derives it from size.
LATENCY_CACHES = ["L1D", "L2C", "LLC"]

# Where each knob lives in the ChampSim config JSON: (section, field).
KNOB_LOCATION = {
    "l1d_sets": ("L1D", "sets"), "l1d_ways": ("L1D", "ways"), "l1d_prefetcher": ("L1D", "prefetcher"),
    "l2_sets": ("L2C", "sets"), "l2_ways": ("L2C", "ways"), "l2_prefetcher": ("L2C", "prefetcher"),
    "llc_sets": ("LLC", "sets"), "llc_ways": ("LLC", "ways"), "llc_prefetcher": ("LLC", "prefetcher"),
    "llc_replacement": ("LLC", "replacement"),
    "l2_mshr": ("L2C", "mshr_size"), "llc_mshr": ("LLC", "mshr_size"),
}

_base_config_cache = {}


def all_configurations(space=SEARCH_SPACE):
    """Every knob combination in the space, as a list of dicts."""
    knob_names = list(space.keys())
    value_lists = [space[knob] for knob in knob_names]
    combinations = []
    # itertools.product walks every combination of the value lists in order.
    for values in itertools.product(*value_lists):
        knobs = {}
        for knob, value in zip(knob_names, values):
            knobs[knob] = value
        combinations.append(knobs)
    return combinations


def config_name(knobs, soc_name):
    """Stable, filesystem-safe name for one SoC + knob combination: a sorted
    knob-value listing. The result tables and the binary cache are keyed by it."""
    # The pooled-surrogate arms tag knobs with "soc"; it is not a ChampSim knob
    # and must not change the name, or the result cache is never hit.
    if "soc" in knobs:
        knobs = dict(knobs)
        del knobs["soc"]
    # No "=" in names: make would read the target as a variable assignment.
    parts = []
    for knob in sorted(knobs.keys()):
        parts.append("{}-{}".format(knob, knobs[knob]))
    return soc_name + "_" + "_".join(parts)


def build_config(knobs, soc_name, base_config_path, space_name=None):
    """The full ChampSim config dict for one SoC + knobs (nothing written)."""
    if base_config_path not in _base_config_cache:
        with open(base_config_path) as base_file:
            _base_config_cache[base_config_path] = json.load(base_file)
    config = json.loads(json.dumps(_base_config_cache[base_config_path]))   # deep copy
    config = apply_profile(config, soc_name)
    config["executable_name"] = config_name(knobs, soc_name)
    for knob in knobs:
        if knob == "soc":
            continue            # pooled-surrogate tag, not a ChampSim field
        section, field = KNOB_LOCATION[knob]
        config[section][field] = knobs[knob]
    # Drop the fixed latencies: ChampSim then uses round((sets*ways)^0.343 * 0.416).
    for cache_name in LATENCY_CACHES:
        if "latency" in config[cache_name]:
            del config[cache_name]["latency"]
    return config


def cache_area_kb(knobs, soc_name):
    """Area proxy: data capacity of the tuned caches, one private L2 per core plus
    the shared LLC, in KB. Computed from the knobs alone (hundreds of thousands of
    designs are checked)."""
    block_bytes = 64
    l2_bytes = knobs["l2_sets"] * knobs["l2_ways"] * block_bytes * num_cores(soc_name)
    llc_bytes = knobs["llc_sets"] * knobs["llc_ways"] * block_bytes
    return (l2_bytes + llc_bytes) / 1024.0


def within_budget(knobs, soc_name, base_config_path):
    """Hard constraint: a config is a candidate only if it fits the SoC's area budget."""
    return cache_area_kb(knobs, soc_name) <= AREA_BUDGET_KB[soc_name]


def make_config(knobs, soc_name, base_config_path, output_dir, space_name=None):
    """Write the ChampSim config JSON for one SoC + knobs; return its path."""
    config = build_config(knobs, soc_name, base_config_path, space_name)
    name = config["executable_name"]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name + ".json")
    # Suite problems evaluate one design on several workloads at once, so two
    # threads write this same file together: write a private temp file and
    # rename it, so a reader never sees a half-written (empty) config.
    temp_path = "{}.{}.{}.tmp".format(output_path, os.getpid(), threading.get_ident())
    with open(temp_path, "w") as output_file:
        json.dump(config, output_file, indent=2)
    os.replace(temp_path, output_path)
    return output_path
