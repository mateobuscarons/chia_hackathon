"""Tier-A search space and ChampSim config generation.

PLACEHOLDER knob values — the team must review these before the real runs.
Tier A is deliberately small (81 configs) so a dense reference sweep is
affordable and "% of true optimum" claims stay rigorous.
"""

import itertools
import json
import os

from loop.socs import apply_profile, AREA_BUDGET_KB

# Tier A: small (81 configs) so a dense reference sweep is affordable.
SEARCH_SPACE = {
    "l2_sets": [512, 1024, 2048],                    # 256KB / 512KB / 1MB at 8 ways
    "llc_sets": [1024, 2048, 4096],                  # 1MB / 2MB / 4MB at 16 ways
    "l2_prefetcher": ["no", "spp_dev", "ip_stride"],
    "llc_replacement": ["lru", "srrip", "drrip"],
}

# Tier B: wide (23,040 configs) - no dense sweep possible; arms are compared on
# a fixed simulation budget. PLACEHOLDER values, team-reviewed Sep 3 2026.
SEARCH_SPACE_B = {
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
}

SPACES = {"A": SEARCH_SPACE, "B": SEARCH_SPACE_B}

# Where each knob lives in the ChampSim config JSON: (section, field).
KNOB_LOCATION = {
    "l1d_sets": ("L1D", "sets"), "l1d_ways": ("L1D", "ways"), "l1d_prefetcher": ("L1D", "prefetcher"),
    "l2_sets": ("L2C", "sets"), "l2_ways": ("L2C", "ways"), "l2_prefetcher": ("L2C", "prefetcher"),
    "llc_sets": ("LLC", "sets"), "llc_ways": ("LLC", "ways"), "llc_prefetcher": ("LLC", "prefetcher"),
    "llc_replacement": ("LLC", "replacement"),
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
    """Stable, filesystem-safe name for one SoC + knob combination.

    Tier-A knobs keep the original short format (the sweep tables use it);
    any other knob set gets a generic knob=value listing.
    """
    tier_a_knobs = set(SEARCH_SPACE.keys())
    if set(knobs.keys()) == tier_a_knobs:
        return "{}_l2s{}_pf-{}_llcs{}_rp-{}".format(
            soc_name, knobs["l2_sets"], knobs["l2_prefetcher"],
            knobs["llc_sets"], knobs["llc_replacement"])
    # No "=" in names: make would read the target as a variable assignment.
    parts = []
    for knob in sorted(knobs.keys()):
        parts.append("{}-{}".format(knob, knobs[knob]))
    return soc_name + "_" + "_".join(parts)


def build_config(knobs, soc_name, base_config_path):
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
    return config


def cache_area_kb(config):
    """Area proxy: total data capacity of the tuned caches (L2 + LLC), in KB."""
    block_bytes = config["block_size"]
    total_bytes = 0
    for cache_name in ["L2C", "LLC"]:
        cache = config[cache_name]
        total_bytes += cache["sets"] * cache["ways"] * block_bytes
    return total_bytes / 1024.0


def within_budget(knobs, soc_name, base_config_path):
    """Hard constraint: a config is a candidate only if it fits the SoC's area budget."""
    config = build_config(knobs, soc_name, base_config_path)
    return cache_area_kb(config) <= AREA_BUDGET_KB[soc_name]


def make_config(knobs, soc_name, base_config_path, output_dir):
    """Write the ChampSim config JSON for one SoC + knobs; return its path."""
    config = build_config(knobs, soc_name, base_config_path)
    name = config["executable_name"]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name + ".json")
    with open(output_path, "w") as output_file:
        json.dump(config, output_file, indent=2)
    return output_path
