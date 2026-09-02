"""Tier-A search space and ChampSim config generation.

PLACEHOLDER knob values — the team must review these before the real runs.
Tier A is deliberately small (81 configs) so a dense reference sweep is
affordable and "% of true optimum" claims stay rigorous.
"""

import json
import os

SEARCH_SPACE = {
    "l2_sets": [512, 1024, 2048],                    # 256KB / 512KB / 1MB
    "llc_sets": [1024, 2048, 4096],                  # 1MB / 2MB / 4MB
    "l2_prefetcher": ["no", "spp_dev", "ip_stride"],
    "llc_replacement": ["lru", "srrip", "drrip"],
}


def all_configurations():
    """Every knob combination in the search space, as a list of dicts."""
    combinations = []
    for l2_sets in SEARCH_SPACE["l2_sets"]:
        for llc_sets in SEARCH_SPACE["llc_sets"]:
            for l2_prefetcher in SEARCH_SPACE["l2_prefetcher"]:
                for llc_replacement in SEARCH_SPACE["llc_replacement"]:
                    knobs = {
                        "l2_sets": l2_sets,
                        "llc_sets": llc_sets,
                        "l2_prefetcher": l2_prefetcher,
                        "llc_replacement": llc_replacement,
                    }
                    combinations.append(knobs)
    return combinations


def config_name(knobs):
    """Stable, filesystem-safe name for one knob combination."""
    return "cs_l2s{}_pf-{}_llcs{}_rp-{}".format(
        knobs["l2_sets"], knobs["l2_prefetcher"],
        knobs["llc_sets"], knobs["llc_replacement"],
    )


def make_config(knobs, base_config_path, output_dir):
    """Write the ChampSim config JSON for one combination; return its path."""
    with open(base_config_path) as base_file:
        config = json.load(base_file)

    name = config_name(knobs)
    config["executable_name"] = name
    config["L2C"]["sets"] = knobs["l2_sets"]
    config["L2C"]["prefetcher"] = knobs["l2_prefetcher"]
    config["LLC"]["sets"] = knobs["llc_sets"]
    config["LLC"]["replacement"] = knobs["llc_replacement"]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name + ".json")
    with open(output_path, "w") as output_file:
        json.dump(config, output_file, indent=2)
    return output_path
