"""The search space and ChampSim config generation.

Thirteen knobs over the three cache levels: sizes, associativity, a prefetcher at
every level, replacement at L2 and LLC, and miss-handling depth (MSHRs). Cache
latency is DERIVED FROM SIZE by ChampSim's own formula, so a bigger cache is
slower and capacity is never free. 6.6 million designs before the area budget:
designs are sampled (random_feasible_designs) or checked one at a time
(in_space + within_budget), never enumerated. PLACEHOLDER values, team to review.
"""

import json
import os
import random
import threading

# ---------------------------------------------------------------- the chip ----

CHIP = "C_server"

PROFILE = {
    "ooo_cpu": {"rob_size": 512, "lq_size": 192, "sq_size": 114, "scheduler_size": 192},
    "L1D": {"mshr_size": 32},
    "L2C": {"mshr_size": 64},
    "LLC": {"ways": 16, "mshr_size": 128},
    "physical_memory": {"data_rate": 3200, "channels": 2},
}

# Hard area budget for L2 + LLC data capacity, in KB. A design over budget is not a candidate.
AREA_BUDGET_KB = 4608


def describe():
    """The chip in one line for the agent's prompt: its overrides over the stock
    ChampSim core and memory, and its area budget. Fields that are search knobs
    (sets, ways, mshr_size, prefetcher, replacement) are not chip parameters and
    are left out."""
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


SEARCH_SPACE = {
    "l1d_sets": [32, 64, 128],
    "l1d_ways": [8, 12],
    "l1d_prefetcher": ["no", "next_line", "ip_stride", "va_ampm_lite"],
    "l2_sets": [256, 512, 1024, 2048],
    "l2_ways": [4, 8, 16],
    "l2_prefetcher": ["no", "spp_dev", "ip_stride", "next_line", "va_ampm_lite"],
    "l2_replacement": ["lru", "srrip", "drrip", "ship"],
    "llc_sets": [1024, 2048, 4096, 8192],
    "llc_ways": [8, 16],
    "llc_prefetcher": ["no", "next_line", "ip_stride", "spp_dev"],
    "llc_replacement": ["lru", "srrip", "drrip", "ship"],
    "l2_mshr": [16, 32, 64],
    "llc_mshr": [32, 64, 128],
}

# The knobs in the order the tables show they matter: prefetchers, replacement and
# capacity first, the L1D geometry (near-inert here, free of area cost) last. The
# memory digest and the agent's fallback walk knobs in this order.
KNOB_PRIORITY = ["l2_prefetcher", "llc_replacement", "llc_sets", "l2_sets", "llc_prefetcher", "l1d_prefetcher",
                 "l2_replacement", "llc_ways", "l2_ways", "l2_mshr", "llc_mshr", "l1d_sets", "l1d_ways"]

# Caches whose fixed latency is dropped so ChampSim derives it from size.
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


def in_space(knobs):
    """Every knob present, every value allowed (the LLM may write 1024 as "1024")."""
    if len(knobs) != len(SEARCH_SPACE):
        return False
    for knob in SEARCH_SPACE:
        if knob not in knobs:
            return False
        allowed = []
        for value in SEARCH_SPACE[knob]:
            allowed.append(str(value))
        if str(knobs[knob]) not in allowed:
            return False
    return True


def typed_knobs(knobs):
    """The same design with every value carrying the space's own type."""
    typed = {}
    for knob in SEARCH_SPACE:
        for value in SEARCH_SPACE[knob]:
            if str(value) == str(knobs[knob]):
                typed[knob] = value
    return typed


def same_knobs(knobs_a, knobs_b):
    """Compare as strings: a value may arrive as 1024 or "1024"."""
    for knob in SEARCH_SPACE:
        if str(knobs_a.get(knob)) != str(knobs_b.get(knob)):
            return False
    return True


def knobs_changed(knobs, reference_knobs):
    """The knobs on which `knobs` differs from `reference_knobs`: {knob: (from, to)}."""
    changed = {}
    for knob in SEARCH_SPACE:
        if str(knobs.get(knob)) != str(reference_knobs.get(knob)):
            changed[knob] = (reference_knobs.get(knob), knobs.get(knob))
    return changed


def random_feasible_designs(how_many, seed=0):
    """`how_many` distinct designs drawn uniformly from the feasible set by
    rejection sampling (a fixed seed gives the same list every time)."""
    generator = random.Random(seed)
    designs = []
    seen = set()
    while len(designs) < how_many:
        knobs = {}
        for knob in SEARCH_SPACE:
            knobs[knob] = generator.choice(SEARCH_SPACE[knob])
        if not within_budget(knobs):
            continue
        key = config_name(knobs)
        if key in seen:
            continue
        seen.add(key)
        designs.append(knobs)
    return designs


def config_name(knobs):
    """Stable, filesystem-safe name for one design: the chip and a sorted
    knob-value listing. The result tables and the binary cache are keyed by it."""
    # No "=" in names: make would read the target as a variable assignment.
    parts = []
    for knob in sorted(knobs.keys()):
        parts.append("{}-{}".format(knob, knobs[knob]))
    return CHIP + "_" + "_".join(parts)


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


def cache_area_kb(knobs):
    """Area proxy: data capacity of L2 plus LLC, in KB, at 64-byte blocks. The L1D
    and the MSHRs carry no area here (a known simplification)."""
    block_bytes = 64
    l2_bytes = int(knobs["l2_sets"]) * int(knobs["l2_ways"]) * block_bytes
    llc_bytes = int(knobs["llc_sets"]) * int(knobs["llc_ways"]) * block_bytes
    return (l2_bytes + llc_bytes) / 1024.0


def within_budget(knobs):
    """Hard constraint: a design is a candidate only if it fits the area budget."""
    return cache_area_kb(knobs) <= AREA_BUDGET_KB


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
