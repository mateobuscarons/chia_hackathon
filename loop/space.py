"""What a design is: thirteen knobs over the three cache levels, and an area budget.

This is the vocabulary everything else speaks - the search, the agent and the
memory all pass designs around as a dict of these knobs, and name them with
`config_name`. Nothing here knows what a simulator is; turning a design into
ChampSim's own configuration is `loop.simulate`.

Sizes, associativity, a prefetcher at every level, replacement at L2 and LLC, and
miss-handling depth (MSHRs). 6.6 million designs before the area budget: they are
sampled (`random_feasible_designs`) or checked one at a time (`in_space` +
`within_budget`), never enumerated. PLACEHOLDER values, team to review.
"""

import random

CHIP = "C_server"

# Hard area budget for L2 + LLC data capacity, in KB. A design over budget is not a candidate.
AREA_BUDGET_KB = 4608

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


def knobs_changed(knobs, reference_knobs):
    """The knobs on which `knobs` differs from `reference_knobs`: {knob: (from, to)}."""
    changed = {}
    for knob in SEARCH_SPACE:
        if str(knobs.get(knob)) != str(reference_knobs.get(knob)):
            changed[knob] = (reference_knobs.get(knob), knobs.get(knob))
    return changed


def cache_area_kb(knobs):
    """Area proxy: data capacity of L2 plus LLC, in KB, at 64-byte blocks. The L1D
    and the MSHRs carry no area here (a known simplification)."""
    block_bytes = 64
    l2_bytes = int(knobs["l2_sets"]) * int(knobs["l2_ways"]) * block_bytes
    llc_bytes = int(knobs["llc_sets"]) * int(knobs["llc_ways"]) * block_bytes
    return (l2_bytes + llc_bytes) / 1024.0


def fit_to_budget(knobs):
    """A design read onto the area budget: shrink the LLC one step at a time, then the L2,
    until it fits."""
    fitted = dict(knobs)
    for knob in ["llc_sets", "llc_ways", "l2_sets", "l2_ways"]:
        while not within_budget(fitted):
            values = SEARCH_SPACE[knob]
            position = None
            for index, value in enumerate(values):
                if str(value) == str(fitted[knob]):
                    position = index
            if position is None or position == 0:
                break
            fitted[knob] = values[position - 1]
    return fitted


def within_budget(knobs):
    """Hard constraint: a design is a candidate only if it fits the area budget."""
    return cache_area_kb(knobs) <= AREA_BUDGET_KB


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
    knob-value listing. The result tables and the binary cache are keyed by it, and
    the name carries the whole design, so nothing has to store both."""
    # No "=" in names: make would read the target as a variable assignment.
    parts = []
    for knob in sorted(knobs.keys()):
        parts.append("{}-{}".format(knob, knobs[knob]))
    return CHIP + "_" + "_".join(parts)
