"""What a design is: thirteen knobs over the three cache levels, and an area budget.

This is the vocabulary everything else speaks - the search, the agent and the
memory all pass designs around as a dict of these knobs, and name them with
`config_name`. Nothing here knows what a simulator is; turning a design into
ChampSim's own configuration is `council_loop.simulate`.

Sizes, associativity, a prefetcher at every level, replacement at L2 and LLC, and
miss-handling depth (MSHRs). The sizes and associativities are derived from the chip
being tuned - a ladder around the hierarchy it ships with, cut to what CACTI will build
and what the caps leave standing - so adding an SoC never means writing a value list.
The space is sampled (`random_feasible_designs`) or checked one at a time (`in_space` +
`within_budget`), never enumerated.
"""

import random

from council_loop import chip                         # the chip this run tunes: its name, cores and area
from council_loop import socs
from council_loop.socs import AREA_BUDGET_KB, AREA_BUDGET_MM2, POWER_BUDGET_W, REGRESSION_TOLERANCE

# ---------------------------------------------------------------- the reachable shapes ----

# How far the ladder reaches either side of the shape the chip already ships with. Its own
# hierarchy is the only anchor the loop has for what "big" and "small" mean on this chip,
# so the sizes it may try are built around it instead of written down per chip.
LADDER_DOWN = 4
LADDER_UP = 4

# The associativities a cache is actually built with. This is a property of caches rather
# than of any one chip, so it is the part of a shape that is the same everywhere.
WAYS_MENU = [4, 8, 12, 16]


def _buildable(level, sets, ways):
    """True when CACTI can characterise this shape and the level alone fits every cap.
    A level that breaks a cap on its own can never be part of a legal design, so it is
    dropped from the vocabulary rather than drawn and refused later at a design's cost."""
    try:
        area = chip.area_mm2(level, sets, ways)
        leakage = chip.leakage_w(level, sets, ways)
    except RuntimeError:
        return False                      # CACTI refuses arrays below a few kilobytes
    if AREA_BUDGET_MM2 is not None and area > AREA_BUDGET_MM2:
        return False
    if POWER_BUDGET_W is not None and leakage > POWER_BUDGET_W:
        return False
    return True


def shape_ladder(level):
    """The sets and the ways this chip can build at one level: a ladder around its own
    stock shape, cut to what CACTI will characterise and what the caps leave standing.
    The stock's own associativity is always kept, so the chip the search starts from is
    always inside the space."""
    stock = chip.base_config()[level]
    stock_sets = int(stock["sets"])
    stock_ways = int(stock["ways"])
    candidate_ways = list(WAYS_MENU)
    if stock_ways not in candidate_ways:
        candidate_ways.append(stock_ways)
    candidate_sets = []
    sets = max(1, stock_sets // LADDER_DOWN)
    while sets <= stock_sets * LADDER_UP:
        candidate_sets.append(sets)
        sets *= 2
    usable_sets = []
    usable_ways = []
    for sets in candidate_sets:
        for ways in candidate_ways:
            if not _buildable(level, sets, ways):
                continue
            if sets not in usable_sets:
                usable_sets.append(sets)
            if ways not in usable_ways:
                usable_ways.append(ways)
    usable_sets.sort()
    usable_ways.sort()
    return usable_sets, usable_ways


# A shape is this chip's business, so it is derived (`shape_ladder`). Which prefetcher,
# which replacement policy and how deep the miss queue are the simulator's menu - what
# ChampSim can be told to build - so they are the same on every chip and are written here.
_l1d_sets, _l1d_ways = shape_ladder("L1D")
_l2_sets, _l2_ways = shape_ladder("L2C")
_llc_sets, _llc_ways = shape_ladder("LLC")

SEARCH_SPACE = {
    "l1d_sets": _l1d_sets,
    "l1d_ways": _l1d_ways,
    "l1d_prefetcher": ["no", "next_line", "ip_stride", "va_ampm_lite"],
    "l2_sets": _l2_sets,
    "l2_ways": _l2_ways,
    "l2_prefetcher": ["no", "spp_dev", "ip_stride", "next_line", "va_ampm_lite"],
    "l2_replacement": ["lru", "srrip", "drrip", "ship"],
    "llc_sets": _llc_sets,
    "llc_ways": _llc_ways,
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
    """Area proxy: data capacity of L2 plus LLC, in KB, at 64-byte blocks, every
    instance of a level counted - a private level is built once per core, so on a
    four-core chip an L2 costs four times its capacity. The L1D and the MSHRs carry no
    area here (a known simplification). `chip.area_mm2` gives the silicon beside it."""
    block_bytes = 64
    l2_bytes = int(knobs["l2_sets"]) * int(knobs["l2_ways"]) * block_bytes
    llc_bytes = int(knobs["llc_sets"]) * int(knobs["llc_ways"]) * block_bytes
    return (l2_bytes + llc_bytes) / 1024.0


LEVEL_SHAPES = [("L1D", "l1d_sets", "l1d_ways"), ("L2C", "l2_sets", "l2_ways"), ("LLC", "llc_sets", "llc_ways")]


def silicon_mm2(knobs):
    """The silicon the three levels take on this chip, from CACTI at its node, every
    instance of a private level counted."""
    total = 0.0
    for level, sets_knob, ways_knob in LEVEL_SHAPES:
        total += chip.area_mm2(level, int(knobs[sets_knob]), int(knobs[ways_knob]))
    return total


DRAM_PJ_PER_BIT = 15.0   # a DDR4-class figure for a bit moved off chip; an assumption, not a measurement


def power_split(knobs, metrics, workloads):
    """Where the power goes, in watts by level and off chip, over the workloads given: every
    access pays its level's read energy, every fill its write energy, every level leaks
    for the run's time, and every byte past the last level pays the DRAM figure."""
    gigahertz = socs.FREQUENCY_MHZ / 1000.0
    joules = {"L1D": 0.0, "L2C": 0.0, "LLC": 0.0, "off-chip": 0.0}
    cycles = 0.0
    for workload in workloads:
        run_cycles = metrics["{}:cycles".format(workload)]
        cycles += run_cycles
        seconds = run_cycles / (gigahertz * 1e9)
        for level, sets_knob, ways_knob in LEVEL_SHAPES:
            read_nj, write_nj, leak_mw = chip.energy(level, int(knobs[sets_knob]), int(knobs[ways_knob]))
            hits = metrics.get("{}:{}_hits".format(workload, level), 0)
            misses = metrics.get("{}:{}_misses".format(workload, level), 0)
            joules[level] += (hits + misses) * read_nj * 1e-9 + misses * write_nj * 1e-9
            joules[level] += leak_mw * 1e-3 * seconds
        off_chip_bits = metrics.get("{}:LLC_misses".format(workload), 0) * 64 * 8
        joules["off-chip"] += off_chip_bits * DRAM_PJ_PER_BIT * 1e-12
    seconds = cycles / (gigahertz * 1e9)
    split = {}
    for part in joules:
        split[part] = joules[part] / seconds
    return split


def watts(knobs, metrics, workloads):
    """Cache plus off-chip power over the suite."""
    return sum(power_split(knobs, metrics, workloads).values())


def violations(metrics, workloads, stock_metrics):
    """The caps a measured design breaks, one line each; empty when it is feasible. These
    are the caps only the counters can check, so they are read after measurement: the
    power cap, and the floor under every workload set by what the stock reached on it.
    Workloads are named by position (W1, W2, ...), as the prompts name them."""
    broken = []
    if POWER_BUDGET_W is not None and metrics["watts"] > POWER_BUDGET_W:
        # Where the power goes, so the team can see which level to shrink to make room.
        split = metrics.get("watts_by_part") or {}
        where = ", ".join("{} {:.2f}".format(part, split[part]) for part in split)
        broken.append("power {:.2f} W over the {} W cap{}".format(
            metrics["watts"], POWER_BUDGET_W, " ({})".format(where) if where else ""))
    if REGRESSION_TOLERANCE is not None:
        for index, workload in enumerate(workloads):
            key = "{}:ipc".format(workload)
            if key not in metrics or key not in stock_metrics:
                continue
            got = metrics[key]
            floor = stock_metrics[key] * (1.0 - REGRESSION_TOLERANCE)
            if got < floor:
                broken.append("W{} slower than the stock ({:.3f} against {:.3f})".format(
                    index + 1, got, floor))
    return broken


def leakage_w(knobs):
    """What the three levels leak before a single access: the part of the power that the
    shape alone decides, so it is known before simulation."""
    total = 0.0
    for level, sets_knob, ways_knob in LEVEL_SHAPES:
        total += chip.leakage_w(level, int(knobs[sets_knob]), int(knobs[ways_knob]))
    return total


def hierarchy_is_ordered(knobs):
    """A hierarchy only reads outwards: each array holds more than the one in front of it
    and is no quicker to answer. A level smaller or faster than the level it backs is not
    a level, and the search buys one if nothing forbids it - left free, it bought a first
    level of 1 MB at 6 cycles sitting in front of a 512 KB level at 4.

    Arrays are compared as built, not per core: a shared last level is one array serving
    every core, which is why a chip may hold more private capacity in total than shared
    and still be ordered. No cycle count and no capacity is written here - the ordering is
    the whole rule, so it reads on any chip and any number of levels."""
    inner_blocks = None
    inner_cycles = None
    for level, sets_knob, ways_knob in LEVEL_SHAPES:
        sets = int(knobs[sets_knob])
        ways = int(knobs[ways_knob])
        blocks = sets * ways
        cycles = chip.latency(level, sets, ways)
        if inner_blocks is not None:
            # At least as much, not more: these caches are non-inclusive, so a level the
            # same size as the one in front of it holds different lines and doubles the
            # capacity at that tier. Shipping parts do exactly this - a 1 MB private level
            # behind a last level of about the same size per core.
            if blocks < inner_blocks:
                return False
            if cycles < inner_cycles:
                return False
        inner_blocks = blocks
        inner_cycles = cycles
    return True


def within_budget(knobs):
    """Hard constraint: a design is a candidate only if it fits the area budget. A chip
    with a cap in mm2 is held to its silicon; the others to the KB capacity proxy. Under a
    power cap a shape that leaks more than the cap on its own is refused before simulation."""
    if POWER_BUDGET_W is not None and leakage_w(knobs) > POWER_BUDGET_W:
        return False
    if not hierarchy_is_ordered(knobs):
        return False
    if AREA_BUDGET_MM2 is not None:
        return silicon_mm2(knobs) <= AREA_BUDGET_MM2
    return cache_area_kb(knobs) <= AREA_BUDGET_KB


def budget_text(knobs):
    """What a design spends against the budget that binds on this chip, for a refusal
    or a report line: "3.07 of 4.0 mm2 of silicon" or "9000 of 9216 KB in L2 + LLC". A
    hierarchy that is not ordered says so first, with the two levels that cross, because
    that refusal has nothing to do with the budget and reads wrongly without the reason."""
    if not hierarchy_is_ordered(knobs):
        shapes = []
        for level, sets_knob, ways_knob in LEVEL_SHAPES:
            shapes.append("{} {} KB at {} cycles".format(
                level, int(knobs[sets_knob]) * int(knobs[ways_knob]) * chip.BLOCK_BYTES // 1024,
                chip.latency(level, int(knobs[sets_knob]), int(knobs[ways_knob]))))
        return "an unordered hierarchy (each level must hold at least as much as the one in front of it and answer no faster): " + ", ".join(shapes)
    if AREA_BUDGET_MM2 is not None:
        text = "{:.2f} of the chip's {} mm2 of cache silicon".format(silicon_mm2(knobs), AREA_BUDGET_MM2)
        if POWER_BUDGET_W is not None:
            text += ", {:.2f} W of leakage alone against the {} W cap".format(leakage_w(knobs), POWER_BUDGET_W)
        return text
    return "{:.0f} of the chip's {} KB in L2 + LLC".format(cache_area_kb(knobs), AREA_BUDGET_KB)


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
    """Stable, filesystem-safe name for one design: the chip, its revision and a
    sorted knob-value listing. The result tables and the binary cache are keyed by it,
    and the name carries the whole design, so nothing has to store both. The revision
    covers everything about the chip that is not a knob - the clock, the node, the
    profile - so a row or a binary from a chip whose latencies have changed is never
    served for this one."""
    # No "=" in names: make would read the target as a variable assignment.
    parts = []
    for knob in sorted(knobs.keys()):
        parts.append("{}-{}".format(knob, knobs[knob]))
    return chip.NAME + "_" + "_".join(parts)
