"""The CPI stack: a mechanistic mean function for the surrogate (offline finding).

The idea in one line: a design changes IPC only by changing how many misses
happen at each level and what each miss costs, so write that down as a formula
instead of asking a GP to learn it again on every new chip.

Cost per miss comes from the interval model (Eyerman, Eeckhout, Karkhanis,
Smith, TOCS 2009): an isolated last-level miss costs a full memory access, but
independent last-level misses that fall inside one ROB window overlap and are
paid once, so the per-miss cost is the memory latency divided by the number of
misses in flight - capped by the MSHRs, because that is how many can be
outstanding at all.

Everything the formula needs about a chip is public geometry (ROB size, MSHR
counts, DRAM data rate and timings), so it can be evaluated for a chip that has
never been simulated. The coefficients are global: fitted once on the old chips
and reused on the new one. Nothing is fitted per program - a program enters
only through its miss counts and its trace profile.
"""

import json
import math

from loop.socs import apply_profile

# ChampSim derives a cache's hit latency from its size when the fixed value is
# absent: champsim/inc/cache_builder.h:316.
LATENCY_EXPONENT = 0.343
LATENCY_SCALE = 0.416

_chip_cache = {}


def cache_latency_cycles(sets, ways):
    """ChampSim's own latency-from-size formula, in CPU cycles."""
    entries = sets * ways
    return round(math.pow(entries, LATENCY_EXPONENT) * LATENCY_SCALE)


def chip_parameters(soc_name, base_config_path):
    """The public spec of one chip: what the formula is allowed to know.

    Read from the SoC profile applied to ChampSim's base config, never fitted,
    so an unseen chip needs no extra work.
    """
    key = (soc_name, base_config_path)
    if key in _chip_cache:
        return _chip_cache[key]
    with open(base_config_path) as config_file:
        config = json.load(config_file)
    config = apply_profile(config, soc_name)
    core = config["ooo_cpu"][0]
    memory = config["physical_memory"]

    cpu_mhz = core["frequency"]
    # The memory controller runs at half the data rate (double data rate).
    memory_mhz = memory["data_rate"] / 2.0
    cycles_per_memory_cycle = cpu_mhz / memory_mhz
    # A DRAM read that has to close and reopen a row pays tRP + tRCD + tCAS;
    # a row-buffer hit pays only tCAS. The average row-hit rate is unknown here,
    # so the fitted LLC coefficient absorbs it and we use the row-miss cost.
    memory_cycles = memory["tRP"] + memory["tRCD"] + memory["tCAS"]

    parameters = {
        "rob_size": core["rob_size"],
        "dram_cycles": memory_cycles * cycles_per_memory_cycle,
    }
    _chip_cache[key] = parameters
    return parameters


def independent_fraction(profile, coefficients):
    """How many of a program's misses are independent enough to overlap.

    The interval model is explicit that dependent misses never overlap: a
    pointer chase pays every miss in full however many MSHRs the chip has, a
    stride walk pays them together. The trace says which kind of program this
    is (its regular-stride fraction), so this stays global - the coefficients
    are the same for every program, only the profile differs.
    """
    regular = profile["stride_regular_fraction"]
    fraction = coefficients["overlap_base"] + coefficients["overlap_stride"] * regular
    return min(max(fraction, 0.0), 1.0)


def misses_in_flight(llc_mpki, rob_size, llc_mshr, overlap_fraction):
    """How many last-level misses are paid together instead of one by one.

    Inside a window of `rob_size` instructions the program issues
    rob_size * llc_mpki / 1000 last-level misses. They can only overlap if the
    MSHRs can hold them, and only the independent ones actually do.
    """
    misses_per_window = rob_size * llc_mpki / 1000.0
    reachable = min(misses_per_window, float(llc_mshr))
    if reachable <= 1.0:
        return 1.0
    return 1.0 + overlap_fraction * (reachable - 1.0)


def prefetch_timeliness(knobs, coefficients):
    """How much of a memory access a prefetcher hides rather than removes.

    Measured fact (chip C, lbm): turning on spp_dev cuts last-level misses by
    17% and raises IPC by 34%. The misses are still there; they are just
    fetched early, so they cost less. Miss counts alone cannot see this, so it
    needs its own factor - one fitted number per prefetcher value, shared by
    every chip and program. With no such coefficients this is 1.0 and the stack
    is the plain miss-count model.
    """
    factor = 1.0
    l2_key = "timeliness_l2_" + str(knobs["l2_prefetcher"])
    llc_key = "timeliness_llc_" + str(knobs["llc_prefetcher"])
    factor += coefficients.get(l2_key, 0.0)
    factor += coefficients.get(llc_key, 0.0)
    return max(factor, 0.05)


def stall_cycles_per_instruction(knobs, level_mpki, profile, chip, coefficients):
    """The memory part of CPI for one design, from its miss counts.

    Each level is charged only for the misses that stop there, so no miss is
    counted twice: an L1D miss that hits in L2 costs an L2 access, an L2 miss
    that hits in the LLC costs an LLC access, and an LLC miss costs memory.
    """
    l2_latency = cache_latency_cycles(knobs["l2_sets"], knobs["l2_ways"])
    llc_latency = cache_latency_cycles(knobs["llc_sets"], knobs["llc_ways"])

    l1d_mpki = level_mpki["L1D"]
    l2_mpki = level_mpki["L2C"]
    llc_mpki = level_mpki["LLC"]

    stopped_at_l2 = max(l1d_mpki - l2_mpki, 0.0)
    stopped_at_llc = max(l2_mpki - llc_mpki, 0.0)

    overlap = independent_fraction(profile, coefficients)
    in_flight = misses_in_flight(llc_mpki, chip["rob_size"], knobs["llc_mshr"], overlap)
    in_flight = in_flight * prefetch_timeliness(knobs, coefficients)

    stall = 0.0
    stall += coefficients["l2_cost"] * stopped_at_l2 / 1000.0 * l2_latency
    stall += coefficients["llc_cost"] * stopped_at_llc / 1000.0 * llc_latency
    stall += coefficients["dram_cost"] * llc_mpki / 1000.0 * chip["dram_cycles"] / in_flight
    return stall


def predicted_log_speedup(row, chip, coefficients):
    """Log speedup of a design over the chip's own baseline run.

    The baseline's measured CPI is the anchor, so the miss-free part of CPI
    never has to be modelled: it is the same program on the same chip and
    cancels. One baseline simulation per chip and workload is all this needs,
    and every arm already pays for it.
    """
    baseline_cpi = 1.0 / row["baseline_ipc"]
    profile = row["profile"]
    design_stall = stall_cycles_per_instruction(row["knobs"], row["level_mpki"], profile,
                                                chip, coefficients)
    baseline_stall = stall_cycles_per_instruction(row["baseline_knobs"], row["baseline_level_mpki"],
                                                  profile, chip, coefficients)
    design_cpi = baseline_cpi + (design_stall - baseline_stall)
    if design_cpi <= 0.0:
        return 0.0
    return math.log(baseline_cpi / design_cpi)
