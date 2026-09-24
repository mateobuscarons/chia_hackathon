"""The SoC a run tunes: one chip's physical description, its area budget and the name
its designs are keyed by.

`SOC` picks one, and everything else reads the chip from here, so adding a chip is one
entry in this file. This is the ONLY file in the loop that names a chip: everything
downstream reads numbers (`council_loop.chip`), never a name or a chip class.

One core, three levels. `frequency_mhz` and `process_nm` are the physical chip - they set
the clock every latency is expressed in and the node CACTI characterises the caches at.
`profile` is the handful of fields this SoC overrides on ChampSim's stock core and memory.
`area_budget_kb` is the L2 + LLC data capacity it may spend.

A profile may set a knob's field (the LLC's ways, an MSHR depth): that is the chip's
stock hierarchy, the design the search starts from.
"""

import os

SOC_PROFILES = {
    # The next generation, same architecture family, four things moved: the node (32 -> 22
    # nm, so capacity is half the silicon and half the energy per access), the core (4-wide
    # to 6-wide, ROB 256 to 384), the memory (DDR4-3200 to a DDR5-class 4800: half again
    # the bandwidth at slightly worse latency) and the workloads it must serve. The silicon
    # the shrink freed went to more cores, so cache silicon is capped in mm2 over all three
    # levels (CACTI at 22 nm) and last generation's design, 9.46 mm2, cannot be ported. The
    # stock is the obvious engineering answer - the same design with the last level shrunk
    # until it fits, 3.07 mm2 - and it is the number to beat.
    "next": {
        "chip": "F_next",
        "area_budget_kb": 9216,
        "area_budget_mm2": 4.0,
        # No more power than what shipped: the shrunk stock draws 0.98 W of cache and
        # off-chip traffic on this suite. Checked after measurement, unlike the silicon.
        "power_budget_w": 1.0,
        # No workload slower than on the stock, at all: the old workload the chip was built
        # for is kept, not traded for the mean.
        "regression_tolerance": 0.0,
        "frequency_mhz": 3800,
        "process_nm": 22,
        "profile": {
            "ooo_cpu": {"rob_size": 384, "fetch_width": 8, "decode_width": 8,
                        "dispatch_width": 8, "execute_width": 6, "retire_width": 6,
                        "lq_size": 160, "sq_size": 96, "scheduler_size": 192},
            "L1D": {"sets": 64, "ways": 8, "prefetcher": "next_line", "mshr_size": 16},
            "L2C": {"sets": 1024, "ways": 8, "prefetcher": "spp_dev",
                    "replacement": "srrip", "mshr_size": 32},
            "LLC": {"sets": 2048, "ways": 16, "prefetcher": "no",
                    "replacement": "srrip", "mshr_size": 128},
            "physical_memory": {"data_rate": 4800, "channels": 2,
                                "tCAS": 40, "tRCD": 39, "tRP": 39, "tRAS": 76},
        },
    },
    # The same chip with the three caps off: the ablation that says what the caps cost the
    # search. Same chip name, profile, clock, node and KB budget, so its designs share the
    # `next` tables; only the silicon cap, the power cap and the per-workload floor are gone.
    "nocap": {
        "chip": "F_next",
        "area_budget_kb": 9216,
        "frequency_mhz": 3800,
        "process_nm": 22,
        "profile": {
            "ooo_cpu": {"rob_size": 384, "fetch_width": 8, "decode_width": 8,
                        "dispatch_width": 8, "execute_width": 6, "retire_width": 6,
                        "lq_size": 160, "sq_size": 96, "scheduler_size": 192},
            "L1D": {"sets": 64, "ways": 8, "prefetcher": "next_line", "mshr_size": 16},
            "L2C": {"sets": 1024, "ways": 8, "prefetcher": "spp_dev",
                    "replacement": "srrip", "mshr_size": 32},
            "LLC": {"sets": 2048, "ways": 16, "prefetcher": "no",
                    "replacement": "srrip", "mshr_size": 128},
            "physical_memory": {"data_rate": 4800, "channels": 2,
                                "tCAS": 40, "tRCD": 39, "tRP": 39, "tRAS": 76},
        },
    },

}

NAME = os.environ.get("SOC", "next")
if NAME not in SOC_PROFILES:
    raise RuntimeError("unknown SOC {}: choose one of {}".format(NAME, sorted(SOC_PROFILES)))

CHIP = SOC_PROFILES[NAME]["chip"]
AREA_BUDGET_KB = SOC_PROFILES[NAME]["area_budget_kb"]
AREA_BUDGET_MM2 = SOC_PROFILES[NAME].get("area_budget_mm2")   # None: the KB budget binds
POWER_BUDGET_W = SOC_PROFILES[NAME].get("power_budget_w")     # None: power is reported, not capped

# A brief changes after a chip is under design: the floorplan takes silicon back, or gives
# some of it up; the power envelope is cut or raised. AREA_CAP and POWER_CAP move a cap
# without inventing a second chip, so the design space, the latencies and the workloads
# are unchanged and only the line moves. The run's report records what was in force.
if os.environ.get("AREA_CAP"):
    AREA_BUDGET_MM2 = float(os.environ["AREA_CAP"])
if os.environ.get("POWER_CAP"):
    POWER_BUDGET_W = float(os.environ["POWER_CAP"])
REGRESSION_TOLERANCE = SOC_PROFILES[NAME].get("regression_tolerance")   # None: no per-workload floor
PROFILE = SOC_PROFILES[NAME]["profile"]
FREQUENCY_MHZ = SOC_PROFILES[NAME]["frequency_mhz"]
PROCESS_NM = SOC_PROFILES[NAME]["process_nm"]
