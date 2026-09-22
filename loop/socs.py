"""The SoC a run tunes: one chip's physical description, its area budget and the name
its designs are keyed by.

`SOC` picks one, and everything else reads the chip from here, so adding a chip is one
entry in this file. This is the ONLY file in the loop that names a chip: everything
downstream reads numbers (`loop.chip`), never a name or a chip class.

A profile has three parts. `frequency_mhz` and `process_nm` are the physical chip - they
set the clock every latency is expressed in and the node CACTI characterises the caches
at. `profile` is the handful of fields this SoC overrides on ChampSim's stock core and
memory. `area_budget_kb` is the L2 + LLC data capacity it may spend, a private level
counted once per core.

A profile may set a knob's field (the LLC's ways, an MSHR depth): that is the chip's
stock hierarchy, the design the search starts from.
"""

import os

SOC_PROFILES = {
    # Server-class: wide, deep ROB, big MSHRs, two channels of fast memory.
    "server": {
        "chip": "C_server",
        "area_budget_kb": 4608,
        "frequency_mhz": 4000,
        "process_nm": 32,
        "profile": {
            "ooo_cpu": {"rob_size": 512, "lq_size": 192, "sq_size": 114, "scheduler_size": 192},
            "L1D": {"mshr_size": 32},
            "L2C": {"mshr_size": 64},
            "LLC": {"ways": 16, "mshr_size": 128},
            "physical_memory": {"data_rate": 3200, "channels": 2},
        },
    },
    # Mobile-class: narrow core, short ROB, shallow MSHRs, one channel of slow memory,
    # half the clock at an older node, and less than half the area. Capacity is the
    # binding constraint here, where it is not on the server.
    "mobile": {
        "chip": "A_mobile",
        "area_budget_kb": 2048,
        "frequency_mhz": 2000,
        "process_nm": 45,
        "profile": {
            "ooo_cpu": {"rob_size": 128, "fetch_width": 4, "decode_width": 4, "dispatch_width": 4,
                        "execute_width": 2, "retire_width": 4, "lq_size": 48, "sq_size": 32,
                        "scheduler_size": 64},
            "L1D": {"sets": 64, "ways": 8, "mshr_size": 8},
            "L2C": {"ways": 4, "mshr_size": 16},
            "LLC": {"ways": 8},
            "physical_memory": {"data_rate": 1600, "channels": 1},
        },
    },
    # Last generation, 32 nm. A single wide core on DDR4, and a cache hierarchy that was
    # tuned for THIS node, THIS clock and THIS memory: 32 KB L1D, a 512 KB L2 and an 8 MB
    # last level, which is exactly its area budget. It is the design a team already
    # shipped, not a weak configuration left to be fixed.
    "prev": {
        "chip": "E_prev",
        "area_budget_kb": 8704,
        "frequency_mhz": 3200,
        "process_nm": 32,
        "profile": {
            "ooo_cpu": {"rob_size": 256, "fetch_width": 6, "decode_width": 6,
                        "dispatch_width": 6, "execute_width": 4, "retire_width": 5,
                        "lq_size": 128, "sq_size": 72, "scheduler_size": 128},
            "L1D": {"sets": 64, "ways": 8, "prefetcher": "next_line", "mshr_size": 16},
            "L2C": {"sets": 1024, "ways": 8, "prefetcher": "spp_dev",
                    "replacement": "srrip", "mshr_size": 32},
            "LLC": {"sets": 8192, "ways": 16, "prefetcher": "no",
                    "replacement": "srrip", "mshr_size": 128},
            "physical_memory": {"data_rate": 3200, "channels": 2,
                                "tCAS": 24, "tRCD": 24, "tRP": 24, "tRAS": 52},
        },
    },
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
    # A many-core cloud CPU, four of its cores modelled: each keeps its own L1 and L2 and
    # all four sit behind one shared last level, which is how a private byte comes to cost
    # four times a shared one. The memory is two narrow channels of the part's own kind, so
    # each core sees about the bandwidth a core sees on the real chip - roughly a tenth of
    # what a single-core model gives it, and the reason a prefetcher here is a decision
    # rather than a free win. The node is the finest CACTI will characterise, so the
    # millimetres stand in for a finer process and the ratios are the scenario.
    "cloud": {
        "chip": "G_cloud",
        "area_budget_kb": 10240,
        # The floorplan's allowance, written the way a tile budget is written: so many
        # square millimetres of cache per core, whatever the design spends it on. It is a
        # product decision about how many cores fit on a die, not a margin around the
        # hierarchy that shipped - which is why the shipped hierarchy does not fit it.
        "area_budget_mm2": 8.0,
        # The part is allowed 500 W across 72 cores. Cache and the traffic it sends off
        # chip get a small share of that per core.
        "power_budget_w": 4.0,
        # What one tenant may lose so another gains. Four programs share the last level, so
        # any reallocation takes from someone: at zero tolerance a design 2.9 % faster over
        # the machine was refused for costing one core 1.7 %, and nothing ever moved. This
        # is the variance a shared machine is actually sold with.
        "regression_tolerance": 0.05,
        "frequency_mhz": 3440,
        "process_nm": 22,
        "cores": 4,
        "profile": {
            "num_cores": 4,
            "ooo_cpu": {"rob_size": 320, "fetch_width": 6, "decode_width": 6,
                        "dispatch_width": 6, "execute_width": 6, "retire_width": 8,
                        "lq_size": 96, "sq_size": 64, "scheduler_size": 160},
            "L1D": {"sets": 256, "ways": 4, "prefetcher": "next_line", "mshr_size": 20},
            "L2C": {"sets": 2048, "ways": 8, "prefetcher": "spp_dev",
                    "replacement": "srrip", "mshr_size": 32},
            "LLC": {"sets": 8192, "ways": 12, "prefetcher": "no",
                    "replacement": "srrip", "mshr_size": 128},
            "physical_memory": {"data_rate": 8533, "channels": 2, "channel_width": 2,
                                # Counted in controller cycles, which are short at this
                                # rate: 20 ns to open a row that is already open, 56 ns
                                # when it is not - the latency this kind of memory trades
                                # for its bandwidth.
                                "tCAS": 85, "tRCD": 77, "tRP": 77, "tRAS": 179},
        },
    },
    # Four cores: stock cores with a deeper ROB, private L1/L2 each, one shared LLC, two
    # channels, and room for four private L2s beside it. A design is measured on mixes of
    # four traces, one per core.
    "quad": {
        "chip": "D_quad",
        "area_budget_kb": 8192,
        "frequency_mhz": 4000,
        "process_nm": 32,
        "cores": 4,
        "profile": {
            "num_cores": 4,
            "ooo_cpu": {"rob_size": 256},
            "L2C": {"mshr_size": 32},
            "LLC": {"ways": 16, "mshr_size": 128},
            "physical_memory": {"data_rate": 3200, "channels": 2},
        },
    },
}

NAME = os.environ.get("SOC", "server")
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
CORES = SOC_PROFILES[NAME].get("cores", 1)
FREQUENCY_MHZ = SOC_PROFILES[NAME]["frequency_mhz"]
PROCESS_NM = SOC_PROFILES[NAME]["process_nm"]
