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
PROFILE = SOC_PROFILES[NAME]["profile"]
CORES = SOC_PROFILES[NAME].get("cores", 1)
FREQUENCY_MHZ = SOC_PROFILES[NAME]["frequency_mhz"]
PROCESS_NM = SOC_PROFILES[NAME]["process_nm"]
