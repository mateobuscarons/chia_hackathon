"""SoC profiles: four deliberately different chips, sharing the same knobs.

PLACEHOLDER values - the team must review. Each profile is a set of
overrides applied on top of ChampSim's stock config before the knobs.
B is the stock config; A is smaller/slower, C is bigger/faster, so the same
knob can pay off differently on each. D has four cores with private L1/L2
and one shared LLC (the shape of the CRC-2 multi-core configs), so private
capacity costs four times what shared capacity costs.
"""

SOC_PROFILES = {
    # Small mobile-like core: narrow, short ROB, small L1D, slow memory.
    "A_mobile": {
        "ooo_cpu": {"rob_size": 128, "fetch_width": 4, "decode_width": 4,
                    "dispatch_width": 4, "execute_width": 2, "retire_width": 4,
                    "lq_size": 48, "sq_size": 32, "scheduler_size": 64},
        "L1D": {"sets": 64, "ways": 8, "mshr_size": 8},
        "L2C": {"ways": 4, "mshr_size": 16},
        "LLC": {"ways": 8, "latency": 30},
        "physical_memory": {"data_rate": 1600},
    },
    # Mid-range: stock ChampSim config, untouched.
    "B_midrange": {},
    # Server-class: wide, deep ROB, big MSHRs, two memory channels.
    "C_server": {
        "ooo_cpu": {"rob_size": 512, "lq_size": 192, "sq_size": 114,
                    "scheduler_size": 192},
        "L1D": {"mshr_size": 32},
        "L2C": {"mshr_size": 64},
        "LLC": {"ways": 16, "mshr_size": 128},
        "physical_memory": {"data_rate": 3200, "channels": 2},
    },
    # Four-core: stock cores with a deeper ROB, private L1/L2 per core, one
    # shared LLC, two memory channels. Every "trace" on this chip is a mix of
    # four programs, one per core.
    "D_quad": {
        "num_cores": 4,
        "ooo_cpu": {"rob_size": 256},
        "L2C": {"mshr_size": 32},
        "LLC": {"ways": 16, "mshr_size": 128},
        "physical_memory": {"data_rate": 3200, "channels": 2},
    },
}

# Hard area budget per SoC: one L2 per core plus the shared LLC, in KB.
# PLACEHOLDER - team to review. The objective is "best IPC that fits";
# over-budget configs are not candidates.
AREA_BUDGET_KB = {
    "A_mobile": 2048,
    "B_midrange": 3072,
    "C_server": 4608,
    "D_quad": 8192,
}


def num_cores(soc_name):
    return SOC_PROFILES[soc_name].get("num_cores", 1)


def apply_profile(config, soc_name):
    """Overwrite the base config's fields with one SoC's overrides."""
    overrides = SOC_PROFILES[soc_name]
    for section_name in overrides:
        section_overrides = overrides[section_name]
        if section_name == "num_cores":
            # ChampSim duplicates the first core (and the private caches) to this count.
            config["num_cores"] = section_overrides
            continue
        if section_name == "ooo_cpu":
            # ChampSim stores cores as a list; the first entry is the template.
            target = config["ooo_cpu"][0]
        else:
            target = config[section_name]
        for field_name in section_overrides:
            target[field_name] = section_overrides[field_name]
    return config
