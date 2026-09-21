"""Everything the loop knows about the chip it is tuning, as numbers.

One source of chip facts, so nothing downstream carries a chip name or a branch on a
chip class. Three kinds of fact, each derived once:

  the card        core width, ROB, clock, cores, the bandwidth the memory channels
                  deliver per core cycle, what a DRAM service costs in those cycles,
                  which levels exist once per core and which are shared
  the caches      CACTI 7 in cache mode at this SoC's node: a level's hit latency in
                  this SoC's cycles, and its silicon area
  the config      the SoC's overrides written onto ChampSim's stock configuration

Two chips differ here and nowhere else. A prompt reads these numbers; it never reads
which chip they came from.
"""

import fcntl
import json
import math
import os
import re
import subprocess
import tempfile

from loop import socs

BASE_CONFIG = "champsim/champsim_config.json"
CACTI_BIN = os.environ.get("CACTI", "cacti/cacti")
CACTI_TEMPLATE = os.environ.get("CACTI_TEMPLATE", "cacti/cache.cfg")
CACTI_CACHE = "results/cacti_{}nm.json"

# ChampSim's topology, not a chip's: each core has its own L1 and L2, every core shares
# the last level. A level's capacity therefore costs once per core or once in total, and
# a shared level's counters are the sum over cores.
SHARING = {"L1D": "private", "L2C": "private", "LLC": "shared"}
# How a level is read. The first level reads every way in parallel with the tag; the
# lower levels read the tag first and then one way, which is why ways cost them little.
ACCESS_MODE = {"L1D": "normal", "L2C": "sequential", "LLC": "sequential"}
# CACTI characterises a level as one of its own; the first two are array-like, the last
# is the outer level with its longer wires.
CACTI_LEVEL = {"L1D": "L2", "L2C": "L2", "LLC": "L3"}
BLOCK_BYTES = 64


# ---------------------------------------------------------------- the configuration ----

_base_cache = {}


def base_config():
    """ChampSim's stock configuration with this SoC's overrides applied and no design
    knobs: the chip before any design is chosen."""
    if "config" not in _base_cache:
        with open(BASE_CONFIG) as base_file:
            _base_cache["config"] = apply_profile(json.load(base_file))
    return _base_cache["config"]


def apply_profile(config):
    """Overwrite the stock config's fields with the chip's overrides, and put every clock
    in the SoC's own domain. ChampSim takes the maximum frequency down each path, so the
    stock LLC's 4 GHz would outrun a 2 GHz core unless it is overwritten here."""
    for section_name in socs.PROFILE:
        if section_name == "num_cores":
            # ChampSim duplicates the first core and its private caches to this count.
            config["num_cores"] = socs.PROFILE[section_name]
            continue
        # ChampSim stores cores as a list; the first entry is the template.
        target = config["ooo_cpu"][0] if section_name == "ooo_cpu" else config[section_name]
        for field_name in socs.PROFILE[section_name]:
            target[field_name] = socs.PROFILE[section_name][field_name]
    config["ooo_cpu"][0]["frequency"] = socs.FREQUENCY_MHZ
    config["LLC"]["frequency"] = socs.FREQUENCY_MHZ
    return config


def revision():
    """A short token for everything about this chip that is not a design knob: the
    profile, the clock and the node. It goes in every design's name, so a design
    measured on one revision of a chip is never read back for another - the result
    tables and the compiled binaries are keyed by that name and would otherwise be
    served for a chip whose latencies have changed."""
    if "revision" not in _base_cache:
        text = json.dumps([socs.CHIP, socs.PROFILE, socs.FREQUENCY_MHZ, socs.PROCESS_NM,
                           socs.CORES, socs.AREA_BUDGET_KB], sort_keys=True)
        digest = 0
        for character in text:
            digest = (digest * 131 + ord(character)) & 0xFFFFFFFF
        _base_cache["revision"] = "{:08x}".format(digest)[:4]
    return _base_cache["revision"]


NAME = socs.CHIP + "-" + revision()


# ---------------------------------------------------------------- the card ----

def card():
    """The chip as numbers. Everything a prompt says about the chip comes from here."""
    if "card" in _base_cache:
        return _base_cache["card"]
    config = base_config()
    core = config["ooo_cpu"][0]
    memory = config["physical_memory"]
    gigahertz = socs.FREQUENCY_MHZ / 1000.0
    # A DRAM controller cycle is half a transfer: its timings are counted in those.
    controller_ns = 1000.0 / (memory["data_rate"] / 2.0)
    facts = {
        "cores": socs.CORES,
        "frequency_ghz": gigahertz,
        "process_nm": socs.PROCESS_NM,
        "issue_width": core["execute_width"],
        "fetch_width": core["fetch_width"],
        "rob": core["rob_size"],
        "lq": core["lq_size"],
        "sq": core["sq_size"],
        "channels": memory["channels"],
        "data_rate": memory["data_rate"],
        "channel_width": memory["channel_width"],
        # What the channels can deliver, in the cycles every other number is counted in.
        "dram_peak_bytes_per_cycle":
            memory["channels"] * memory["data_rate"] * memory["channel_width"] / float(socs.FREQUENCY_MHZ),
        "dram_row_miss_cycles":
            (memory["tRP"] + memory["tRCD"] + memory["tCAS"]) * controller_ns * gigahertz,
        "dram_row_hit_cycles": memory["tCAS"] * controller_ns * gigahertz,
        "area_budget_kb": socs.AREA_BUDGET_KB,
        "sharing": SHARING,
    }
    facts["dram_peak_per_core"] = facts["dram_peak_bytes_per_cycle"] / socs.CORES
    _base_cache["card"] = facts
    return facts


def instances(level):
    """How many of this level the chip has: one per core when it is private."""
    return socs.CORES if SHARING[level] == "private" else 1


def chip_text_diff():
    """The chip as it was described before it was derived: the handful of fields this SoC
    overrides on ChampSim's stock core and memory, which is a diff against a configuration
    the reader never sees. Kept only as the CHIP_VIEW ablation's chip line."""
    knob_fields = ["sets", "ways", "prefetcher", "replacement", "mshr_size"]
    parts = []
    for section in sorted(socs.PROFILE):
        if section == "num_cores":
            continue
        fields = ["{} {}".format(field, socs.PROFILE[section][field])
                  for field in sorted(socs.PROFILE[section])
                  if not (section in SHARING and field in knob_fields)]
        if fields:
            parts.append("{}: {}".format(section, ", ".join(fields)))
    return "Chip {} ({}; area budget {} KB for L2 + LLC): {}".format(
        socs.CHIP, "one core" if socs.CORES == 1 else
        "{} cores, private L1 and L2 each, one shared LLC".format(socs.CORES),
        socs.AREA_BUDGET_KB, "; ".join(parts))


def chip_text():
    """The chip in four lines, for the prompts. Absolute numbers, no chip name, no
    comparison against a configuration the reader cannot see."""
    facts = card()
    private = [level for level in SHARING if SHARING[level] == "private"]
    shared = [level for level in SHARING if SHARING[level] == "shared"]
    return "\n".join([
        "Core: {}-wide issue ({}-wide fetch), ROB {}, LQ {} / SQ {}, {:.1f} GHz, {} core{}, {} nm.".format(
            facts["issue_width"], facts["fetch_width"], facts["rob"], facts["lq"], facts["sq"],
            facts["frequency_ghz"], facts["cores"], "" if facts["cores"] == 1 else "s", facts["process_nm"]),
        "Memory: {} channel{} x {} MT/s x {} B = {:.1f} B per core-cycle{}; a DRAM service costs "
        "~{:.0f} cycles on a row miss, ~{:.0f} on a row hit.".format(
            facts["channels"], "" if facts["channels"] == 1 else "s", facts["data_rate"],
            facts["channel_width"], facts["dram_peak_bytes_per_cycle"],
            "" if facts["cores"] == 1 else " shared by the cores, {:.1f} each".format(facts["dram_peak_per_core"]),
            facts["dram_row_miss_cycles"], facts["dram_row_hit_cycles"]),
        "Levels: {}, one of each.".format(", ".join(private + shared)) if facts["cores"] == 1 else
        "Levels: {} private, one per core; {} shared by all {} cores.".format(
            " and ".join(private), ", ".join(shared), facts["cores"]),
        "Area: {} KB of L2 + LLC data, a private level counted once per core.".format(facts["area_budget_kb"]),
    ])


# ---------------------------------------------------------------- the caches, from CACTI ----

_ACCESS_TIME = re.compile(r"Access time \(ns\):\s*([\d.eE+\-]+)")
_DIMENSIONS = re.compile(r"Cache height x width \(mm\):\s*([\d.eE+\-]+)\s*x\s*([\d.eE+\-]+)")
_ladder = {}


def _cacti_config(size_bytes, ways, mode, cacti_level):
    """The CACTI 7 config for one cache: this SoC's node, a 64-byte block, a tag array
    and the level's access mode. The template supplies everything else."""
    with open(CACTI_TEMPLATE) as template_file:
        lines = []
        for line in template_file.read().splitlines():
            stripped = line.strip()
            if stripped.startswith("-size (bytes)"):
                line = "-size (bytes) {}".format(size_bytes)
            elif stripped.startswith("-block size (bytes)"):
                line = "-block size (bytes) {}".format(BLOCK_BYTES)
            elif stripped.startswith("-associativity"):
                line = "-associativity {}".format(ways)
            elif stripped.startswith("-technology (u)"):
                line = "-technology (u) {:.3f}".format(socs.PROCESS_NM / 1000.0)
            elif stripped.startswith("-access mode"):
                line = '-access mode (normal, sequential, fast) - "{}"'.format(mode)
            elif stripped.startswith("-Cache level"):
                line = '-Cache level (L2/L3) - "{}"'.format(cacti_level)
            elif stripped.startswith("-Add ECC"):
                line = '-Add ECC - "false"'
            lines.append(line)
    return "\n".join(lines) + "\n"


def _run_cacti(size_bytes, ways, mode, cacti_level):
    """(access time in ns, area in mm2) for one cache, or None if CACTI cannot build it."""
    work_dir = tempfile.mkdtemp(prefix="cacti_")
    config_path = os.path.join(work_dir, "cache.cfg")
    with open(config_path, "w") as config_file:
        config_file.write(_cacti_config(size_bytes, ways, mode, cacti_level))
    try:
        finished = subprocess.run([os.path.abspath(CACTI_BIN), "-infile", config_path],
                                  cwd=os.path.dirname(os.path.abspath(CACTI_BIN)),
                                  capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None
    access = _ACCESS_TIME.search(finished.stdout)
    dimensions = _DIMENSIONS.search(finished.stdout)
    if access is None or dimensions is None:
        return None
    return float(access.group(1)), float(dimensions.group(1)) * float(dimensions.group(2))


def _characterise(level, sets, ways):
    """One level's (access time in ns, area in mm2) at this SoC's node.

    CACTI takes only power-of-two associativity, so a width it refuses is interpolated on
    log ways between the two powers of two that bracket it at the same number of sets."""
    sets, ways = int(sets), int(ways)
    mode, cacti_level = ACCESS_MODE[level], CACTI_LEVEL[level]
    if ways & (ways - 1) == 0:
        return _run_cacti(sets * ways * BLOCK_BYTES, ways, mode, cacti_level)
    lower = 1 << int(math.floor(math.log2(ways)))
    upper = lower * 2
    below = _run_cacti(sets * lower * BLOCK_BYTES, lower, mode, cacti_level)
    above = _run_cacti(sets * upper * BLOCK_BYTES, upper, mode, cacti_level)
    if below is None or above is None:
        return None
    share = (math.log(ways) - math.log(lower)) / (math.log(upper) - math.log(lower))
    return tuple(low + share * (high - low) for low, high in zip(below, above))


def _cacti_cache_path():
    os.makedirs("results", exist_ok=True)
    return CACTI_CACHE.format(socs.PROCESS_NM)


def _characterised(level, sets, ways):
    """The cached characterisation, computing and storing it the first time. Keyed by the
    node, so every SoC at one node shares the ladder and no SoC reads another's."""
    key = "{}:{}:{}:{}".format(level, int(sets), int(ways), ACCESS_MODE[level])
    if key in _ladder:
        return _ladder[key]
    path = _cacti_cache_path()
    if os.path.exists(path):
        try:
            with open(path) as cache_file:
                _ladder.update(json.load(cache_file))
        except (json.JSONDecodeError, ValueError):
            pass
        if key in _ladder:
            return _ladder[key]
    result = _characterise(level, sets, ways)
    if result is None:
        raise RuntimeError("CACTI could not characterise {} {}x{} at {} nm".format(
            level, sets, ways, socs.PROCESS_NM))
    _ladder[key] = list(result)
    # Several processes characterise at once on the first run of a chip: merge under a
    # lock rather than overwrite, so no one's entries are lost.
    with open(path + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        stored = {}
        if os.path.exists(path):
            try:
                with open(path) as cache_file:
                    stored = json.load(cache_file)
            except (json.JSONDecodeError, ValueError):
                stored = {}
        stored.update(_ladder)
        temporary = "{}.tmp.{}".format(path, os.getpid())
        with open(temporary, "w") as cache_file:
            json.dump(stored, cache_file, indent=1, sort_keys=True)
        os.replace(temporary, path)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
    return _ladder[key]


def latency(level, sets, ways):
    """A level's hit latency in this SoC's cycles: CACTI's access time at this node,
    rounded up to a whole cycle of this SoC's clock. A slower clock buys the same
    capacity for fewer cycles, which is the whole of why two chips want different
    geometry."""
    access_ns = _characterised(level, sets, ways)[0]
    return max(1, int(math.ceil(access_ns * socs.FREQUENCY_MHZ / 1000.0)))


def area_mm2(level, sets, ways):
    """The silicon one level costs, every instance of it counted."""
    return _characterised(level, sets, ways)[1] * instances(level)
