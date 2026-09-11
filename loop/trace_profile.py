"""Workload profile from the trace alone: chip-independent descriptors.

A ChampSim trace is the recorded address stream of a program. Everything here
is computed from that stream, never from a simulated chip, so the numbers
describe the PROGRAM and are the same for every chip. The chip-relative half of
every descriptor is derived from these plus the chip's cache geometry in
champsim_problem.chip_descriptors.

Capacity behaviour comes from footprint theory (Xiang, Bao, Ding, Gao, "HOTL:
a higher order theory of locality", ASPLOS 2013): from the reuse-time
histogram alone one gets the exact average footprint fp(w) of every window
length w, and the LRU miss ratio at cache size c is the slope of fp at the
window whose footprint is c. No cache simulation is needed.

Usage: python -m loop.trace_profile <trace.xz|trace.gz> [warmup_instructions] [instructions]
Writes results/profile_<trace>.json and prints it.
"""

import json
import os
import subprocess
import sys

import numpy

# ChampSim input_instr: 64 bytes, little endian (see champsim/inc/trace_instruction.h).
RECORD = numpy.dtype([("ip", "<u8"), ("is_branch", "u1"), ("branch_taken", "u1"),
                      ("dst_regs", "u1", (2,)), ("src_regs", "u1", (4,)),
                      ("dst_mem", "<u8", (2,)), ("src_mem", "<u8", (4,))])
RECORD_BYTES = 64
BLOCK_BYTES = 64
CHUNK_INSTRUCTIONS = 1_000_000

# Descriptors that depend on the program only (the chip-relative ones are derived
# in champsim_problem from these plus the chip's cache sizes).
PROFILE_METRICS = ["mem_accesses_per_kinstr", "write_fraction", "footprint_kb",
                   "stride_regular_fraction", "reuse_local_fraction"]
REUSE_LOCAL_ACCESSES = 1024          # "reused soon": within this many memory accesses
CAPACITY_GRID_KB = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def short_name(trace_path):
    """The trace's name without directory or compression suffix: "605.mcf_s-665B",
    "bfs.urand-36B", "merced_0000"."""
    name = os.path.basename(trace_path)
    for suffix in [".champsimtrace.xz", ".champsimtrace.gz", ".champsim.gz", ".champsim.xz", ".xz", ".gz"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def profile_path(trace_path):
    return "results/profile_{}.json".format(short_name(trace_path))


def read_accesses(trace_path, skip_instructions, count_instructions):
    """Decompress the trace and return the memory-access stream of the measured
    window: block address, PC and is-write per access, in program order."""
    if trace_path.endswith(".gz"):
        command = ["gzip", "-dc", trace_path]
    else:
        command = ["xz", "-dc", trace_path]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=16 * 1024 * 1024)
    stream = process.stdout

    skipped = 0
    while skipped < skip_instructions * RECORD_BYTES:
        wanted = min(CHUNK_INSTRUCTIONS * RECORD_BYTES, skip_instructions * RECORD_BYTES - skipped)
        data = stream.read(wanted)
        if len(data) == 0:
            break
        skipped += len(data)

    blocks = []
    addresses_all = []
    pcs = []
    writes = []
    read_instructions = 0
    while read_instructions < count_instructions:
        wanted = min(CHUNK_INSTRUCTIONS, count_instructions - read_instructions)
        data = stream.read(wanted * RECORD_BYTES)
        if len(data) < RECORD_BYTES:
            break
        usable = (len(data) // RECORD_BYTES) * RECORD_BYTES
        records = numpy.frombuffer(data[:usable], dtype=RECORD)
        read_instructions += len(records)
        # Up to 4 loads and 2 stores per instruction; zero means "no access".
        # Order within an instruction: loads first, then stores (program order
        # between instructions is what matters for reuse).
        for column in range(4):
            addresses = records["src_mem"][:, column]
            mask = addresses != 0
            blocks.append(addresses[mask] >> 6)
            addresses_all.append(addresses[mask])
            pcs.append(records["ip"][mask])
            writes.append(numpy.zeros(int(mask.sum()), dtype=bool))
        for column in range(2):
            addresses = records["dst_mem"][:, column]
            mask = addresses != 0
            blocks.append(addresses[mask] >> 6)
            addresses_all.append(addresses[mask])
            pcs.append(records["ip"][mask])
            writes.append(numpy.ones(int(mask.sum()), dtype=bool))
        # Re-interleave this chunk into program order: sort by instruction index.
        chunk_index = []
        for column in range(4):
            chunk_index.append(numpy.nonzero(records["src_mem"][:, column] != 0)[0])
        for column in range(2):
            chunk_index.append(numpy.nonzero(records["dst_mem"][:, column] != 0)[0])
        order = numpy.argsort(numpy.concatenate(chunk_index), kind="stable")
        parts = len(chunk_index)
        chunk_blocks = numpy.concatenate(blocks[-parts:])[order]
        chunk_addresses = numpy.concatenate(addresses_all[-parts:])[order]
        chunk_pcs = numpy.concatenate(pcs[-parts:])[order]
        chunk_writes = numpy.concatenate(writes[-parts:])[order]
        del blocks[-parts:]
        del addresses_all[-parts:]
        del pcs[-parts:]
        del writes[-parts:]
        blocks.append(chunk_blocks)
        addresses_all.append(chunk_addresses)
        pcs.append(chunk_pcs)
        writes.append(chunk_writes)
    process.stdout.close()
    process.terminate()
    process.wait()
    return (numpy.concatenate(blocks), numpy.concatenate(addresses_all), numpy.concatenate(pcs),
            numpy.concatenate(writes), read_instructions)


def reuse_times(blocks):
    """For every access, the number of accesses since the previous access to the
    same block (0 for a first access). Also first-access and last-access times
    per block, as footprint theory needs them."""
    n = len(blocks)
    positions = numpy.arange(n, dtype=numpy.int64)
    order = numpy.argsort(blocks, kind="stable")           # by block, then by time
    sorted_blocks = blocks[order]
    sorted_positions = positions[order]
    same_as_previous = numpy.empty(n, dtype=bool)
    same_as_previous[0] = False
    same_as_previous[1:] = sorted_blocks[1:] == sorted_blocks[:-1]
    reuse = numpy.zeros(n, dtype=numpy.int64)
    reuse[same_as_previous] = sorted_positions[same_as_previous] - sorted_positions[numpy.nonzero(same_as_previous)[0] - 1]
    # first access time (1-based) of each distinct block, and "time to end" of its last access
    first_mask = ~same_as_previous
    first_times = sorted_positions[first_mask] + 1
    last_mask = numpy.empty(n, dtype=bool)
    last_mask[:-1] = ~same_as_previous[1:]
    last_mask[-1] = True
    last_times = n - sorted_positions[last_mask]
    return reuse[same_as_previous], first_times, last_times


def footprint_curve(reuse, first_times, last_times, n, distinct, windows):
    """Xiang's exact average footprint for each window length w:
    fp(w) = m - [ sum_{t>w} (t-w) rt(t) + sum_i (f_i-w)[f_i>w] + sum_i (l_i-w)[l_i>w] ] / (n-w+1)."""
    reuse_sorted = numpy.sort(reuse)
    reuse_cumsum = numpy.concatenate([[0], numpy.cumsum(reuse_sorted, dtype=numpy.float64)])
    first_sorted = numpy.sort(first_times)
    first_cumsum = numpy.concatenate([[0], numpy.cumsum(first_sorted, dtype=numpy.float64)])
    last_sorted = numpy.sort(last_times)
    last_cumsum = numpy.concatenate([[0], numpy.cumsum(last_sorted, dtype=numpy.float64)])

    def tail_excess(sorted_values, cumsum, w):
        # sum over values v > w of (v - w)
        start = numpy.searchsorted(sorted_values, w, side="right")
        count = len(sorted_values) - start
        total = cumsum[-1] - cumsum[start]
        return total - w * count

    footprints = []
    for w in windows:
        excess = tail_excess(reuse_sorted, reuse_cumsum, w)
        excess += tail_excess(first_sorted, first_cumsum, w)
        excess += tail_excess(last_sorted, last_cumsum, w)
        footprints.append(distinct - excess / (n - w + 1))
    return numpy.array(footprints)


def miss_ratio_curve(reuse, first_times, last_times, n, distinct, capacities_blocks):
    """LRU miss ratio at each capacity (in blocks): the slope of fp(w) at the
    window whose footprint equals the capacity."""
    windows = numpy.unique(numpy.geomspace(1, max(n - 1, 2), 400).astype(numpy.int64))
    footprints = footprint_curve(reuse, first_times, last_times, n, distinct, windows)
    slopes = numpy.gradient(footprints, windows)
    ratios = []
    for capacity in capacities_blocks:
        if capacity >= footprints[-1]:
            ratios.append(float(distinct) / n)          # everything fits: only cold misses
            continue
        index = int(numpy.searchsorted(footprints, capacity))
        index = min(max(index, 0), len(slopes) - 1)
        ratios.append(float(max(min(slopes[index], 1.0), 0.0)))
    return ratios


def stride_regular_fraction(addresses, pcs):
    """Fraction of accesses whose BYTE-address stride from the same PC's previous
    access repeats the previous stride (what a stride/stream prefetcher catches).
    Byte addresses, not blocks: a stencil walking 8 bytes at a time is regular
    even though several accesses land in the same 64-byte block."""
    n = len(addresses)
    if n < 3:
        return 0.0
    order = numpy.argsort(pcs, kind="stable")
    sorted_pcs = pcs[order]
    sorted_blocks = addresses[order].astype(numpy.int64)
    same_pc_1 = numpy.zeros(n, dtype=bool)
    same_pc_1[1:] = sorted_pcs[1:] == sorted_pcs[:-1]
    same_pc_2 = numpy.zeros(n, dtype=bool)
    same_pc_2[2:] = sorted_pcs[2:] == sorted_pcs[:-2]
    stride_1 = numpy.zeros(n, dtype=numpy.int64)
    stride_1[1:] = sorted_blocks[1:] - sorted_blocks[:-1]
    stride_2 = numpy.zeros(n, dtype=numpy.int64)
    stride_2[2:] = sorted_blocks[1:-1] - sorted_blocks[:-2]
    regular = same_pc_1 & same_pc_2 & (stride_1 == stride_2) & (stride_1 != 0)
    return float(regular.sum()) / n


def profile_trace(trace_path, warmup_instructions=5_000_000, instructions=10_000_000):
    blocks, addresses, pcs, writes, read_instructions = read_accesses(trace_path, warmup_instructions, instructions)
    n = len(blocks)
    distinct = int(len(numpy.unique(blocks)))
    reuse, first_times, last_times = reuse_times(blocks)
    capacities_blocks = []
    for kb in CAPACITY_GRID_KB:
        capacities_blocks.append(kb * 1024 // BLOCK_BYTES)
    curve = miss_ratio_curve(reuse, first_times, last_times, n, distinct, capacities_blocks)
    profile = {
        "trace": os.path.basename(trace_path),
        "instructions": int(read_instructions),
        "mem_accesses": int(n),
        "mem_accesses_per_kinstr": 1000.0 * n / max(read_instructions, 1),
        "write_fraction": float(writes.sum()) / max(n, 1),
        "footprint_kb": distinct * BLOCK_BYTES / 1024.0,
        "stride_regular_fraction": stride_regular_fraction(addresses, pcs),
        "reuse_local_fraction": float((reuse <= REUSE_LOCAL_ACCESSES).sum()) / max(n, 1),
        "miss_ratio_curve": dict(zip([str(kb) for kb in CAPACITY_GRID_KB], curve)),
    }
    return profile


def load_or_build(trace_path, warmup_instructions=5_000_000, instructions=10_000_000):
    path = profile_path(trace_path)
    if os.path.exists(path):
        with open(path) as profile_file:
            return json.load(profile_file)
    profile = profile_trace(trace_path, warmup_instructions, instructions)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as profile_file:
        json.dump(profile, profile_file, indent=2)
    return profile


def predicted_miss_ratio(profile, capacity_kb):
    """Interpolate the profile's miss-ratio curve (log-capacity) at one size."""
    sizes = sorted(int(kb) for kb in profile["miss_ratio_curve"])
    ratios = [profile["miss_ratio_curve"][str(kb)] for kb in sizes]
    if capacity_kb <= sizes[0]:
        return ratios[0]
    if capacity_kb >= sizes[-1]:
        return ratios[-1]
    log_sizes = numpy.log2(numpy.array(sizes, dtype=float))
    return float(numpy.interp(numpy.log2(capacity_kb), log_sizes, ratios))


if __name__ == "__main__":
    trace = sys.argv[1]
    warmup = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000
    import time
    started = time.time()
    result = profile_trace(trace, warmup, count)
    result["seconds"] = round(time.time() - started, 1)
    print(json.dumps(result, indent=2))
    if warmup == 5_000_000 and count == 10_000_000:
        os.makedirs("results", exist_ok=True)
        with open(profile_path(trace), "w") as out:
            json.dump(result, out, indent=2)
        print("saved", profile_path(trace))
