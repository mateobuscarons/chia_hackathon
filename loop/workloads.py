"""Workloads: screen them for free, probe them, sample them, fetch their traces.

  python -m loop.workloads admit
      The admission gate over every cached profile (results/profile_*.json), no
      simulator. CAPACITY channel: `movable`, the misses per kilo-instruction a
      bigger cache can remove between the smallest and the largest LLC the space
      can buy (validated against measured variance shares on this chip). POLICY
      channel: a miss floor, enough misses for a prefetcher or a replacement
      policy to matter, WITHOUT claiming which. `unfixable` is the compulsory
      MPKI: first touches, which no design removes. Stride regularity is
      reported, not used: mcf is 0.157 and its L2 prefetcher still owns 0.46 of
      its variance, because SPP learns repeated irregular paths.
  python -m loop.workloads headroom <trace> [trace ...]
      From the cached tables (after `probe`): per workload, and for every
      3-subset, the headroom over the stock chip and the share of it that no
      single-knob design reaches. A set where one knob takes the headroom ties
      every optimizer; the share beyond one knob is what a search can show.
  python -m loop.workloads probe <trace> [trace ...]
      The 11-design headroom probe for a candidate test workload: the stock chip,
      the four L2 prefetchers, srrip and ship at the LLC, the LLC doubled, the L2
      doubled, and the two best composites the graph searches found.
  python -m loop.workloads uniform <how_many> <trace> [trace ...]
      A uniform random sample of the feasible designs (seed 0): a cell's reference.
  python -m loop.workloads fetch <url> <out_path> [prefix_mb]
      Download a trace, or only its first prefix_mb megabytes (an HTTP range): a
      100 MB prefix holds ~88M instructions, enough to profile and to simulate.
  python -m loop.workloads fetch_gap <zip_url> <member> <out_dir>
      One GAP trace out of a 10 GB Zenodo zip without downloading the zip.
  python -m loop.workloads merge <other_results_dir>
      Union tables simulated on another machine into results/, by design name.

Env: PARALLEL_TRACES (traces at once, default 3), SIM_THREADS per trace,
LOOP_WARMUP / LOOP_SIM (instructions; default 5M / 10M).
"""

import fcntl
import glob
import itertools
import json
import math
import os
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

from loop import champsim_problem, trace_profile
from loop.configs import knobs_changed, random_feasible_designs

PROBE_COMPOSITE_A = {"l1d_prefetcher": "next_line", "l2_sets": 512, "l2_ways": 16, "l2_prefetcher": "va_ampm_lite",
                     "llc_sets": 4096, "llc_replacement": "ship", "llc_mshr": 64}
PROBE_COMPOSITE_B = {"l1d_sets": 128, "l1d_ways": 8, "l1d_prefetcher": "next_line", "l2_sets": 256, "l2_ways": 4,
                     "l2_prefetcher": "va_ampm_lite", "llc_sets": 8192, "llc_ways": 8, "llc_replacement": "srrip"}


def probe_designs():
    stock = champsim_problem.stock_design()
    designs = [dict(stock)]
    for prefetcher in ["spp_dev", "ip_stride", "next_line", "va_ampm_lite"]:
        design = dict(stock)
        design["l2_prefetcher"] = prefetcher
        designs.append(design)
    for policy in ["srrip", "ship"]:
        design = dict(stock)
        design["llc_replacement"] = policy
        designs.append(design)
    design = dict(stock)
    design["llc_sets"] = 4096
    designs.append(design)
    design = dict(stock)
    design["l2_sets"] = 2048
    designs.append(design)
    for composite in [PROBE_COMPOSITE_A, PROBE_COMPOSITE_B]:
        design = dict(stock)
        design.update(composite)
        designs.append(design)
    return designs


def collect_trace(trace_path, designs):
    holder = champsim_problem.ChampSimProblem(trace_path, allow_simulation=True, dispatch="local")
    missing = 0
    for knobs in designs:
        if champsim_problem.config_name(knobs) not in holder.sweep_table:
            missing += 1
    started = time.time()
    print("{}: {} designs, {} to simulate -> {}".format(holder.trace_name, len(designs), missing, holder.table_path), flush=True)
    holder.evaluate_many(designs)
    print("{}: done in {:.0f} min".format(holder.trace_name, (time.time() - started) / 60), flush=True)


def collect(trace_paths, designs):
    pool = ThreadPoolExecutor(max_workers=int(os.environ.get("PARALLEL_TRACES", "3")))
    futures = []
    for trace_path in trace_paths:
        futures.append(pool.submit(collect_trace, trace_path, designs))
    for future in futures:
        future.result()


def merge(other_results_dir):
    """Union every table found in `other_results_dir` into results/. Holds the
    table's lock while it reads and rewrites, so a collection running at the same
    time cannot have its rows dropped."""
    for other_path in sorted(glob.glob(os.path.join(other_results_dir, "table_*.json"))):
        own_path = os.path.join("results", os.path.basename(other_path))
        other = champsim_problem.load_table(other_path)
        with open(own_path + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            own = champsim_problem.load_table(own_path)
            added = 0
            for name in other:
                if name not in own and other[name]["metrics"] is not None:
                    own[name] = other[name]
                    added += 1
            champsim_problem.save_table(own, own_path)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        print("{}: +{} rows -> {}".format(os.path.basename(other_path), added, len(own)), flush=True)


def fetch(url, out_path, prefix_mb=None):
    """Download a file, or only its first prefix_mb megabytes with an HTTP range."""
    request = urllib.request.Request(url)
    if prefix_mb is not None:
        request.add_header("Range", "bytes=0-{}".format(prefix_mb * 1024 * 1024 - 1))
    with urllib.request.urlopen(request) as response, open(out_path, "wb") as out_file:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if len(chunk) == 0:
                break
            out_file.write(chunk)
    print("fetched {} -> {} ({} MB)".format(url, out_path, os.path.getsize(out_path) // (1024 * 1024)), flush=True)


def fetch_gap(zip_url, member_name, out_dir):
    """Read the zip64 central directory from the archive's tail, find the member,
    then byte-range just that member and inflate it."""
    def fetch_range(start, end):
        request = urllib.request.Request(zip_url, headers={"Range": "bytes={}-{}".format(start, end)})
        with urllib.request.urlopen(request) as response:
            return response.read()

    with urllib.request.urlopen(urllib.request.Request(zip_url, method="HEAD")) as response:
        total = int(response.headers["Content-Length"])
    tail = fetch_range(total - 256 * 1024, total - 1)
    tail_start = total - len(tail)
    # zip64 end-of-central-directory locator: where the central directory starts.
    locator = tail.rfind(b"PK\x06\x07")
    if locator >= 0:
        end_record_offset = struct.unpack("<Q", tail[locator + 8:locator + 16])[0]
        end_record = fetch_range(end_record_offset, end_record_offset + 56)
        directory_size = struct.unpack("<Q", end_record[40:48])[0]
        directory_offset = struct.unpack("<Q", end_record[48:56])[0]
    else:
        end = tail.rfind(b"PK\x05\x06")
        directory_size = struct.unpack("<I", tail[end + 12:end + 16])[0]
        directory_offset = struct.unpack("<I", tail[end + 16:end + 20])[0]
    if directory_offset >= tail_start:
        directory = tail[directory_offset - tail_start:directory_offset - tail_start + directory_size]
    else:
        directory = fetch_range(directory_offset, directory_offset + directory_size - 1)
    position = 0
    while position < len(directory):
        if directory[position:position + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", directory[position + 10:position + 12])[0]
        compressed_size = struct.unpack("<I", directory[position + 20:position + 24])[0]
        name_length = struct.unpack("<H", directory[position + 28:position + 30])[0]
        extra_length = struct.unpack("<H", directory[position + 30:position + 32])[0]
        comment_length = struct.unpack("<H", directory[position + 32:position + 34])[0]
        local_offset = struct.unpack("<I", directory[position + 42:position + 46])[0]
        name = directory[position + 46:position + 46 + name_length].decode()
        extra = directory[position + 46 + name_length:position + 46 + name_length + extra_length]
        # zip64 extra field: 64-bit sizes and offset where the 32-bit fields are saturated.
        extra_position = 0
        while extra_position + 4 <= len(extra):
            field_id, field_size = struct.unpack("<HH", extra[extra_position:extra_position + 4])
            if field_id == 1:
                field = extra[extra_position + 4:extra_position + 4 + field_size]
                values = []
                for index in range(0, len(field) - 7, 8):
                    values.append(struct.unpack("<Q", field[index:index + 8])[0])
                cursor = 0
                if struct.unpack("<I", directory[position + 24:position + 28])[0] == 0xFFFFFFFF:
                    cursor += 1
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = values[cursor]
                    cursor += 1
                if local_offset == 0xFFFFFFFF:
                    local_offset = values[cursor]
            extra_position += 4 + field_size
        position += 46 + name_length + extra_length + comment_length
        if os.path.basename(name) != member_name:
            continue
        header = fetch_range(local_offset, local_offset + 29)
        header_name_length = struct.unpack("<H", header[26:28])[0]
        header_extra_length = struct.unpack("<H", header[28:30])[0]
        data_start = local_offset + 30 + header_name_length + header_extra_length
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, member_name)
        print("fetching {} ({} MB) ...".format(member_name, compressed_size // (1024 * 1024)), flush=True)
        data = fetch_range(data_start, data_start + compressed_size - 1)
        if method == 8:
            data = zlib.decompress(data, -15)
        with open(out_path, "wb") as out_file:
            out_file.write(data)
        print("saved", out_path, flush=True)
        return out_path
    raise RuntimeError("member not found: " + member_name)


# ---------------------------------------------------------------- the free screens ----

BUYABLE_LLC_SMALLEST_KB = 256 * 4 * 64 / 1024.0 + 1024 * 8 * 64 / 1024.0
BUYABLE_LLC_LARGEST_KB = 4608.0
MOVABLE_MPKI_FLOOR = 1.0
POLICY_MPKI_FLOOR = 5.0
# Measured variance shares on this chip (dense 81-design sweeps): the ground truth the capacity channel was validated on.
CAPACITY_VARIANCE = {"605.mcf_s-665B": 0.46, "619.lbm_s-2676B": 0.01, "620.omnetpp_s-874B": 0.86}
PREFETCHER_VARIANCE = {"605.mcf_s-665B": 0.46, "619.lbm_s-2676B": 0.96, "620.omnetpp_s-874B": 0.00}


def admit():
    print("== admission gate: the chip can buy an LLC read at {:.0f}-{:.0f} KB".format(BUYABLE_LLC_SMALLEST_KB, BUYABLE_LLC_LARGEST_KB))
    print("  {:<40s} {:>9s} {:>8s} {:>8s} {:>9s} {:>7s} {:>7s} {:>7s}  {}".format(
        "trace", "footprint", "MPKI", "movable", "unfixable", "stride", "cap.var", "pf.var", "admit as"))
    for path in sorted(glob.glob("results/profile_*.json")):
        with open(path) as profile_file:
            profile = json.load(profile_file)
        name = os.path.basename(path).replace("profile_", "").replace(".json", "")
        accesses = profile["mem_accesses_per_kinstr"]
        smallest = trace_profile.predicted_miss_ratio(profile, BUYABLE_LLC_SMALLEST_KB)
        largest = trace_profile.predicted_miss_ratio(profile, BUYABLE_LLC_LARGEST_KB)
        movable = accesses * (smallest - largest)
        curve = profile["miss_ratio_curve"]
        floor = min(curve[size] for size in curve)
        channels = []
        if movable >= MOVABLE_MPKI_FLOOR:
            channels.append("capacity")
        if accesses * smallest >= POLICY_MPKI_FLOOR:
            channels.append("policy")
        verdict = " + ".join(channels)
        if len(channels) == 0:
            verdict = "REJECT (nothing any knob can fix)"
        print("  {:<40s} {:7.0f}KB {:8.1f} {:8.2f} {:9.1f} {:7.3f} {:>7s} {:>7s}  {}".format(
            name[:40], profile["footprint_kb"], accesses * smallest, movable, accesses * floor,
            profile["stride_regular_fraction"], known(CAPACITY_VARIANCE, name), known(PREFETCHER_VARIANCE, name), verdict))


def known(table, name):
    if name in table:
        return "{:.2f}".format(table[name])
    return "-"


def headroom(traces):
    stock = champsim_problem.stock_design()
    rows_by_trace = {}
    for trace in traces:
        name = trace_profile.short_name(trace)
        table = champsim_problem.load_table(champsim_problem.table_path(name))
        rows = {}
        for row in champsim_problem.measured_rows(table):
            rows[row["name"]] = (row["ipc"], len(knobs_changed(row["knobs"], stock)))
        rows_by_trace[name] = rows
    names = list(rows_by_trace.keys())
    print("== headroom per workload (stock, best measured, best single-knob design, share of headroom beyond one knob)")
    for name in names:
        print("  " + headroom_line(name, [name], rows_by_trace))
    if len(names) >= 3:
        print()
        print("== every 3-subset, geomean objective, best share beyond one knob first")
        lines = []
        for subset in itertools.combinations(names, 3):
            lines.append(headroom_line(" + ".join(subset), list(subset), rows_by_trace))
        lines.sort(key=lambda line: -float(line.split("beyond one knob")[1].strip().rstrip("%")))
        for line in lines:
            print("  " + line)


def headroom_line(label, subset, rows_by_trace):
    common = set(rows_by_trace[subset[0]].keys())
    for name in subset[1:]:
        common = common & set(rows_by_trace[name].keys())
    stock_value = None
    best_value = 0.0
    best_single = 0.0
    for design in common:
        log_sum = 0.0
        for name in subset:
            log_sum += math.log(rows_by_trace[name][design][0])
        value = math.exp(log_sum / len(subset))
        changed = rows_by_trace[subset[0]][design][1]
        if changed == 0:
            stock_value = value
        if value > best_value:
            best_value = value
        if changed == 1 and value > best_single:
            best_single = value
    if stock_value is None:
        return "{:<50s} no stock row".format(label[:50])
    headroom_pct = 100.0 * (best_value / stock_value - 1.0)
    single_pct = 100.0 * (best_single / stock_value - 1.0)
    beyond = 0.0
    if headroom_pct > 0:
        beyond = 100.0 * (1.0 - max(single_pct, 0.0) / headroom_pct)
    return "{:<50s} {:4d} designs | stock {:.4f} best {:.4f} | headroom {:+6.1f}% | one knob {:+6.1f}% | beyond one knob {:5.1f}%".format(
        label[:50], len(common), stock_value, best_value, headroom_pct, single_pct, beyond)


def main():
    mode = sys.argv[1]
    if mode == "merge":
        merge(sys.argv[2])
    elif mode == "fetch":
        prefix_mb = None
        if len(sys.argv) > 4:
            prefix_mb = int(sys.argv[4])
        fetch(sys.argv[2], sys.argv[3], prefix_mb)
    elif mode == "fetch_gap":
        fetch_gap(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "probe":
        collect(sys.argv[2:], probe_designs())
    elif mode == "uniform":
        collect(sys.argv[3:], random_feasible_designs(int(sys.argv[2]), seed=0))
    elif mode == "admit":
        admit()
    elif mode == "headroom":
        headroom(sys.argv[2:])
    else:
        raise SystemExit("unknown mode " + mode)


if __name__ == "__main__":
    main()
