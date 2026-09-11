"""Simulate a fixed list of designs per chip on the traces given, into the shared
result tables. Three uses:

  python -m loop.collect structured <how_many> <trace.xz> [more traces]
      The learn set: the baseline, every design one knob away from it, then a fixed
      random order of the rest of the chip's table. Every knob gets a controlled
      pair, so every knob can get a rule.
  python -m loop.collect top <how_many> <trace.xz> [more traces]
      The chip's best designs by the geomean over the traces given (plus the
      baseline and 5 random ones): with LOOP_WARMUP / LOOP_SIM set this is the
      fidelity check and the final validation, landing in their own tables.
  python -m loop.collect uniform <how_many> <trace.xz> [more traces]
      A uniform random sample of the chip's feasible designs (seed 0): the cell's
      reference (its best design) and the random-search null curve at once.
  python -m loop.collect fetch_gap <zip_url> <member> <out_dir>
      One GAP trace out of a 10 GB Zenodo zip without downloading the zip: read the
      zip64 central directory from the last 256 KB, then byte-range the one member.
  python -m loop.collect merge <other_results_dir>
      Union tables simulated elsewhere (the Mac and the VM both simulate) into
      results/, by design name. Identical designs give identical rows.
  python -m loop.collect add_knob <results_dir> <knob> <default>
      A knob added to the space at the value every earlier design implicitly had:
      every cached row gets it and is renamed, so nothing measured is lost.

Env: COLLECT_SOCS (comma-separated chips; default all three single-core chips),
PARALLEL_PAIRS (chip x trace pairs at once, default 4), SIM_THREADS per pair,
LOOP_WARMUP / LOOP_SIM (instructions; default 5M / 10M).
"""

import glob
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from loop import champsim_problem
from loop.configs import SEARCH_SPACE, random_feasible_designs

ALL_SOCS = ["A_mobile", "B_midrange", "C_server"]
REFERENCE_TRACE = "605.mcf_s-665B"
SPACE = "C"
RANDOM_EXTRA = 5


def short_table(soc_name, trace_path):
    """The chip's SHORT-run table for one trace (the search's cache), whatever LOOP_WARMUP says."""
    trace_name = os.path.basename(trace_path).split(".champsimtrace")[0]
    return champsim_problem.load_table("results/tierC_{}_{}.json".format(soc_name, trace_name))


# The two-knob pairs the learn set always contains: the prefetcher against the
# LLC size (does prefetching change what capacity is worth?) and the two sizes
# against each other under the area cap (private capacity against shared).
PAIRED_KNOBS = [("l2_prefetcher", "llc_sets"), ("l2_sets", "llc_sets")]


def structured_designs(soc_name, how_many):
    """The learn set: the baseline, every feasible design one knob away from it
    (so every knob value gets a controlled pair and can get a rule), the key
    two-knob pairs, then the chip's already-measured designs in a fixed random
    order, up to `how_many`."""
    baseline = champsim_problem.profiled_baseline(soc_name, SPACE)
    designs = [baseline]
    seen = set()
    seen.add(champsim_problem.config_name(baseline, soc_name))

    def add(knobs):
        name = champsim_problem.config_name(knobs, soc_name)
        if name in seen or not within_budget(knobs, soc_name, champsim_problem.BASE_CONFIG):
            return
        seen.add(name)
        designs.append(knobs)

    for knob in SEARCH_SPACE:
        for value in SEARCH_SPACE[knob]:
            if str(value) == str(baseline[knob]):
                continue
            knobs = dict(baseline)
            knobs[knob] = value
            add(knobs)
    for first_knob, second_knob in PAIRED_KNOBS:
        for first_value in SEARCH_SPACE[first_knob]:
            for second_value in SEARCH_SPACE[second_knob]:
                if str(first_value) == str(baseline[first_knob]) or str(second_value) == str(baseline[second_knob]):
                    continue
                knobs = dict(baseline)
                knobs[first_knob] = first_value
                knobs[second_knob] = second_value
                add(knobs)
    table = champsim_problem.load_table("results/tierC_{}_{}.json".format(soc_name, REFERENCE_TRACE))
    others = []
    for name in table:
        entry = table[name]
        if entry["metrics"] is None or "soc" in entry["knobs"]:
            continue
        others.append(entry["knobs"])
    random.Random(0).shuffle(others)
    for knobs in others:
        if len(designs) >= how_many:
            break
        add(knobs)
    # The baseline, the one-knob designs and the pairs are always in; how_many caps the rest.
    return designs


def top_designs(soc_name, how_many, trace_paths):
    """The chip's best `how_many` designs by geomean over the traces, among designs
    measured on all of them, plus the baseline and RANDOM_EXTRA random others."""
    tables = []
    for trace_path in trace_paths:
        tables.append(short_table(soc_name, trace_path))
    scored = []
    for name in tables[0]:
        log_sum = 0.0
        complete = True
        for table in tables:
            entry = table.get(name)
            if entry is None or entry["metrics"] is None:
                complete = False
                break
            log_sum += math.log(max(entry["metrics"]["ipc"], 1e-9))
        if not complete or "soc" in tables[0][name]["knobs"]:
            continue
        scored.append((math.exp(log_sum / len(tables)), name))
    scored.sort(reverse=True)
    baseline = champsim_problem.profiled_baseline(soc_name, SPACE)
    chosen = [baseline]
    for score, name in scored[:how_many]:
        chosen.append(tables[0][name]["knobs"])
    rest = []
    for score, name in scored[how_many:]:
        rest.append(tables[0][name]["knobs"])
    random.Random(0).shuffle(rest)
    chosen = chosen + rest[:RANDOM_EXTRA]
    return chosen


def uniform_designs(soc_name, how_many):
    """`how_many` feasible designs drawn uniformly at random (fixed seed) plus the chip's baseline."""
    return [champsim_problem.profiled_baseline(soc_name, SPACE)] + random_feasible_designs(soc_name, how_many, seed=0)


def add_knob(results_dir, knob, default_value):
    """A new knob whose default equals what every earlier design had: every cached
    row gets the knob at that value and is renamed, so the rows stay valid."""
    for path in sorted(glob.glob(os.path.join(results_dir, "tierC_*.json")) + glob.glob(os.path.join(results_dir, "sweep_*.json"))):
        table = champsim_problem.load_table(path)
        renamed = {}
        changed = 0
        for name, row in table.items():
            knobs = dict(row["knobs"])
            soc_name = name.split("_l1d_")[0]
            if knob not in knobs:
                knobs[knob] = default_value
                changed += 1
            renamed[champsim_problem.config_name(knobs, soc_name)] = {"knobs": knobs, "metrics": row["metrics"]}
        champsim_problem.save_table(renamed, path)
        print("{}: {} rows, {} given {}={}".format(os.path.basename(path), len(renamed), changed, knob, default_value), flush=True)


def collect_pair(soc_name, trace_path, designs):
    problem = champsim_problem.make_problem(soc_name, trace_path, allow_simulation=True, space_name=SPACE)
    holder = problem["holder"]
    missing = 0
    for knobs in designs:
        name = champsim_problem.config_name(knobs, soc_name)
        if name not in holder.sweep_table:
            missing += 1
    started = time.time()
    print("{} {}: {} designs, {} to simulate -> {}".format(
        soc_name, os.path.basename(trace_path), len(designs), missing, holder.table_path), flush=True)
    holder.evaluate_many(designs)
    print("{} {}: done in {:.0f} min".format(soc_name, os.path.basename(trace_path),
                                             (time.time() - started) / 60), flush=True)


def chosen_socs():
    """Which chips to collect on. A headroom check only needs the target chip."""
    setting = os.environ.get("COLLECT_SOCS", "")
    if setting == "":
        return ALL_SOCS
    names = []
    for name in setting.split(","):
        names.append(name.strip())
    return names


def merge(other_results_dir):
    """Union every tierC_*.json table found in `other_results_dir` into results/."""
    for other_path in sorted(glob.glob(os.path.join(other_results_dir, "tierC_*.json"))):
        own_path = os.path.join("results", os.path.basename(other_path))
        own = champsim_problem.load_table(own_path)
        other = champsim_problem.load_table(other_path)
        added = 0
        for name in other:
            if name not in own and other[name]["metrics"] is not None:
                own[name] = other[name]
                added += 1
        champsim_problem.save_table(own, own_path)
        print("{}: +{} rows -> {} rows".format(os.path.basename(own_path), added, len(own)))


def fetch_gap(zip_url, member_name, out_dir):
    """Download one member of a remote zip by HTTP range requests. The member is
    stored deflated inside the zip and is itself an .xz trace."""
    import struct
    import urllib.request
    import zlib

    def fetch_range(start, end):
        request = urllib.request.Request(zip_url, headers={"Range": "bytes={}-{}".format(start, end)})
        with urllib.request.urlopen(request) as response:
            return response.read()

    with urllib.request.urlopen(urllib.request.Request(zip_url, method="HEAD")) as response:
        total = int(response.headers["Content-Length"])
    tail = fetch_range(total - 256 * 1024, total - 1)
    # The zip64 end-of-central-directory record holds the directory's offset and size.
    position = tail.rfind(b"PK\x06\x06")
    if position < 0:
        raise RuntimeError("no zip64 end record in the last 256 KB")
    directory_size = struct.unpack("<Q", tail[position + 40:position + 48])[0]
    directory_offset = struct.unpack("<Q", tail[position + 48:position + 56])[0]
    directory = fetch_range(directory_offset, directory_offset + directory_size - 1)
    cursor = 0
    while cursor < len(directory):
        if directory[cursor:cursor + 4] != b"PK\x01\x02":
            break
        method = struct.unpack("<H", directory[cursor + 10:cursor + 12])[0]
        compressed_size = struct.unpack("<I", directory[cursor + 20:cursor + 24])[0]
        name_length = struct.unpack("<H", directory[cursor + 28:cursor + 30])[0]
        extra_length = struct.unpack("<H", directory[cursor + 30:cursor + 32])[0]
        comment_length = struct.unpack("<H", directory[cursor + 32:cursor + 34])[0]
        local_offset = struct.unpack("<I", directory[cursor + 42:cursor + 46])[0]
        name = directory[cursor + 46:cursor + 46 + name_length].decode()
        extra = directory[cursor + 46 + name_length:cursor + 46 + name_length + extra_length]
        # zip64 extra field: 64-bit sizes and offset for the fields stored as 0xFFFFFFFF.
        extra_cursor = 0
        while extra_cursor + 4 <= len(extra):
            field_id = struct.unpack("<H", extra[extra_cursor:extra_cursor + 2])[0]
            field_size = struct.unpack("<H", extra[extra_cursor + 2:extra_cursor + 4])[0]
            if field_id == 1:
                values = []
                data = extra[extra_cursor + 4:extra_cursor + 4 + field_size]
                for index in range(0, len(data) - 7, 8):
                    values.append(struct.unpack("<Q", data[index:index + 8])[0])
                fields = []
                uncompressed_size = struct.unpack("<I", directory[cursor + 24:cursor + 28])[0]
                if uncompressed_size == 0xFFFFFFFF:
                    fields.append("uncompressed")
                if compressed_size == 0xFFFFFFFF:
                    fields.append("compressed")
                if local_offset == 0xFFFFFFFF:
                    fields.append("offset")
                for field, value in zip(fields, values):
                    if field == "compressed":
                        compressed_size = value
                    if field == "offset":
                        local_offset = value
            extra_cursor += 4 + field_size
        cursor += 46 + name_length + extra_length + comment_length
        if name.endswith(member_name):
            print("found {} at offset {} ({:.0f} MB compressed, method {})".format(
                name, local_offset, compressed_size / 1e6, method), flush=True)
            header = fetch_range(local_offset, local_offset + 29)
            header_name_length = struct.unpack("<H", header[26:28])[0]
            header_extra_length = struct.unpack("<H", header[28:30])[0]
            data_start = local_offset + 30 + header_name_length + header_extra_length
            payload = fetch_range(data_start, data_start + compressed_size - 1)
            if method == 8:
                payload = zlib.decompress(payload, -15)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(name))
            with open(out_path, "wb") as out_file:
                out_file.write(payload)
            print("wrote {} ({:.0f} MB)".format(out_path, len(payload) / 1e6), flush=True)
            return out_path
    raise RuntimeError("member not found: " + member_name)


def main():
    mode = sys.argv[1]
    if mode == "merge":
        merge(sys.argv[2])
        return
    if mode == "fetch_gap":
        fetch_gap(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    if mode == "add_knob":
        # python -m loop.collect add_knob <results dir> <knob> <default value>
        add_knob(sys.argv[2], sys.argv[3], sys.argv[4])
        return
    how_many = int(sys.argv[2])
    trace_paths = sys.argv[3:]
    socs = chosen_socs()
    designs_by_soc = {}
    for soc_name in socs:
        if mode == "top":
            designs_by_soc[soc_name] = top_designs(soc_name, how_many, trace_paths)
        elif mode == "uniform":
            designs_by_soc[soc_name] = uniform_designs(soc_name, how_many)
        else:
            designs_by_soc[soc_name] = structured_designs(soc_name, how_many)
    pairs = []
    for trace_path in trace_paths:
        for soc_name in socs:
            pairs.append((soc_name, trace_path))
    pool = ThreadPoolExecutor(max_workers=int(os.environ.get("PARALLEL_PAIRS", "4")))
    futures = []
    for soc_name, trace_path in pairs:
        futures.append(pool.submit(collect_pair, soc_name, trace_path, designs_by_soc[soc_name]))
    for future in futures:
        future.result()


if __name__ == "__main__":
    main()
