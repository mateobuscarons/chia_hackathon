"""The memory: one design, carried from searches on other workloads.

What a memory IS, on disk, is one design and the receipt for it:

  {"chip": "C_server",
   "handover": {13 knobs},
   "chosen_from": {"designs": 163, "measured_on": 6, "mean_gap_share": 0.90,
                   "per_workload": {"605.mcf_s-665B": {"stock": .., "best": .., "share": ..}, ...}}}

`handover()` is the whole interface a search sees: 13 knob values, fitted to this
chip's area budget, which is where `pooled_bo` starts instead of the stock chip.
The receipt is for a reader, not for the code: it says how far that one design
gets on each workload it is remembered from, so "the memory is worth something"
is a claim anyone can check without rerunning anything.

Choosing it needs, per remembered workload, its stock IPC, its best IPC and every
design measured on it. All three are read back out of the shared result tables,
so none of them is stored: the tables are the dataset, this file is the answer.

Filling a memory takes two stages:

  1. SEARCH   one agent search per group of memory workloads (a group is a suite,
              e.g. the SPEC trio), SEARCH_DESIGNS designs, every design the agent
              proposes measured with nothing in between.
  2. CONFIRM  each group's best CONFIRM_TOP designs measured on EVERY memory
              workload. A design may be handed over only if it was measured on
              all but one of them, so this stage is what makes a choice possible
              at all - a design that looks good on three workloads it was searched
              on has said nothing about the other three.

  python -m loop.memory build <out.json> <traces...> -- <traces...>    # simulates

Env: MB_SEARCH (designs per group, default 40), MB_CONFIRM (top per group, default 20).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from loop import suite
from loop.suite import make_suite_problem, short_name
from loop.space import CHIP, SEARCH_SPACE, same_knobs, within_budget

SEARCH_DESIGNS = int(os.environ.get("MB_SEARCH", "40"))
# Eight per round, and both groups searching at once: a round then holds 2 x 8 x 3
# simulations, enough to fill the machine. Eight is also the agent's proposal
# count, so every design it chooses is measured.
PER_ROUND = 8
CONFIRM_TOP = int(os.environ.get("MB_CONFIRM", "20"))
CHUNK = 8                 # designs measured per progress report


# ---------------------------------------------------------------- what a search reads ----

def handover(path):
    """The design the memory hands over, fitted to this chip's area budget."""
    if not os.path.exists(path):
        raise RuntimeError("no memory at " + str(path))
    with open(path) as memory_file:
        shelf = json.load(memory_file)
    if shelf.get("handover") is None:
        raise RuntimeError("the memory holds no handover design: " + str(path))
    return fit_to_budget(shelf["handover"])


def fit_to_budget(knobs):
    """A remembered design read onto this chip's area budget: shrink the LLC one
    step at a time, then the L2, until it fits. A memory built where more area was
    affordable is still usable here, one size down."""
    fitted = dict(knobs)
    for knob in ["llc_sets", "llc_ways", "l2_sets", "l2_ways"]:
        while not within_budget(fitted):
            values = SEARCH_SPACE[knob]
            position = None
            for index, value in enumerate(values):
                if str(value) == str(fitted[knob]):
                    position = index
            if position is None or position == 0:
                break
            fitted[knob] = values[position - 1]
    return fitted


# ---------------------------------------------------------------- choosing it ----

def table_rows(workload, only_names):
    """Every measured design in one workload's result table, kept to the designs
    this build asked for. The tables are a cache shared by everything that has
    ever run on this workload, so without the filter a memory would claim designs
    it never measured."""
    table = suite.load_table(suite.table_path(workload))
    rows = suite.measured_rows(table)
    kept = []
    for row in rows:
        if row["name"] in only_names:
            kept.append(row)
    return kept


def gap_rows(trace_paths, only_names):
    """Per remembered workload: the stock IPC, the best IPC measured, and the
    rows. The gap between the first two is what a design's share is measured
    against, and it is that workload's own gap, not a shared one."""
    stock = suite.stock_design()
    per_workload = {}
    for trace_path in trace_paths:
        workload = short_name(trace_path)
        rows = table_rows(workload, only_names)
        if len(rows) == 0:
            print("[memory] nothing measured on {}, skipped".format(workload), flush=True)
            continue
        stock_ipc = None
        best_ipc = rows[0]["ipc"]
        for row in rows:
            if same_knobs(row["knobs"], stock):
                stock_ipc = row["ipc"]
            if row["ipc"] > best_ipc:
                best_ipc = row["ipc"]
        per_workload[workload] = {"stock": stock_ipc, "best": best_ipc, "rows": rows}
    return per_workload


def choose_handover(per_workload, min_workloads):
    """The design with the best mean gap share over the remembered workloads that
    measured it, counting only designs measured on at least `min_workloads` of
    them. Returns (knobs, {workload: share}) or (None, {})."""
    shares = {}
    knobs_of = {}
    for workload in per_workload:
        stock_ipc = per_workload[workload]["stock"]
        best_ipc = per_workload[workload]["best"]
        if stock_ipc is None or best_ipc <= stock_ipc:
            continue
        gap = best_ipc - stock_ipc
        for row in per_workload[workload]["rows"]:
            shares.setdefault(row["name"], {})[workload] = (row["ipc"] - stock_ipc) / gap
            knobs_of[row["name"]] = row["knobs"]
    best_name = None
    best_mean = None
    for name in shares:
        if len(shares[name]) < min_workloads:
            continue
        total = 0.0
        for workload in shares[name]:
            total += shares[name][workload]
        mean = total / len(shares[name])
        if best_mean is None or mean > best_mean:
            best_mean = mean
            best_name = name
    if best_name is None:
        return None, {}
    return knobs_of[best_name], shares[best_name]


def write_memory(trace_paths, output_path, only_names):
    """Choose the design and write the file. Writes nothing while no design has
    been measured on enough workloads to be chosen honestly, so a half-finished
    build never replaces a finished memory with an empty one."""
    per_workload = gap_rows(trace_paths, only_names)
    # Measured on all the remembered workloads but one (at least two). The guard is
    # worth four points of gap share, and it is what the confirm stage exists for.
    guard = max(2, len(per_workload) - 1)
    knobs, shares = choose_handover(per_workload, guard)
    if knobs is None:
        print("[memory] no design measured on {} of {} workloads yet, nothing written".format(
            guard, len(per_workload)), flush=True)
        return None
    measured = set()
    receipt = {}
    total = 0.0
    for workload in per_workload:
        for row in per_workload[workload]["rows"]:
            measured.add(row["name"])
        if workload not in shares:
            continue
        total += shares[workload]
        receipt[workload] = {"stock": round(per_workload[workload]["stock"], 4),
                             "best": round(per_workload[workload]["best"], 4),
                             "share": round(shares[workload], 3)}
    shelf = {"chip": CHIP, "handover": knobs,
             "chosen_from": {"designs": len(measured), "measured_on": len(shares),
                             "mean_gap_share": round(total / len(shares), 3),
                             "per_workload": receipt}}
    with open(output_path, "w") as memory_file:
        json.dump(shelf, memory_file, indent=1)
    print("[memory] hands over a design at {:.0f}% of the gap on {} of {} workloads, "
          "chosen among {} designs -> {}".format(
              100.0 * shelf["chosen_from"]["mean_gap_share"], len(shares), len(per_workload),
              len(measured), output_path), flush=True)
    for workload in sorted(receipt):
        print("[memory]   {:<22s} stock {:.4f}  best {:.4f}  handover {:.0%}".format(
            workload, receipt[workload]["stock"], receipt[workload]["best"], receipt[workload]["share"]), flush=True)
    return shelf


# ---------------------------------------------------------------- filling a memory ----

def measure(problem, designs, label):
    """Simulate a list of designs on a problem's whole suite, reporting the cost.

    One design that makes the simulator crash must not end a build that has been
    running for hours, so a failed batch is retried design by design (everything
    that already finished is in the table, so the retry is nearly free) and
    whatever still fails is dropped."""
    if len(designs) == 0:
        return
    started = time.time()
    print("[{}] {} designs x {} workloads".format(label, len(designs), len(problem["workloads"])), flush=True)
    # In chunks, so a stage that runs for an hour still reports progress. A chunk
    # of eight designs is 24 to 48 simulations, which keeps the machine full.
    done = 0
    while done < len(designs):
        chunk = designs[done:done + CHUNK]
        measure_chunk(problem, chunk, label)
        done += len(chunk)
        print("[{}] {}/{} designs, {:.0f} min elapsed".format(
            label, done, len(designs), (time.time() - started) / 60.0), flush=True)
    print("[{}] done in {:.0f} min".format(label, (time.time() - started) / 60.0), flush=True)


def measure_chunk(problem, designs, label):
    try:
        problem["evaluate_many"](designs)
    except Exception as error:
        print("[{}] batch failed ({}), retrying one design at a time".format(label, repr(error)[-150:]), flush=True)
        dropped = 0
        for knobs in designs:
            try:
                problem["evaluate"](knobs)
            except Exception as design_error:
                dropped += 1
                print("[{}] dropped {}: {}".format(label, problem["name_of"](knobs), repr(design_error)[-120:]), flush=True)
        print("[{}] {} designs dropped".format(label, dropped), flush=True)


def top_designs(history, objective, how_many):
    scored = []
    for entry in history:
        scored.append((entry["metrics"][objective], entry["knobs"]))
    scored.sort(reverse=True, key=first_item)
    designs = []
    for score, knobs in scored[:how_many]:
        designs.append(knobs)
    return designs


def first_item(pair):
    return pair[0]


def search(problem, tag):
    """Stage 1: an LLM search over one group from the stock chip,
    PER_ROUND designs a round, every proposal measured. It starts from nothing - a
    memory is what this build produces, not something it reads."""
    # Imported here rather than at the top: loop.search reads the handover from here.
    from loop.search import llm_search
    return llm_search(problem, SEARCH_DESIGNS, PER_ROUND, tag)


def build(groups, output_path):
    """Both stages over the groups of memory workloads, writing the memory after
    each group is confirmed."""
    all_traces = []
    for group in groups:
        all_traces.extend(group)
    everywhere = make_suite_problem(all_traces)
    started = time.time()
    print("memory build: {} designs per group, top {} confirmed | groups {}".format(
        SEARCH_DESIGNS, CONFIRM_TOP, [[short_name(t) for t in g] for g in groups]), flush=True)

    # Only designs this build asked for may enter the memory; the tables hold more.
    asked = {everywhere["name_of"](everywhere["stock"])}

    # ---- stage 1: one search per group, the groups searching at the same time ----
    searched = []
    pool = ThreadPoolExecutor(max_workers=len(groups))
    handles = []
    for index, group in enumerate(groups):
        problem = make_suite_problem(group)
        tag = "membuild-g{}".format(index + 1)
        handles.append((problem, tag, pool.submit(search, problem, tag)))
    for problem, tag, handle in handles:
        history = handle.result()["designs"]
        best = history[0]
        for entry in history:
            asked.add(problem["name_of"](entry["knobs"]))
            if entry["metrics"][problem["objective"]] > best["metrics"][problem["objective"]]:
                best = entry
        searched.append(history)
        print("[{}] search done: best {:.4f} over {} designs".format(
            tag, best["metrics"][problem["objective"]], len(history) - 1), flush=True)

    # ---- stage 2: each group's best designs measured on every memory workload ----
    for index, group in enumerate(groups):
        problem = make_suite_problem(group)
        confirmed = top_designs(searched[index], problem["objective"], CONFIRM_TOP)
        for knobs in confirmed:
            asked.add(problem["name_of"](knobs))
        measure(everywhere, confirmed, "confirm:g{}".format(index + 1))
        write_memory(all_traces, output_path, asked)

    print("[memory] build done in {:.1f} h, {} designs asked for".format(
        (time.time() - started) / 3600.0, len(asked)), flush=True)
    return output_path


if __name__ == "__main__":
    if sys.argv[1] != "build":
        raise SystemExit("usage: python -m loop.memory build <out.json> <traces...> -- <traces...>")
    trace_groups = [[]]
    for argument in sys.argv[3:]:
        if argument == "--":
            trace_groups.append([])
        else:
            trace_groups[-1].append(argument)
    build(trace_groups, sys.argv[2])
