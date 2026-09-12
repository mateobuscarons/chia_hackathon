"""Build a memory from nothing, by a declared procedure with a declared budget.

The memory the agent reads is written by `loop.memory` from result tables. Until
now those tables were whatever had piled up, so nobody could rebuild the memory
or say which designs an LLM contributed. This is the procedure that fills them,
in three stages, with every design's origin recorded:

  1. SEARCH   one search per group of memory workloads (a group is a suite, e.g.
              the SPEC trio), SEARCH_DESIGNS designs, by `bo` or by the `llm`
              agent. This is the only stage the searcher changes.
  2. ANCHOR   one-knob sweeps: from the stock chip, measured on EVERY memory
              workload, and from each group's own winner, measured on that group.
              Two sweeps give each single-knob step the two controlled pairs the
              digest requires; searching first keeps outside knowledge out of the
              anchor.
  3. CONFIRM  each group's best CONFIRM_TOP designs measured on every memory
              workload, so the design the memory hands over is chosen from
              designs that were actually measured everywhere.

  python -m loop.memory_build bo  results/memory_bo.json  <spec traces> -- <gap traces>
  python -m loop.memory_build llm results/memory_llm.json <spec traces> -- <gap traces>

Writes the memory to <out.json> and the build record (every design, its stage,
its searcher, its group) next to it as <out>_record.json.
Env: MB_SEARCH (designs per group, default 40), MB_CONFIRM (top per group,
default 20), MB_ANCHOR_KNOBS (knobs swept, default all), MB_SEED.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from loop import agent, bo, memory
from loop.champsim_problem import make_suite_problem, trace_short_name
from loop.configs import KNOB_PRIORITY, SEARCH_SPACE, in_space, knobs_changed, within_budget

SEARCH_DESIGNS = int(os.environ.get("MB_SEARCH", "40"))
# Eight per round, and both groups searching at once: a round then holds 2 x 8 x 3
# simulations, enough to fill the machine. At the LLM's eight proposals per round
# this also means every design the LLM chooses is measured, with no GP in between,
# which is what makes the two memories a clean comparison of their searchers.
PER_ROUND = 8
CONFIRM_TOP = int(os.environ.get("MB_CONFIRM", "20"))
CHUNK = 8                 # designs measured per progress report
ANCHOR_KNOBS = int(os.environ.get("MB_ANCHOR_KNOBS", str(len(KNOB_PRIORITY))))
SEED = int(os.environ.get("MB_SEED", "0"))


def one_knob_sweep(base_design):
    """Every design one knob away from `base_design`: each knob at each of its
    other values, dropped if it leaves the space or the area budget."""
    designs = []
    for knob in KNOB_PRIORITY[:ANCHOR_KNOBS]:
        for value in SEARCH_SPACE[knob]:
            if str(value) == str(base_design[knob]):
                continue
            design = dict(base_design)
            design[knob] = value
            if not in_space(design):
                continue
            if not within_budget(design):
                continue
            designs.append(design)
    return designs


def best_design(history, objective):
    best = history[0]
    for entry in history:
        if entry["metrics"][objective] > best["metrics"][objective]:
            best = entry
    return best


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


class Record:
    """Design name -> where it came from. The first stage that asks for a design
    owns it; a later stage asking again is a cache hit, not a new origin."""

    def __init__(self):
        self.origins = {}

    def note(self, problem, designs, stage, source, group):
        fresh = 0
        for knobs in designs:
            name = problem["name_of"](knobs)
            if name in self.origins:
                continue
            self.origins[name] = {"stage": stage, "source": source, "group": group, "knobs": knobs}
            fresh += 1
        return fresh


def measure(problem, designs, label):
    """Simulate a list of designs on a problem's whole suite, reporting the cost.

    The sweeps deliberately visit corners no search would pick, which is where a
    simulator crash lives. One crash must not end a build that has been running
    for hours, so a failed batch is retried design by design (everything that
    already finished is in the table, so the retry is nearly free) and whatever
    still fails is dropped."""
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
        print("[{}] {}/{} designs, {:.0f} min elapsed".format(label, done, len(designs), (time.time() - started) / 60.0), flush=True)
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


def search_group(problem, searcher, tag):
    rounds = SEARCH_DESIGNS // PER_ROUND
    if searcher == "bo":
        return bo.run_bo(problem, rounds, PER_ROUND, SEED, tag)
    if searcher == "llm":
        return agent.run_agent(problem, rounds, PER_ROUND, tag, None, False, True, seed=SEED)
    raise ValueError("unknown searcher " + searcher)


def build(searcher, groups, output_path):
    all_traces = []
    for group in groups:
        all_traces.extend(group)
    everywhere = make_suite_problem(all_traces)
    record = Record()
    started = time.time()
    print("memory build: searcher {} | {} designs per group, top {} confirmed | groups {}".format(
        searcher, SEARCH_DESIGNS, CONFIRM_TOP, [[trace_short_name(t) for t in g] for g in groups]), flush=True)

    record.note(everywhere, [everywhere["stock"]], "stock", "stock", "all")

    # ---- stage 1: one search per group, the groups searching at the same time ----
    winners = []
    searched = []
    pool = ThreadPoolExecutor(max_workers=len(groups))
    handles = []
    for index, group in enumerate(groups):
        problem = make_suite_problem(group)
        tag = "membuild-{}-g{}".format(searcher, index + 1)
        handles.append((problem, tag, pool.submit(search_group, problem, searcher, tag)))
    for problem, tag, handle in handles:
        history = handle.result()["designs"]
        for entry in history[1:]:
            # "bo", or "llm:pick" / "llm:pick+gp" / "llm:perturbation": who proposed this design.
            origin = entry["source"]
            if origin != searcher:
                origin = searcher + ":" + origin
            record.note(problem, [entry["knobs"]], "search", origin, tag)
        winners.append(best_design(history, problem["objective"])["knobs"])
        searched.append(history)
        print("[{}] search done: best {:.4f} over {} designs".format(
            tag, best_design(history, problem["objective"])["metrics"][problem["objective"]], len(history) - 1), flush=True)

    write_outputs(record, all_traces, output_path, searcher, groups, started, "search")

    # ---- stage 2: anchor sweeps ----
    stock_sweep = one_knob_sweep(everywhere["stock"])
    record.note(everywhere, stock_sweep, "anchor", "sweep:stock", "all")
    measure(everywhere, stock_sweep, "anchor:stock")
    for index, group in enumerate(groups):
        problem = make_suite_problem(group)
        winner_sweep = one_knob_sweep(winners[index])
        tag = "membuild-{}-g{}".format(searcher, index + 1)
        record.note(problem, winner_sweep, "anchor", "sweep:winner", tag)
        measure(problem, winner_sweep, "anchor:winner:g{}".format(index + 1))

    write_outputs(record, all_traces, output_path, searcher, groups, started, "anchor")

    # ---- stage 3: confirm the best designs everywhere ----
    for index, group in enumerate(groups):
        problem = make_suite_problem(group)
        confirmed = top_designs(searched[index], problem["objective"], CONFIRM_TOP)
        record.note(problem, confirmed, "confirm", "top{}".format(CONFIRM_TOP), "membuild-{}-g{}".format(searcher, index + 1))
        measure(everywhere, confirmed, "confirm:g{}".format(index + 1))

    write_outputs(record, all_traces, output_path, searcher, groups, started, "complete")
    return output_path


def write_outputs(record, all_traces, output_path, searcher, groups, started, after_stage):
    """The memory and the build record, written after every stage: a build stopped
    halfway still leaves a usable memory of what it had measured by then.

    The tables are a shared simulation cache, so a design another build already
    measured costs nothing here; the memory still holds only what this one asked for."""
    built = memory.build(all_traces, output_path, set(record.origins.keys()))
    built["built_by"] = {"searcher": searcher, "search_designs": SEARCH_DESIGNS, "confirm_top": CONFIRM_TOP,
                         "anchor_knobs": ANCHOR_KNOBS, "seed": SEED, "after_stage": after_stage,
                         "groups": [[trace_short_name(t) for t in g] for g in groups],
                         "designs_requested": len(record.origins), "wall_hours": (time.time() - started) / 3600.0}
    memory.save(built, output_path)
    record_path = output_path.replace(".json", "") + "_record.json"
    with open(record_path, "w") as record_file:
        json.dump({"built_by": built["built_by"], "origins": record.origins}, record_file, indent=1)
    counts = {}
    for name in record.origins:
        key = record.origins[name]["stage"] + " / " + record.origins[name]["source"]
        counts[key] = counts.get(key, 0) + 1
    print("[membuild] after {}: {} designs requested in {:.1f} h -> {}".format(
        after_stage, len(record.origins), built["built_by"]["wall_hours"], record_path), flush=True)
    for key in sorted(counts):
        print("[membuild]   {:<24s} {}".format(key, counts[key]), flush=True)


def combine(output_path, record_paths):
    """One memory from the designs several builds asked for, together. The builds
    share stages 2 and 3 by construction and differ only in who searched, so the
    union is the same procedure with more than one searcher in stage 1 — at the
    sum of their budgets, which `built_by` states."""
    from loop import run
    trace_of = {}
    for trace_path in run.TRACE.values():
        trace_of[trace_short_name(trace_path)] = trace_path
    names = set()
    searchers = []
    workloads = []
    requested = 0
    for record_path in record_paths:
        with open(record_path) as record_file:
            record = json.load(record_file)
        names.update(record["origins"].keys())
        searchers.append(record["built_by"]["searcher"])
        requested += record["built_by"]["designs_requested"]
        for group in record["built_by"]["groups"]:
            for workload in group:
                if workload not in workloads:
                    workloads.append(workload)
    traces = []
    for workload in workloads:
        traces.append(trace_of[workload])
    built = memory.build(traces, output_path, names)
    built["built_by"] = {"searcher": "+".join(searchers), "combined_from": record_paths,
                         "designs_requested": len(names), "designs_requested_separately": requested}
    memory.save(built, output_path)
    print("[combine] {} designs from {} builds ({} before removing the designs they share) -> {}".format(
        len(names), len(record_paths), requested, output_path), flush=True)


def compare(cell_name, memory_paths):
    """Read two or more memories side by side, with no simulation: what each one
    hands over, how much evidence it holds, and how well its remembered effects
    carry from one of its own workloads to another. The full digest of each goes
    to <memory>_digest.txt, which is what to diff."""
    from loop import run
    from loop.champsim_problem import make_suite_problem
    problem = make_suite_problem(run.CELLS[cell_name]["test"], allow_simulation=False)
    stock = problem["stock"]
    print("memories read against cell {} ({})".format(cell_name, " + ".join(problem["workloads"])))
    print()
    print("{:<26s} {:>6s} {:>9s} {:>8s} {:>9s} {:>7s}  {}".format(
        "memory", "cases", "designs", "effects", "survival", "share", "hands over (changed from stock)"))
    for path in memory_paths:
        shelf = memory.load(path)
        if len(shelf["cases"]) == 0:
            print("{:<26s} (empty)".format(os.path.basename(path)))
            continue
        designs = 0
        solid = 0
        for case in shelf["cases"]:
            designs += case["designs_measured"]
            solid += count_solid(case["effects"])
        checked_total = 0
        survived_total = 0
        for source_case in shelf["cases"]:
            for target_case in shelf["cases"]:
                if source_case["id"] == target_case["id"]:
                    continue
                checked, survived = memory.sign_survival(source_case, target_case)
                checked_total += checked
                survived_total += survived
        survival = 0.0
        if checked_total > 0:
            survival = 100.0 * survived_total / checked_total
        handover = "(none)"
        share = ""
        if shelf.get("pooled") is not None:
            changed = knobs_changed(shelf["pooled"]["knobs"], stock)
            parts = []
            for knob in changed:
                parts.append("{} {}".format(knob, changed[knob][1]))
            handover = ", ".join(parts)
            share = "{:.0f}%".format(100.0 * shelf["pooled"]["mean_share"])
        print("{:<26s} {:>6d} {:>9d} {:>8d} {:>8.0f}% {:>7s}  {}".format(
            os.path.basename(path), len(shelf["cases"]), designs, solid, survival, share, handover))
        retrieval = memory.retrieve(shelf, problem)
        digest_path = path.replace(".json", "") + "_digest.txt"
        with open(digest_path, "w") as digest_file:
            digest_file.write(memory.digest(shelf, problem, retrieval))
    print()
    print("full digests written next to each memory as <memory>_digest.txt")


def count_solid(effects):
    count = 0
    for effect in effects:
        if effect["pairs"] >= memory.MIN_PAIRS:
            count += 1
    return count


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3:])
        raise SystemExit(0)
    if sys.argv[1] == "combine":
        combine(sys.argv[2], sys.argv[3:])
        raise SystemExit(0)
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        from loop import run
        run.start_chia()
    searcher_name = sys.argv[1]
    out = sys.argv[2]
    trace_groups = [[]]
    for argument in sys.argv[3:]:
        if argument == "--":
            trace_groups.append([])
        else:
            trace_groups[-1].append(argument)
    build(searcher_name, trace_groups, out)
