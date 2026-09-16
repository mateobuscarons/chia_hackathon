"""One specialist per architectural concern, one concern moving per round, from the
memory's handover.

The thirteen knobs split by concern - what prefetches, how big, what policy, how many
misses in flight - which is the hierarchy's own structure. A specialist owns a whole
concern and changes it in one go, so a move whose parts each look worse separately
is still one move.

A round: one concern leads, its specialist proposes new values for its knobs, and the
design measured is the incumbent with only those knobs changed, so every measurement
is attributable. The concerns rotate, offset by the seed; a concern whose design
became the incumbent leads again; a specialist that holds spends no design and passes
the turn. When every concern has held on the current design the search stops.

Each view is built from this run's history alone, never the shared tables, so the arm
sees what `bo` and `pooled_bo` see. The principles say what a knob does, never which
value to pick, and name no workload, trace or chip.
"""

import json
import os
import time

from loop import analyst
from loop import search as engine
from loop.memory import fit_to_budget
from loop.simulate import derived_latency
from loop.space import SEARCH_SPACE, cache_area_kb, knobs_changed, typed_knobs

# A concern owns a disjoint set of knobs and what it knows before it has measured
# anything. The levels it can read a report for are the levels it owns knobs at.
CONCERNS = {
    "prefetch": {
        "knobs": ["l1d_prefetcher", "l2_prefetcher", "llc_prefetcher"],
        "principles": [
            "A prefetcher hides the latency of the level below it. At the first level it must be "
            "timely above all; deeper levels see more history and less timing pressure but sit "
            "further from the use.",
            "Judge a prefetcher by coverage (the share of misses it removes) and accuracy (useful "
            "prefetches over issued). How much IPC responds to each varies by workload.",
            "The same mechanisms are available at more than one level, so the decision is which "
            "mechanism goes where, not which is best at each level on its own. A mechanism that "
            "works at one level may work better one level down, and moving it there frees the "
            "level above for a simpler one. Exchanging two mechanisms you already have is a move.",
            "Stronger is not better. Depth, timeliness and pollution decide whether a mechanism "
            "fits a level: inaccurate or over-aggressive prefetching spends bandwidth on lines "
            "never used, so a weaker mechanism at a level can beat a stronger one.",
            "Strided and streaming access is covered by simple stride prefetchers. Irregular "
            "access needs spatial or correlating designs, and some of it cannot be covered at all.",
        ],
    },
    "geometry": {
        "knobs": ["l1d_sets", "l1d_ways", "l2_sets", "l2_ways", "llc_sets", "llc_ways"],
        "principles": [
            "Capacity is not free. Hit latency is a function of capacity alone: re-splitting the "
            "same capacity between sets and ways changes associativity, not latency. Enlarging a "
            "level lowers its miss rate and raises its hit time - and the hit time is paid on "
            "every access that reaches it, including the ones that go on to miss.",
            "A level with a low hit ratio charges its hit latency on nearly every access and then "
            "misses anyway. Shrinking it takes that latency out of the path to the level below "
            "and frees area. The right size is set by where the hits are, not by the budget.",
            "Area is a budget to allocate between levels, not a quantity to spend in full. "
            "Capacity placed where it does not remove misses buys latency and nothing else.",
            "The level worth growing is the one whose misses are both frequent and not already "
            "covered by a prefetcher. A level whose misses a prefetcher hides gains little from "
            "capacity.",
        ],
    },
    "replacement": {
        "knobs": ["l2_replacement", "llc_replacement"],
        "principles": [
            "Replacement matters only when lines are evicted before they are reused. If the "
            "working set fits, every policy performs alike; if there is no reuse at all, no "
            "policy helps. Read the level's miss and hit evidence to see which pattern it is in.",
            "Three access patterns drive the choice. Recency-friendly: the working set fits and "
            "reuse distances are short. Thrashing: the working set exceeds the capacity and so do "
            "the reuse distances, where LRU can be the worst possible choice. Scans: bursts of "
            "references that never repeat, which under LRU evict the working set.",
            "LRU suits recency-friendly access. SRRIP resists scans. DRRIP resists both scans and "
            "thrashing. SHiP predicts reuse distance and refines where a line is inserted.",
            "The gain is largest at the last level, where reuse distances are longest and the "
            "capacity pressure is greatest.",
            "Replacement and prefetching interact. Prefetched lines that go unused pollute, and a "
            "prefetcher that removes misses also removes the reuse a cleverer policy was "
            "exploiting, so a policy worth points alone can be worth nothing behind a prefetcher.",
        ],
    },
    "concurrency": {
        "knobs": ["l2_mshr", "llc_mshr"],
        "principles": [
            "Miss-handling registers bound how many misses can be outstanding, and are the main "
            "limit on memory-level parallelism.",
            "By Little's law the bandwidth a level can sustain is about its outstanding misses "
            "divided by the average miss latency. Too few registers leave a design latency-bound "
            "while bandwidth is still available; adding more past that point changes nothing.",
            "They matter in proportion to how often the level misses. Where the miss rate is low, "
            "miss concurrency is not the constraint; where it is high, they are the first thing to check.",
            "Prefetches occupy the same registers as demand misses, so a more aggressive "
            "prefetcher raises the concurrency a level needs.",
        ],
    },
}

ORDER = ["prefetch", "geometry", "replacement", "concurrency"]

# The knob prefix names the level it sits at, and the level has a shape.
LEVEL_OF = {"l1d": "L1D", "l2": "L2C", "llc": "LLC"}
SHAPE_OF = {"L1D": ("l1d_sets", "l1d_ways"), "L2C": ("l2_sets", "l2_ways"), "LLC": ("llc_sets", "llc_ways")}

for _name in ORDER:
    _levels = []
    for _knob in CONCERNS[_name]["knobs"]:
        _level = LEVEL_OF[_knob.split("_")[0]]
        if _level not in _levels:
            _levels.append(_level)
    CONCERNS[_name]["levels"] = _levels

# Merging relies on the concerns covering every knob exactly once.
_owned = [knob for name in ORDER for knob in CONCERNS[name]["knobs"]]
if sorted(_owned) != sorted(SEARCH_SPACE):
    raise RuntimeError("concerns do not partition the search space: " + str(sorted(_owned)))


def note(tag, text):
    """Append to this run's transcript as it happens, so a run in flight can be read
    round by round. One file per seed; `results/*.log` is already gitignored."""
    with open(os.path.join("results", tag + ".log"), "a") as log_file:
        log_file.write(text + "\n")


# ---------------------------------------------------------------- what a specialist is shown ----

def assignment(knobs, concern_knobs):
    """The concern's own slice of a design."""
    return ", ".join("{}={}".format(knob, knobs[knob]) for knob in concern_knobs)


# What each mechanism tracks and the history it needs before it can predict. The
# counterpart of the geometry ladder: facts about the values, not advice on them.
MECHANISMS = {
    "no": "no prefetching.",
    "next_line": "fetches the line after the one accessed; no training and no history, so it is "
                 "always timely and is right only where access is sequential.",
    "ip_stride": "learns a constant stride per load instruction and runs ahead of it; needs a few "
                 "repeats of the same instruction to train, covers regular strides only.",
    "va_ampm_lite": "keeps an access map of the lines touched in each page and prefetches the lines "
                    "whose offset pattern already appeared in the map; needs the page's history to "
                    "accumulate, covers irregular but spatially clustered access.",
    "spp_dev": "compresses the recent deltas within a page into a signature, predicts the next delta "
               "from it and follows the path several lines ahead; the deepest and most aggressive, "
               "and the one that needs the longest history and the most misses to learn from.",
}


def placement_table(concern_knobs):
    """The union of the mechanisms and the levels that accept each. One table, not
    one list per level, because where a mechanism goes is a single decision."""
    accepts = {}
    for knob in concern_knobs:
        for value in SEARCH_SPACE[knob]:
            accepts.setdefault(value, []).append(LEVEL_OF[knob.split("_")[0]])
    return "\n".join("  {:14s} {:14s} {}".format(value, ", ".join(accepts[value]), MECHANISMS.get(value, ""))
                     for value in accepts)


def latency_ladder(levels):
    """Every capacity a level can take and the hit latency it costs. Derived from
    the shape, so it is free to print and it is the same rule on any chip."""
    lines = []
    for level in levels:
        sets_knob, ways_knob = SHAPE_OF[level]
        costs = {}
        for sets in SEARCH_SPACE[sets_knob]:
            for ways in SEARCH_SPACE[ways_knob]:
                costs[sets * ways] = derived_latency(sets, ways)
        lines.append("  {}: {}".format(level, " | ".join(
            "{} KB {} cy".format(blocks * 64 // 1024, costs[blocks]) for blocks in sorted(costs))))
    return "\n".join(lines)


def level_report(entry, levels, workloads):
    """One design's per-level report, for the levels a concern can read: the shape,
    which is what fixes the hit latency paid on every access that reaches the level,
    then what each workload measured there. The levels are listed top down, so a
    level's misses are read against the line below it."""
    lines = []
    for level in levels:
        sets_knob, ways_knob = SHAPE_OF[level]
        sets = int(entry["knobs"][sets_knob])
        ways = int(entry["knobs"][ways_knob])
        cells = []
        for workload in workloads:
            def got(metric):
                return entry["metrics"].get("{}:{}_{}".format(workload, level, metric))
            if got("mpki") is None:
                continue
            cell = "{} mpki {:.2f}, hit {:.2f}".format(workload, got("mpki"), got("hit_ratio") or 0.0)
            if got("pf_coverage") is not None:
                cell += ", prefetch cov {:.2f} acc {:.2f}".format(
                    got("pf_coverage"), got("pf_accuracy") or 0.0)
            if got("miss_latency"):
                cell += ", miss {:.0f} cy".format(got("miss_latency"))
            cells.append(cell)
        lines.append("  {}: {} x {} = {} blocks, {} KB, hit latency {} cycles | {}".format(
            level, sets, ways, sets * ways, sets * ways * 64 // 1024, derived_latency(sets, ways),
            " | ".join(cells) if cells else "counters not measured for this design"))
    return "\n".join(lines)


def measured_view(history, concern_knobs, objective, incumbent):
    """What this run has measured for each assignment of the concern's knobs, best
    first. Best-of rather than latest: another specialist may have spoiled a design
    that carried a good assignment.

    A row's score belongs to the WHOLE design, not to this concern's slice of it, so
    each row also says how many knobs outside the concern differed from the current
    design. Only a row at 0 is a clean comparison."""
    best = {}
    for entry in history:
        key = assignment(entry["knobs"], concern_knobs)
        value = entry["metrics"][objective]
        if key not in best or value > best[key][0]:
            outside = [knob for knob in knobs_changed(entry["knobs"], incumbent) if knob not in concern_knobs]
            best[key] = (value, entry["index"], len(outside))
    rows = sorted(best.items(), key=lambda item: item[1][0], reverse=True)
    lines = ["  {:.4f}  (D{}, {} other knobs differ from the current design)  {}".format(
        value, index, outside, key) for key, (value, index, outside) in rows[:10]]
    return "\n".join(lines) if lines else "  nothing measured yet"


def recent(history, problem, how_many=6):
    """The team's shared record: the last few designs and what they scored."""
    lines = []
    for entry in history[-how_many:]:
        changed = knobs_changed(entry["knobs"], problem["stock"])
        text = ", ".join("{}={}".format(knob, changed[knob][1]) for knob in changed) or "stock"
        lines.append("  D{} {}={:.4f}  {}".format(
            entry["index"], problem["objective"], entry["metrics"][problem["objective"]], text))
    return "\n".join(lines)


TASK = """## Task
You alone move this round: the design measured next is the current design with your knobs set
to what you return. Every other knob stays as it is.

A design costs one simulation, so move as far as your levels' evidence supports in one design
rather than one notch at a time; the next round tells you if it was too far. Hold (return the
current values) when the evidence at your levels shows nothing worth a simulation: a hold costs
nothing and passes the turn.

Do not re-propose an assignment of your knobs that this run has already measured below the
current design, unless that row says other knobs differed and those differences change your
reasoning.

Return JSON with exactly these keys:
- "reasoning": one sentence - the mechanism you are acting on, or why you are holding.
- "knobs": an object with exactly your knobs and one allowed value each.
"""


def specialist(problem, name, history, incumbent, incumbent_entry, tag):
    """One specialist's proposal for its own knobs, or None if it could not answer."""
    concern = CONCERNS[name]
    objective = problem["objective"]
    parts = ["## Role",
             "You are the {} specialist in a team tuning one cache hierarchy.".format(name),
             "You choose only these knobs: " + ", ".join(concern["knobs"]) + ".",
             "Other specialists own every other knob; this round only yours move.",
             "",
             "## Principles"]
    parts += ["- " + line for line in concern["principles"]]
    parts += ["",
              "## Chip", problem["chip_text"],
              "Objective: maximise {}, the geometric mean of IPC over {}.".format(
                  objective, " + ".join(problem["workloads"])),
              "",
              "## Your knobs and their allowed values"]
    if name == "prefetch":
        parts += ["One mechanism per level. The same mechanisms serve several levels, so this is one",
                  "placement decision across the levels, not one choice per level.",
                  placement_table(concern["knobs"])]
    else:
        parts += ["- {}: {}".format(knob, json.dumps(SEARCH_SPACE[knob])) for knob in concern["knobs"]]
    if name == "geometry":
        parts += ["", "Capacity and the hit latency it costs (latency follows capacity alone):",
                  latency_ladder(concern["levels"])]
    parts += ["",
              "## The current design (the best this run has measured)",
              "  yours:  " + assignment(incumbent, concern["knobs"]),
              "  area:   {:.0f} of {} KB of the budget in use by L2 + LLC".format(
                  cache_area_kb(incumbent), problem["area_budget_kb"]),
              level_report(incumbent_entry, concern["levels"], problem["workloads"])]
    if history[-1]["index"] != incumbent_entry["index"]:
        parts += ["",
                  "## The design measured last round (D{}, {}={:.4f})".format(
                      history[-1]["index"], objective, history[-1]["metrics"][objective]),
                  "  yours:  " + assignment(history[-1]["knobs"], concern["knobs"]),
                  "  it moved: " + (", ".join("{} {}->{}".format(knob, moved_from, moved_to)
                                              for knob, (moved_from, moved_to)
                                              in sorted(knobs_changed(history[-1]["knobs"], incumbent).items()))
                                    or "nothing"),
                  level_report(history[-1], concern["levels"], problem["workloads"])]
    parts += ["",
              "## What your knobs have scored",
              measured_view(history, concern["knobs"], objective, incumbent),
              "",
              "## The team's recent designs",
              recent(history, problem),
              "",
              TASK]
    prompt_text = "\n".join(parts)
    note(tag, "\n---------------- {} specialist ----------------\n{}".format(name, prompt_text))
    for attempt in range(2):
        try:
            answer = analyst.ask(prompt_text)
        except Exception as error:
            print("[{}] {} specialist failed ({})".format(tag, name, repr(error)[-120:]), flush=True)
            note(tag, "-> call failed: " + repr(error)[-160:])
            return None
        proposed = read_proposal(answer, concern["knobs"])
        if proposed is not None:
            held = proposed["knobs"] == {knob: incumbent[knob] for knob in concern["knobs"]}
            note(tag, "-> {}{}\n   because: {}".format(
                assignment(proposed["knobs"], concern["knobs"]), "   (held)" if held else "",
                proposed["reasoning"]))
            return proposed
        print("[{}] {} specialist answered off-contract, one retry".format(tag, name), flush=True)
        note(tag, "-> off-contract, retrying: " + json.dumps(answer)[:300])
    note(tag, "-> off-contract twice, no proposal")
    return None


def read_proposal(answer, concern_knobs):
    """The answer as this concern's knobs, or None if it is not one."""
    if isinstance(answer, list) and answer:
        answer = answer[0]
    if not isinstance(answer, dict) or not isinstance(answer.get("knobs"), dict):
        return None
    proposed = {}
    for knob in concern_knobs:
        value = answer["knobs"].get(knob)
        if value is None or str(value) not in [str(option) for option in SEARCH_SPACE[knob]]:
            return None
        proposed[knob] = value
    return {"knobs": proposed, "reasoning": str(answer.get("reasoning", ""))[:300]}


def merge(incumbent, proposal, tag):
    """The next design: the incumbent with one concern's knobs replaced. A design over
    the area budget is fitted back into it, which shrinks the LLC before the L2. Say so
    when it happens: the specialist is scored for a design it did not propose."""
    design = dict(incumbent)
    design.update(proposal["knobs"])
    fitted = fit_to_budget(design)
    trimmed = knobs_changed(fitted, design)
    if trimmed:
        print("[{}] over budget: the fit trimmed {}".format(tag, ", ".join(
            "{} {}->{}".format(knob, trimmed[knob][0], trimmed[knob][1]) for knob in trimmed)), flush=True)
    return typed_knobs(fitted)


# ---------------------------------------------------------------- the search ----

def search(problem, budget, seed, tag, start_design=None):
    """`budget` designs, one a round, one concern moving per round."""
    objective = problem["objective"]
    started = time.time()
    history = []
    rounds = []

    def buy(knobs, source, round_number, hypothesis):
        knobs_list, metrics_list, _ = engine.measure_batch(problem, [typed_knobs(knobs)], tag)
        if not knobs_list:
            return None
        entry = {"index": len(history), "round": round_number, "name": problem["name_of"](knobs_list[0]),
                 "knobs": knobs_list[0], "metrics": metrics_list[0], "source": source, "hypothesis": hypothesis}
        history.append(entry)
        note(tag, "\n======== D{} [{}] {}={:.4f} | {} ========".format(
            entry["index"], source, objective, entry["metrics"][objective],
            ", ".join("{}={}".format(knob, entry["knobs"][knob])
                      for knob in sorted(knobs_changed(entry["knobs"], problem["stock"]))) or "stock"))
        print("[{}] round {} | D{} | {}={:.4f} | {} | {:.0f} min".format(
            tag, round_number, entry["index"], objective, entry["metrics"][objective], source,
            (time.time() - started) / 60), flush=True)
        return entry

    buy(problem["stock"], "stock", 0, None)
    if start_design is not None:
        buy(start_design, "memory", 0, "the design the memory hands over")

    round_number = 0
    pointer = seed % len(ORDER)
    held = set()          # concerns that held on the current incumbent
    held_on = None
    leader = None
    won = False
    while len(history) - 1 < budget:
        incumbent_entry = max(history, key=lambda entry: entry["metrics"][objective])
        incumbent = incumbent_entry["knobs"]
        if held_on != incumbent_entry["index"]:
            held, held_on = set(), incumbent_entry["index"]
        if len(held) == len(ORDER):
            print("[{}] every concern held on D{}: no move left, stopping".format(
                tag, incumbent_entry["index"]), flush=True)
            break
        round_number += 1
        if not won:
            while ORDER[pointer] in held:
                pointer = (pointer + 1) % len(ORDER)
            leader = ORDER[pointer]
            pointer = (pointer + 1) % len(ORDER)
        note(tag, "\n\n################ round {} | incumbent D{} {}={:.4f} | leader {} ################".format(
            round_number, incumbent_entry["index"], objective, incumbent_entry["metrics"][objective], leader))

        proposal = specialist(problem, leader, history, incumbent, incumbent_entry, tag)
        if proposal is None or proposal["knobs"] == {knob: incumbent[knob] for knob in CONCERNS[leader]["knobs"]}:
            held.add(leader)
            won = False
            print("[{}] round {} | {} held".format(tag, round_number, leader), flush=True)
            continue
        design = merge(incumbent, proposal, tag)
        moved = sorted(knobs_changed(design, incumbent))
        if not moved:
            held.add(leader)
            won = False
            continue
        note(tag, "\n======== {} moved {} ========".format(leader, ", ".join(moved)))
        rounds.append({"round": round_number, "leader": leader, "knobs_moved": moved, "proposal": proposal})
        entry = buy(design, "council:" + leader, round_number, proposal["reasoning"])
        if entry is None:
            print("[{}] round {} | design could not be measured, stopping".format(tag, round_number), flush=True)
            break
        won = entry["metrics"][objective] > incumbent_entry["metrics"][objective]

    return {"designs": history, "rounds": rounds}
