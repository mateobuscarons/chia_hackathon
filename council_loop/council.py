"""The council: an analyst, four specialists and a statistician tuning one cache hierarchy from
the stock chip, in rounds, one wave of parallel simulations a round.

Round 1 opens the search: the analyst reads the stock chip's report, writes the sheet and one
opening design, and that design is measured in a wave with random feasible designs beside it,
OPENINGS designs in all - the opening every arm gets.

Every later round is one wave of WAVE designs. The statistician, a Gaussian-process surrogate
over every design this run measured, writes its five best candidates into every prompt and
proposes its top pick. The analyst reads every design measured and writes the sheet (a verdict
per level, the bottleneck with its evidence, each finding with its numbers, what is
unattributable, which knobs fail by their own counters, a prediction) and one whole design of
its own; when it writes none, the statistician's second candidate takes the slot. All four
specialists read the sheet and propose at once, one move each on their own knobs, or hold. A
gain the power cap refused last round comes back shrunk until its own counters estimate it
under the cap. Untried one-knob moves at the level the sheet blames top the wave up. Everything
is measured together at the run's fidelity, and the incumbent is the best feasible design
measured, whoever proposed it. A design a cap refuses after measurement is recorded with the
reason and never stood on, and the reason follows it into every ledger the team reads. The
search stops when the budget is spent, after PLATEAU designs without a new incumbent, or when
two rounds in a row measured nothing.

The concerns split the thirteen knobs by what they are - what prefetches, how big, what
policy, how many misses in flight. The principles say what a knob does, never which value to
pick, and name no workload, trace or chip; a workload is shown by position (W1, W2, ...).
"""

import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from chia.models.call_gate import gate_and_log, log_outcome
from chia.trace.ledger import Ledger

from council_loop import analyst
from council_loop import chip
from council_loop import search as engine
from council_loop.socs import REGRESSION_TOLERANCE
from council_loop import space
from council_loop.space import (SEARCH_SPACE, budget_text, cache_area_kb, config_name, knobs_changed, leakage_w, power_split,
                        random_feasible_designs, silicon_mm2, typed_knobs, violations, within_budget)

# CHIP_VIEW off reverts what the council READS about the chip to the form it had before the
# chip was derived: the override-diff chip line, the misses-per-kilo-instruction bottleneck
# ranking, the old off-chip figure, no memory-time decomposition, no traffic against what the
# channels deliver, no sharing or per-core or silicon in the report, unfiltered knob lists and
# no workload line. The chip itself is unchanged - CACTI latency, the per-core area budget and
# the per-core mixes stay, because those are the machine and not the reading. This is the
# ablation arm: it answers whether the reading helped, or only the machine.
CHIP_VIEW = os.environ.get("CHIP_VIEW", "1") == "1"

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
            "A prefetch consumes the same off-chip bandwidth as a demand miss. Where the traffic "
            "already reaching memory is close to what the channels deliver, coverage is taken from "
            "somewhere else - a weaker mechanism at another level, or accuracy - not added on top.",
        ],
    },
    "geometry": {
        "knobs": ["l1d_sets", "l1d_ways", "l2_sets", "l2_ways", "llc_sets", "llc_ways"],
        "principles": [
            "Capacity is not free, and on some levels neither is associativity: the ladder gives "
            "what every shape costs on this chip, in cycles and in silicon, and the two do not "
            "always move together. Enlarging a level lowers its miss rate and raises its hit time "
            "- and the hit time is paid on every access that reaches it, including the ones that "
            "go on to miss.",
            "A level with a low hit ratio charges its hit latency on nearly every access and then "
            "misses anyway. Shrinking it takes that latency out of the path to the level below "
            "and frees area. The right size is set by where the hits are, not by the budget. But a "
            "low hit ratio is not on its own a reason to shrink: read it against what a miss past "
            "the level costs, and against whether the level above is sending it anything to reuse.",
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
            "Off-chip traffic cannot exceed what the channels deliver. As it approaches that "
            "figure a miss queues on top of its service time, so the measured miss latency climbs "
            "above what an idle memory costs and more registers buy nothing.",
        ],
    },
}

if not CHIP_VIEW:
    # The ablation reads the chip the way the council read it before the chip was derived, so
    # the principles go back too: no bandwidth, no silicon, and the capacity-alone claim the
    # CACTI ladder later contradicted. Both are defects, and being defects is the point - they
    # are part of what the change removed.
    CONCERNS["prefetch"]["principles"] = CONCERNS["prefetch"]["principles"][:-1]
    CONCERNS["concurrency"]["principles"] = CONCERNS["concurrency"]["principles"][:-1]
    CONCERNS["geometry"]["principles"][0] = (
        "Capacity is not free. Hit latency is a function of capacity alone: re-splitting the "
        "same capacity between sets and ways changes associativity, not latency. Enlarging a "
        "level lowers its miss rate and raises its hit time - and the hit time is paid on "
        "every access that reaches it, including the ones that go on to miss.")
    CONCERNS["geometry"]["principles"][1] = (
        "A level with a low hit ratio charges its hit latency on nearly every access and then "
        "misses anyway. Shrinking it takes that latency out of the path to the level below "
        "and frees area. The right size is set by where the hits are, not by the budget.")

ORDER = ["prefetch", "geometry", "replacement", "concurrency"]

# FLAT_SHARE of the incumbent is the tolerance below which a measured change counts as flat in
# the refused list.
FLAT_SHARE = 0.005

# A round is one wave of WAVE designs measured in parallel, the same for every arm, so a round
# is the unit of time and a design the unit of cost in every comparison. Round 1 opens with
# OPENINGS designs (the analyst's opening and random feasible designs beside it) for every arm
# alike. PLATEAU designs in a row without a new incumbent end the search (AgentDSE's rule).
WAVE = int(os.environ.get("WAVE", "5"))
OPENINGS = int(os.environ.get("OPENINGS", "10"))
PLATEAU = int(os.environ.get("PLATEAU", "20"))

# ROUNDS caps the search by rounds rather than by designs. A round is the unit of time -
# one wave of parallel simulations - while a design is the unit of cost, and the two arms
# spend designs per round at different rates, so an arm is sized in whichever the
# comparison is drawn on. 0 leaves the budget to stop the run.
MAX_ROUNDS = int(os.environ.get("ROUNDS", "0"))

# The knob prefix names the level it sits at, and the level has a shape.
LEVEL_OF = {"l1d": "L1D", "l2": "L2C", "llc": "LLC"}
SHAPE_OF = {"L1D": ("l1d_sets", "l1d_ways"), "L2C": ("l2_sets", "l2_ways"), "LLC": ("llc_sets", "llc_ways")}
ALL_LEVELS = ["L1D", "L2C", "LLC"]

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


TRANSCRIPTS = os.path.join("results", "transcripts")


def note(tag, text):
    """Append to this run's transcript as it happens, so a run in flight can be read
    round by round. One file per (SoC, arm, seed), named so it pairs with its report."""
    os.makedirs(TRANSCRIPTS, exist_ok=True)
    with open(os.path.join(TRANSCRIPTS, tag + ".log"), "a") as log_file:
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


def feasible_values(knob, current):
    """The values this knob can take with the rest of the design as it is: a geometry the
    area budget forbids on this chip is not an option, so it is never offered."""
    if not CHIP_VIEW:
        return SEARCH_SPACE[knob]
    return [value for value in SEARCH_SPACE[knob]
            if within_budget(dict(current, **{knob: value}))]


def feasible_shapes(level, current):
    """The (sets, ways) this level can take, everything else held at the current design."""
    sets_knob, ways_knob = SHAPE_OF[level]
    shapes = []
    for sets in SEARCH_SPACE[sets_knob]:
        for ways in SEARCH_SPACE[ways_knob]:
            if within_budget(dict(current, **{sets_knob: sets, ways_knob: ways})):
                shapes.append((sets, ways))
    return shapes


def latency_ladder(levels, current):
    """What every capacity this chip can build costs at each level: the hit latency in
    this chip's cycles and the silicon it takes, every instance of a private level
    counted. Both come from the cache characterised at this chip's node and clock, so
    the ladder belongs to this chip and to no other."""
    lines = []
    for level in levels:
        if not CHIP_VIEW:
            costs = {sets * ways: chip.latency(level, sets, ways)
                     for sets in SEARCH_SPACE[SHAPE_OF[level][0]]
                     for ways in SEARCH_SPACE[SHAPE_OF[level][1]]}
            lines.append("  {}: {}".format(level, " | ".join(
                "{} KB {} cy".format(blocks * 64 // 1024, costs[blocks]) for blocks in sorted(costs))))
            continue
        by_capacity = {}
        for sets, ways in feasible_shapes(level, current):
            silicon = chip.area_mm2(level, sets, ways)
            cost = (chip.latency(level, sets, ways),
                    "{:.2f}".format(silicon) if silicon < 1 else "{:.1f}".format(silicon),
                    " {:.2f} nJ leaks {:.2f} W".format(chip.energy(level, sets, ways)[0], chip.leakage_w(level, sets, ways))
                    if chip.card()["power_budget_w"] is not None else "")
            by_capacity.setdefault(sets * ways * 64 // 1024, {}).setdefault(cost, set()).add(ways)
        cells = []
        for kilobytes in sorted(by_capacity):
            for (cycles, area, energy), widths in sorted(by_capacity[kilobytes].items()):
                at = "" if len(by_capacity[kilobytes]) == 1 else " at {} ways".format(
                    " or ".join(str(width) for width in sorted(widths)))
                cells.append("{} KB{} {} cy {} mm2{}".format(kilobytes, at, cycles, area, energy))
        lines.append("  {}: {}".format(level, " | ".join(cells)))
    return "\n".join(lines)


def capacity_ladder(history, levels, workloads, objective, current):
    """What the run's own measurements say a capacity buys: per level, the capacities it has
    been measured at, the misses per kilo-instruction the level still takes there and the best
    objective reached with it. The latency ladder says what a capacity costs; this says what it
    removes, which the counters of the designs already evaluated answer for free. Only designs
    carrying the current design's prefetcher at that level are counted, because a prefetcher
    moves a level's misses as much as its capacity does."""
    lines = []
    for level in levels:
        sets_knob, ways_knob = SHAPE_OF[level]
        prefetcher = {"L1D": "l1d_prefetcher", "L2C": "l2_prefetcher", "LLC": "llc_prefetcher"}[level]
        seen = {}
        for entry in history:
            if str(entry["knobs"][prefetcher]) != str(current[prefetcher]):
                continue
            measured = [entry["metrics"].get("{}:{}_mpki".format(workload, level)) for workload in workloads]
            measured = [value for value in measured if value is not None]
            if not measured:
                continue
            kilobytes = int(entry["knobs"][sets_knob]) * int(entry["knobs"][ways_knob]) * 64 // 1024
            stats = seen.setdefault(kilobytes, {"n": 0, "mpki": 0.0, "best": None})
            stats["n"] += 1
            stats["mpki"] += sum(measured) / len(measured)
            value = entry["metrics"][objective]
            if stats["best"] is None or value > stats["best"]:
                stats["best"] = value
        if seen:
            lines.append("  {}: {}".format(level, " | ".join(
                "{} KB: {} design{}, mpki {:.2f}, best {:.4f}".format(
                    kilobytes, seen[kilobytes]["n"], "" if seen[kilobytes]["n"] == 1 else "s",
                    seen[kilobytes]["mpki"] / seen[kilobytes]["n"], seen[kilobytes]["best"])
                for kilobytes in sorted(seen))))
    return "\n".join(lines) if lines else "  nothing measured yet"


def chip_line(problem):
    """The chip as the prompts see it: the derived card, or the old override diff."""
    return problem["chip_text"] if CHIP_VIEW else chip.chip_text_diff()


def cost_line(entry, problem):
    """What the design spends: the budgeted quantity, and on a capped chip the measured
    power against its cap, split by where it goes."""
    knobs = entry["knobs"]
    if not CHIP_VIEW:
        return "  area:   {:.0f} of {} KB of the budget in use by L2 + LLC".format(
            cache_area_kb(knobs), problem["area_budget_kb"])
    card = chip.card()
    if card["area_budget_mm2"] is None:
        silicon = sum(chip.area_mm2(level, knobs[SHAPE_OF[level][0]], knobs[SHAPE_OF[level][1]])
                      for level in ALL_LEVELS)
        return "  area:   {}; the three levels take {:.2f} mm2".format(budget_text(knobs), silicon)
    split = power_split(knobs, entry["metrics"], problem["workloads"])
    cap = " of {} W".format(card["power_budget_w"]) if card["power_budget_w"] is not None else " W"
    return "  cost:   {:.2f} of {} mm2 | {:.2f}{}: {}".format(
        entry["metrics"]["mm2"], card["area_budget_mm2"], sum(split.values()), cap,
        ", ".join("{} {:.2f}".format(part, split[part]) for part in split))


def refused_lines(entry):
    """What the prompts say under a design that breaks a cap: it is never the answer, and a
    move that cuts what it breaks is worth more than one that gains speed."""
    if not entry.get("violations"):
        return []
    return ["  REFUSED: " + "; ".join(entry["violations"]) + ". A refused design is never the answer: "
            "the round's design must cut what it breaks, and a move that does is worth more than one "
            "that gains speed."]


def labels_for(workloads):
    """Workloads are shown by position (W1, W2, ...), never by name: a trace's file
    name can say what the program is, and the specialist is not told."""
    return {workload: "W{}".format(index + 1) for index, workload in enumerate(workloads)}


def shape_text(level, sets, ways):
    """A level's shape as this chip builds it: the array, what it costs in silicon and in
    cycles, and - where the chip has more than one core - whether it exists once per core
    or once for all of them, which is what decides whether its capacity is paid once or
    per core and whether its counters are one core's or every core's."""
    kilobytes = sets * ways * 64 // 1024
    if not CHIP_VIEW:
        return "{} x {} = {} blocks, {} KB, hit latency {} cycles".format(
            sets, ways, sets * ways, kilobytes, chip.latency(level, sets, ways))
    return "{} x {} = {} blocks, {} KB, {:.2f} mm2, hit latency {} cycles".format(
        sets, ways, sets * ways, kilobytes,
        chip.area_mm2(level, sets, ways), chip.latency(level, sets, ways))


def level_report(entry, levels, workloads):
    """One design's per-level report, for the levels a concern can read: the shape,
    which is what fixes the hit latency paid on every access that reaches the level,
    then what each workload measured there. The levels are listed top down, so a
    level's misses are read against the line below it."""
    labels = labels_for(workloads)
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
            cell = "{} mpki {:.2f}".format(labels[workload], got("mpki"))
            per_core = got("mpki_cores") if CHIP_VIEW else None
            if per_core and len(per_core) > 1:
                cell += " (per core {:.0f} to {:.0f})".format(min(per_core), max(per_core))
            cell += ", hit {:.2f}".format(got("hit_ratio") or 0.0)
            if got("pf_coverage") is not None:
                cell += ", prefetch cov {:.2f} acc {:.2f}".format(
                    got("pf_coverage"), got("pf_accuracy") or 0.0)
            if got("miss_latency"):
                cell += ", miss {:.0f} cy".format(got("miss_latency"))
            cells.append(cell)
        lines.append("  {}: {} | {}".format(
            level, shape_text(level, sets, ways),
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


def violation_note(entry):
    """" | refused: ..." for a design that broke a cap, else nothing."""
    if not entry.get("violations"):
        return ""
    return " | refused: " + "; ".join(entry["violations"])


LEDGER_DIR = "results/ledgers"      # one live ledger per run, <tag>.jsonl, a row per design measured
SKIP_LOG = os.path.join("results", "skip_gate.jsonl")
# CALL_GATE: "shadow" calls every specialist and only logs the decider's verdicts; "apply" does not
# call a specialist the decider rates under CALL_GATE_THRESHOLD. Needs CONTEXT_GATE set for the decider.
# The ablation switches. STATISTICIAN=0 leaves the surrogate member out; FILL=0 stops untried
# one-knob moves topping the wave up; REPAIR=none drops the repair of a gain the power cap refused.
STATISTICIAN = os.environ.get("STATISTICIAN", "1") == "1"
FILL = os.environ.get("FILL", "1") == "1"
REPAIR = os.environ.get("REPAIR", "fit")
CALL_GATE = os.environ.get("CALL_GATE", "shadow")
CALL_GATE_THRESHOLD = float(os.environ.get("CALL_GATE_THRESHOLD", "0.4"))
_call_gate = None


def call_gate():
    """The CHIA call-gate block around the loop's one decider."""
    global _call_gate
    if _call_gate is None:
        from chia.models.call_gate import CallGate
        _call_gate = CallGate(analyst.decider(), threshold=CALL_GATE_THRESHOLD, mode=CALL_GATE)
    return _call_gate


def skip_verdicts(sheet, state, incumbent, tag, round_number):
    """Shadow call-gating: for each concern, the context gate's decider answers whether the sheet
    and the ledgers give evidence for a move on that concern's knobs this round. Nothing is
    skipped; the verdicts are logged beside what the specialist then did, so the precision of
    skipping a call can be read before any call is skipped. Empty when no gate is configured."""
    if not analyst.CONTEXT_GATE:
        return {}
    try:
        text = sheet_text(sheet) if sheet is not None else "(no sheet)"
        state_text = {"sheet": text[:2500], "current_design": assignment(incumbent, list(SEARCH_SPACE)),
                      "statistician_expects": (state.get("statistician") or "")[:800],
                      "mode": state["mode"]}
        questions = {}
        for name in ORDER:
            questions[name] = {
                "instructions": "Does the sheet give evidence for changing the {} knobs ({}) this round, rather than holding them?".format(
                    name, ", ".join(CONCERNS[name]["knobs"])),
                "criteria": {"true": "the sheet's bottleneck, findings or failing list point at these knobs",
                             "false": "nothing in the sheet points at these knobs; a hold costs nothing"}}
        verdicts = gate_and_log(call_gate(), state_text, questions, label="round {}".format(round_number))
        note(tag, "-> call gate ({}): ".format(CALL_GATE) + ", ".join("{} {:.2f}".format(name, verdicts[name]["probability"]) for name in ORDER))
        return verdicts
    except Exception as error:
        note(tag, "-> call gate failed: " + repr(error)[-120:])
        return {}


def consulted_specialists(verdicts, tag):
    """Which specialists to call this round: all of them in shadow mode. In apply mode a
    specialist the decider rated under the threshold is not called: that call is the one most
    likely to end in a hold, and a hold costs the call and buys nothing. The slots it leaves in
    the wave go to untried one-knob moves, exactly as when a called specialist holds."""
    if CALL_GATE != "apply" or not verdicts:
        return list(ORDER)
    consulted = []
    skipped = []
    for name in ORDER:
        if name in verdicts and not verdicts[name]["consult"]:
            skipped.append("{} ({:.2f})".format(name, verdicts[name]["probability"]))
        else:
            consulted.append(name)
    if skipped:
        note(tag, "-> call gate (apply): not consulting " + ", ".join(skipped) + "; threshold {:.2f}".format(CALL_GATE_THRESHOLD))
    return consulted


def log_skip(tag, round_number, verdicts, proposals, probes):
    """One row per concern and round: the decider's probability that a move was warranted, what
    the specialist did (held or proposed) and what its design measured."""
    if not verdicts:
        return
    with open(SKIP_LOG + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        with open(SKIP_LOG, "a") as log_file:
            for name in ORDER:
                verdict = verdicts.get(name) or {"probability": None, "consult": True}
                row = {"tag": tag, "round": round_number, "concern": name, "probability": verdict["probability"],
                       "skipped": not verdict["consult"],
                       "proposed": name in proposals,
                       "gain": (probes or {}).get("parts", {}).get(name),
                       "refused": name in (probes or {}).get("refused", {})}
                log_file.write(json.dumps(row) + "\n")
                # The same fact as a `call_gate_outcome` event in the run's profiler log, in the loop's words.
                if not verdict["consult"]:
                    outcome = "skipped by the gate"
                elif name not in proposals:
                    outcome = "held"
                else:
                    outcome = "proposed, gain {:+.4f}{}".format(row["gain"] or 0.0, ", refused by the cap" if row["refused"] else "")
                log_outcome("round {}".format(round_number), name, verdict, made=verdict["consult"], outcome=outcome)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def statistician(problem, history, incumbent_entry, pool, tag):
    """The fifth member of the team: a Gaussian-process surrogate over every design this run
    measured, proposing the candidate with the largest expected improvement over the incumbent
    from a fixed random pool plus the incumbent's one-knob neighbours. It reads the table where
    the architects read the counters. A design a cap refused trains at a value below the stock,
    so the surrogate learns where the caps bite as well as where the speed is. None when every
    candidate has been measured."""
    objective = problem["objective"]
    stock_value = history[0]["metrics"][objective]
    knobs_list = []
    values = []
    for entry in history:
        knobs_list.append(entry["knobs"])
        if entry.get("violations"):
            values.append(stock_value - 0.05)
        else:
            values.append(entry["metrics"][objective])
    measured = set()
    for entry in history:
        measured.add(problem["name_of"](entry["knobs"]))
    candidates = {}
    for name in pool:
        if name not in measured:
            candidates[name] = pool[name]
    for name, knobs in engine.neighbours(incumbent_entry["knobs"], problem).items():
        if name not in measured:
            candidates[name] = knobs
    if not candidates:
        return None
    names = list(candidates)
    rows = [candidates[name] for name in names]
    model = engine.fit_gp(knobs_list, values)
    means, spreads = engine.predict(model, rows)
    scores = engine.expected_improvement(means, spreads, incumbent_entry["metrics"][objective])
    best = 0
    for index in range(len(names)):
        if scores[index] > scores[best]:
            best = index
    runner_up = None
    for index in range(len(names)):
        if index != best and (runner_up is None or scores[index] > scores[runner_up]):
            runner_up = index
    reasoning = "the surrogate over {} measured designs predicts {:.3f} +- {:.3f} here (expected improvement {:.4f})".format(
        len(history), means[best], spreads[best], scores[best])
    # What the surrogate expects, for everyone to read: its five best candidates by expected
    # improvement (the highest predicted means are the incumbent's own neighbours at the
    # incumbent's value, which say nothing), as moves from the incumbent. Predictions to test.
    order = sorted(range(len(names)), key=lambda index: -scores[index])
    lines = []
    for index in order[:5]:
        moved = knobs_changed(rows[index], incumbent_entry["knobs"])
        lines.append("  predicted {:.3f} +- {:.3f}: {}".format(
            means[index], spreads[index],
            ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())) or "the current design"))
    view = "\n".join(lines)
    note(tag, "-> statistician: {}\n   because: {}\n   expects:\n{}".format(
        assignment(rows[best], sorted(knobs_changed(rows[best], incumbent_entry["knobs"]))), reasoning, view))
    return {"knobs": rows[best], "reasoning": reasoning, "view": view,
            "runner_up": None if runner_up is None else rows[runner_up]}


def repair_to_fit(refused_design, entry, problem, is_candidate, measured, margin=0.95, steps=8):
    """The refused design shrunk until it is estimated to fit under the power cap: at each step
    the level holding the most power (estimated from the refused design's own counters with the
    candidate's shapes, so leakage is exact and dynamic energy per access is the new shape's)
    loses one rung, sets first, else ways; the estimate is redone after every rung. The margin
    covers what the estimate cannot see: a smaller cache misses more. Returns the repaired knobs
    and the estimate, or None when no shape fits or the result is already measured."""
    cap = space.POWER_BUDGET_W
    candidate = dict(refused_design)
    estimate = space.watts(candidate, entry["metrics"], problem["workloads"])
    for _ in range(steps):
        if estimate <= cap * margin:
            break
        split = power_split(candidate, entry["metrics"], problem["workloads"])
        levels = sorted(["L1D", "L2C", "LLC"], key=lambda name: -split.get(name, 0.0))
        moved = None
        for level in levels:
            sets_knob, ways_knob = SHAPE_OF[level]
            for knob in (sets_knob, ways_knob):
                smaller = [value for value in SEARCH_SPACE[knob] if int(value) < int(candidate[knob])]
                if not smaller:
                    continue
                trial = dict(candidate, **{knob: max(smaller)})
                if within_budget(typed_knobs(trial)):
                    moved = trial
                    break
            if moved is not None:
                break
        if moved is None:
            return None
        candidate = moved
        estimate = space.watts(candidate, entry["metrics"], problem["workloads"])
    if estimate > cap * margin or candidate == refused_design:
        return None
    if not is_candidate(candidate) or problem["name_of"](candidate) in measured:
        return None
    return candidate, estimate


def repair_candidates(record, history, problem, is_candidate):
    """A design that gained but broke the power cap in the last wave comes back once, repaired
    to fit: shrunk, level by level and rung by rung, until its own counters say it is under the
    cap (one rung was not enough in three of four stalled seeds before this rule). The gain
    the cap refused is the most informative measurement of the round, and the repair is the
    coupled move the refusal itself points to. Returns {label: {"knobs": part, "reasoning": ...}}."""
    repairs = {}
    if record is None or REPAIR != "fit":
        return repairs
    probes = record.get("probes") or {}
    base = history[record["against"]]["knobs"]
    measured = set()
    for entry in history:
        measured.add(problem["name_of"](entry["knobs"]))
    for label in probes.get("refused", {}):
        reasons = " ".join(probes["refused"][label])
        gained = probes.get("parts", {}).get(label, probes.get("pairs", {}).get(label, probes.get("joint") if label == "joint" else None))
        if "power" not in reasons or gained is None or gained <= 0:
            continue
        part = (record.get("proposals") or {}).get(label, {}).get("knobs")
        if part is None:
            continue
        refused_design = with_parts(base, [part])
        entry = None
        for candidate in history:
            if problem["name_of"](candidate["knobs"]) == problem["name_of"](refused_design):
                entry = candidate
        if entry is None:
            continue
        repaired = repair_to_fit(refused_design, entry, problem, is_candidate, measured)
        if repaired is None:
            continue
        repaired_knobs, estimate = repaired
        shrunk = ", ".join("{} {}->{}".format(knob, refused_design[knob], repaired_knobs[knob])
                           for knob in sorted(repaired_knobs) if repaired_knobs[knob] != refused_design[knob])
        repairs["repair:" + label] = {"knobs": {name: repaired_knobs[name] for name in knobs_changed(repaired_knobs, base)},
                                      "reasoning": "the {} gain the power cap refused ({:.2f} W), shrunk to fit: {} (estimated {:.2f} W from its own counters)".format(
                                          label, entry["metrics"].get("watts", 0.0), shrunk, estimate)}
        if len(repairs) >= 2:
            break
    return repairs


def best_designs(history, problem, how_many=3):
    """The best feasible designs measured so far, every knob written out: what they share is
    the regime the workload rewards, which no per-knob ledger can show."""
    feasible = [entry for entry in history if not entry.get("violations")]
    top = sorted(feasible, key=lambda entry: -entry["metrics"][problem["objective"]])[:how_many]
    lines = []
    for entry in top:
        lines.append("  D{} {}={:.4f}  {}".format(
            entry["index"], problem["objective"], entry["metrics"][problem["objective"]],
            assignment(entry["knobs"], list(SEARCH_SPACE))))
    return "\n".join(lines) if lines else "  nothing measured yet"


def recent(history, problem, how_many=6):
    """The team's shared record: the last few designs and what they scored."""
    lines = []
    for entry in history[-how_many:]:
        changed = knobs_changed(entry["knobs"], problem["stock"])
        text = ", ".join("{}={}".format(knob, changed[knob][1]) for knob in changed) or "stock"
        lines.append("  D{} {}={:.4f}  {}{}".format(
            entry["index"], problem["objective"], entry["metrics"][problem["objective"]], text, violation_note(entry)))
    return "\n".join(lines)


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



# ---------------------------------------------------------------- views ----

def moves_ledger(history, objective):
    """Every move this run has made, oldest first: which concern moved what against which
    design, and what it did to the objective. The team's own case record."""
    lines = []
    for entry in history:
        if entry.get("against") is None:
            continue
        moved = entry["moved"]
        line = "  D{} {}: {} | {:+.4f} against D{}".format(
            entry["index"], entry["source"].split(":")[-1],
            ", ".join("{} {}->{}".format(knob, moved[knob][0], moved[knob][1]) for knob in sorted(moved)),
            entry["metrics"][objective] - entry["baseline"], entry["against"])
        probes = entry.get("probes")
        if probes:
            line += " | sketch predicted {:+.4f}{}".format(
                probes["predicted"], "" if probes["exact"] else " (sum of parts)")
            if len(probes["parts"]) > 1:
                line += ", parts alone {}, together {:+.4f}, interaction {:+.4f}".format(
                    ", ".join("{} {:+.4f}".format(name, delta) for name, delta in probes["parts"].items()),
                    probes["joint"], probes["interaction"])
        lines.append(line + violation_note(entry))
    return "\n".join(lines) if lines else "  no move yet"


def concern_moves(history, knobs, objective):
    """The move ledger restricted to the moves that touched these knobs: the concern's own
    case record, deltas included."""
    lines = []
    for entry in history:
        if entry.get("against") is None or not any(knob in entry["moved"] for knob in knobs):
            continue
        moved = entry["moved"]
        lines.append("  D{} {}: {} | {:+.4f} against D{}".format(
            entry["index"], entry["source"].split(":")[-1],
            ", ".join("{} {}->{}".format(knob, moved[knob][0], moved[knob][1]) for knob in sorted(moved)),
            entry["metrics"][objective] - entry["baseline"], entry["against"]))
    return "\n".join(lines) if lines else "  none yet"


def value_ledger(history, knobs, workloads, objective):
    """What each value of each knob has done across every design of the run: how many
    designs carried it, the best of them, and for a prefetcher the mean coverage and
    accuracy it reached at its level. Cross-design evidence per value, which the
    assignment ledger cannot show."""
    lines = []
    for knob in knobs:
        level = LEVEL_OF[knob.split("_")[0]]
        seen = {}
        for entry in history:
            stats = seen.setdefault(str(entry["knobs"][knob]), {"n": 0, "sum": 0.0, "cov": [], "acc": [], "best": None, "watts": []})
            stats["n"] += 1
            stats["sum"] += entry["metrics"][objective]
            if entry["metrics"].get("watts") is not None:
                stats["watts"].append(entry["metrics"]["watts"])
            if knob.endswith("_prefetcher"):
                for workload in workloads:
                    coverage = entry["metrics"].get("{}:{}_pf_coverage".format(workload, level))
                    if coverage is not None:
                        stats["cov"].append(coverage)
                        stats["acc"].append(entry["metrics"].get("{}:{}_pf_accuracy".format(workload, level)) or 0.0)
            value = entry["metrics"][objective]
            if stats["best"] is None or value > stats["best"]:
                stats["best"] = value
        total_n = len(history)
        total_sum = sum(entry["metrics"][objective] for entry in history)
        cells = []
        for value in sorted(seen, key=lambda v: -seen[v]["best"]):
            stats = seen[value]
            # Mean with the value against the mean without it: the cheapest cross-design signal
            # there is, and the one a surrogate model reads first.
            without = ""
            if total_n > stats["n"]:
                without = " vs {:.3f} without".format((total_sum - stats["sum"]) / (total_n - stats["n"]))
            cell = "{}: {} design{}, best {:.4f}, mean {:.3f}{}".format(
                value, stats["n"], "" if stats["n"] == 1 else "s", stats["best"], stats["sum"] / stats["n"], without)
            if stats["cov"]:
                cell += ", cov {:.2f} acc {:.2f}".format(
                    sum(stats["cov"]) / len(stats["cov"]), sum(stats["acc"]) / len(stats["acc"]))
            if stats["watts"] and chip.card()["power_budget_w"] is not None:
                cell += ", mean {:.2f} W".format(sum(stats["watts"]) / len(stats["watts"]))
            cells.append(cell)
        lines.append("  {}: {}".format(knob, " | ".join(cells)))
    return "\n".join(lines)


VERDICTS = ["capacity-bound", "latency-heavy", "prefetch-uncovered", "prefetch-polluting", "bandwidth-bound", "fine"]


def memory_time(entry, workload):
    """Where a thousand instructions' memory time goes, as shares of the total: each
    level's demand hits pay that level's hit latency, and the last level's demand misses
    pay the DRAM service the counters measured. Each level's own hits, not the difference
    between two levels' misses: a miss that merges into a miss already outstanding is
    counted as a miss at its level and never reaches the level below, and on a streaming
    workload three misses in four merge, so the difference credits the level below with
    hits it never served. Overlap between outstanding misses is not modelled, so this
    ranks the tiers; it does not account for cycles."""
    metrics = entry["metrics"]

    def got(key):
        return metrics.get("{}:{}".format(workload, key))

    instructions = got("instructions")
    if not instructions:
        return None
    tiers = {}
    for level in ALL_LEVELS:
        hits = got(level + "_hits")
        if hits is None:
            return None
        sets_knob, ways_knob = SHAPE_OF[level]
        tiers[level] = hits * 1000.0 / instructions * chip.latency(
            level, entry["knobs"][sets_knob], entry["knobs"][ways_knob])
    tiers["DRAM"] = (got("LLC_misses") or 0) * 1000.0 / instructions * (got("LLC_miss_latency") or 0.0)
    total = sum(tiers.values())
    return {tier: tiers[tier] / total for tier in tiers} if total else None


def off_chip_bytes_per_cycle(entry, workload):
    """The traffic that actually reaches memory, in bytes per chip cycle, counted at the
    memory controller. A level's miss count is not this number: a prefetch that merges
    into an outstanding miss is counted as a miss and never leaves the chip, and on a
    design with a deep prefetcher most of them do."""
    metrics = entry["metrics"]

    def got(key):
        return metrics.get("{}:{}".format(workload, key)) or 0

    requests = (got("dram_rq_row_buffer_hit") + got("dram_rq_row_buffer_miss")
                + got("dram_wq_row_buffer_hit") + got("dram_wq_row_buffer_miss"))
    cycles = metrics.get("{}:cycles".format(workload))
    if not cycles or not requests:
        return None
    return requests * 64.0 / cycles


def derived_view(entry, workloads, levels=ALL_LEVELS):
    """The quantities an architect reasons with, computed from the counters and this
    chip's own numbers. Per workload: what share of the accesses reaching a level miss it
    and how many of the level above's misses miss again; where the memory time goes;
    what reaches memory against what the channels can deliver; what a miss costs against
    what it would cost on an idle memory; the row-buffer hit rate; and, where the chip
    has more than one core, what each core got."""
    labels = labels_for(workloads)
    card = chip.card()
    lines = []
    for workload in workloads:
        metrics = entry["metrics"]

        def got(key):
            return metrics.get("{}:{}".format(workload, key))

        kilo = (got("instructions") or 0) / 1000.0
        if not kilo:
            continue
        cells = []
        above_misses = None
        for level in ALL_LEVELS:
            hits, misses = got(level + "_hits"), got(level + "_misses")
            if hits is None or misses is None:
                continue
            # What leaves a level is its misses less the ones that merged into a miss
            # already outstanding; only those reach the level below.
            leaving = misses - (got(level + "_merges") or 0)
            if level not in levels:
                above_misses = leaving
                continue
            accesses = hits + misses
            cell = "{}: miss share {:.2f}".format(level, misses / accesses if accesses else 0.0)
            if above_misses:
                cell += ", of the level above's misses {:.2f} miss again".format(min(misses / above_misses, 9.99))
            issued, useless = got(level + "_pf_issued") or 0, got(level + "_pf_useless") or 0
            if issued:
                cell += ", prefetch issued {:.1f}/ki useless {:.1f}/ki".format(issued / kilo, useless / kilo)
            if not CHIP_VIEW:
                sets_knob, ways_knob = SHAPE_OF[level]
                ratio = hits / accesses if accesses else 0.0
                cell += ", {:.1f} hit cycles per useful hit".format(
                    chip.latency(level, entry["knobs"][sets_knob], entry["knobs"][ways_knob]) / ratio
                    if ratio else float("inf"))
            cells.append(cell)
            above_misses = leaving
        if not CHIP_VIEW:
            # The figure the council read before the controller's own counts were used: it
            # counts prefetch misses that merge into an outstanding miss and never leave the
            # chip, which on a deep prefetcher is most of them.
            leaving = (got("LLC_misses") or 0) + (got("LLC_pf_misses") or 0)
            cells.append("off-chip {:.1f} KB/ki".format(leaving * 64 / 1024 / kilo))
            row_hit, row_miss = got("dram_rq_row_buffer_hit") or 0, got("dram_rq_row_buffer_miss") or 0
            if row_hit + row_miss:
                cells.append("DRAM row-buffer hit {:.2f}".format(row_hit / (row_hit + row_miss)))
            lines.append("  {} | {}".format(labels[workload], " | ".join(cells)))
            continue
        shares = memory_time(entry, workload)
        if shares:
            cells.append("memory time " + " ".join(
                ["{} hits {:.0f}%".format(level, 100 * shares[level]) for level in ALL_LEVELS]
                + ["DRAM {:.0f}%".format(100 * shares["DRAM"])]))
        if card["power_budget_w"] is not None:
            split = power_split(entry["knobs"], metrics, [workload])
            total = sum(split.values())
            cells.append("power " + " ".join("{} {:.0f}%".format(part, 100 * split[part] / total) for part in split))
        traffic = off_chip_bytes_per_cycle(entry, workload)
        if traffic:
            peak = card["dram_peak_bytes_per_cycle"]
            cells.append("off-chip {:.2f} of the {:.1f} B/cycle the channels deliver ({:.0f}%)".format(
                traffic, peak, 100 * traffic / peak))
        if got("LLC_miss_latency"):
            cells.append("a miss past the last level costs {:.0f} cy, against the {:.0f} a DRAM access "
                         "takes with the channels idle ({:.0f} if its row is already open)".format(
                             got("LLC_miss_latency"), card["dram_row_miss_cycles"], card["dram_row_hit_cycles"]))
        row_hit, row_miss = got("dram_rq_row_buffer_hit") or 0, got("dram_rq_row_buffer_miss") or 0
        if row_hit + row_miss:
            cells.append("DRAM row-buffer hit {:.2f}".format(row_hit / (row_hit + row_miss)))
        lines.append("  {} | {}".format(labels[workload], " | ".join(cells)))
    return "\n".join(lines)


def workload_view(entry, history, workloads):
    """What each workload is contributing and how much is left in it: its IPC in this
    design, and how far that is below the best this run has measured for that workload
    alone. The objective is a geometric mean, so it follows whichever is furthest back."""
    labels = labels_for(workloads)
    values = {workload: entry["metrics"].get(workload + ":ipc") for workload in workloads}
    present = [workload for workload in workloads if values[workload] is not None]
    if not present:
        return "  nothing measured yet"
    lowest = min(present, key=lambda workload: values[workload])
    cells = []
    for workload in present:
        best = max([row["metrics"].get(workload + ":ipc") or 0.0 for row in history] + [values[workload]])
        behind = ("at the best this run has measured for it" if values[workload] >= best - 1e-9
                  else "{:.0f}% below the best this run has measured for it".format(
                      100.0 * (best - values[workload]) / best))
        floor = ""
        if REGRESSION_TOLERANCE is not None and history:
            # The floor under this workload: the stock's IPC on it, which no design may go below.
            floor_value = history[0]["metrics"][workload + ":ipc"] * (1.0 - REGRESSION_TOLERANCE)
            floor = ", floor {:.3f} ({:+.1f}%)".format(floor_value, 100.0 * (values[workload] / floor_value - 1.0))
        cells.append("{} ipc {:.3f}{}, {}{}".format(
            labels[workload], values[workload],
            " (the lowest; the objective follows it)" if workload == lowest else "", behind, floor))
    return "  " + " | ".join(cells)


def last_round_section(problem, history, incumbent, levels):
    """The design measured last round, if it lost: what moved and its report."""
    objective = problem["objective"]
    return ["",
            "## The design measured last round (D{}, {}={:.4f})".format(
                history[-1]["index"], objective, history[-1]["metrics"][objective]),
            "  it moved: " + (", ".join("{} {}->{}".format(knob, moved_from, moved_to)
                                        for knob, (moved_from, moved_to)
                                        in sorted(knobs_changed(history[-1]["knobs"], incumbent).items()))
                              or "nothing"),
            level_report(history[-1], levels, problem["workloads"])]


def knob_section(current):
    """Every knob and the values this chip can take with the rest of the design as it
    stands, the prefetch placement table, and this chip's own capacity ladder."""
    lines = ["## Knobs and the values available (a geometry this chip's area budget forbids is left out)"
             if CHIP_VIEW else "## Knobs and allowed values"]
    lines += ["- {}: {}".format(knob, json.dumps(feasible_values(knob, current))) for knob in SEARCH_SPACE]
    lines += ["Prefetch mechanisms and the levels that accept each:", placement_table(CONCERNS["prefetch"]["knobs"]),
              "What a capacity costs at each level on this chip, in cycles and in silicon:"
              if CHIP_VIEW else "Capacity and the hit latency it costs:",
              latency_ladder(ALL_LEVELS, current),
              "A level must hold more than the level in front of it (L1D < L2 < LLC in KB) and answer no faster; "
              "a design that breaks this, or the silicon cap, or leaks more than the power cap, is not built."]
    return lines


# One principle each where power changes the decision; read only where the chip caps power,
# because only there does the prompt carry the power split it refers to.
POWER_PRINCIPLES = {
    "prefetch": "A prefetch that goes unused pays its level's energy per access and, past the last "
                "level, the off-chip cost of the line; useless issues show in the power split as "
                "well as in the bandwidth.",
    "geometry": "Every access to a level pays that level's energy, and every instance leaks for the "
                "whole run: a level with a low hit ratio spends power as well as latency for "
                "nothing, and a large level leaks whether or not it hits.",
}

OVER_BUDGET_KEY = """
- "over_budget": an object {"cost": one of silicon, power, "driver": one sentence naming the level
  or the traffic that holds most of it, "number": the headroom left before the cap, in mm2 or W}:
  the cost nearest its cap in the cost line. Headroom is room to spend on speed, not a thing to
  save: the objective is the ipc, and a cap only says where a design may not go."""

# ---------------------------------------------------------------- the diagnosis ----

RANK_BY_TIME = """Rank the levels by the share of the
  memory time they hold in the derived view, not by how often they miss: the tier holding the most
  time is where the objective is lost, and it is never fine, whatever its misses per
  kilo-instruction or its hit ratio; its verdict names what would cut that time. A level's share is
  the time spent serving its hits, which is cut by making the hits cheaper or by moving them to a
  cheaper level; DRAM's share is time waiting for memory, which is cut by removing the traffic."""

DRAM_IS_A_TIER = """ DRAM is a tier of that same ranking: when it
  holds the most memory time it is the bottleneck, and the evidence is what reaches the channels
  against what they deliver, and what a miss costs against an idle memory."""

RANK_BY_MISSES = """Rank the levels by misses per
  kilo-instruction first: the level that misses most is where the objective is lost, and it is
  never fine, whatever its hit ratio; its verdict names what would cut those misses."""

DIAGNOSIS_TASK = """## Task
Write the sheet for the current design. You choose no knob and you name no knob value: the
specialists decide, from what you establish. Read every design measured, not only the last, and
read the counters, not only the objective. One design is one measurement: write what it shows, not
what it confirms. A design whose outcome cannot be assigned to one knob because several moved
together teaches nothing about any one of them: say so, and draw no per-knob lesson from it.
A counter is a proxy; a measured change in the objective is the fact. Where they disagree - a
prefetcher at 0.00 accuracy whose removal lost 0.2 of ipc - the measurement wins, and the counter
is noted as unreliable at that level. Power matters only against its cap: the cost line says how
close the current design is, and a move that gives up ipc to save power the design does not need
is a loss.

Return JSON with exactly these keys:
- "levels": an object with one entry each for L1D, L2C and LLC: {"verdict": one of capacity-bound,
  latency-heavy, prefetch-uncovered, prefetch-polluting, bandwidth-bound, fine; "number": the one
  figure from the report or the derived view that says so}. {ranking}
- "bottleneck": an object {"level": one of L1D, L2C, LLC, DRAM, "cause": one sentence naming the
  mechanism, "evidence": the numbers that show it}.{dram}{budget}
- "learned": a list of up to six strings. Each states one finding with its numbers: the knob or
  level, what moved, the change in the objective with its sign and the design ids, and the counter
  that explains it - of the form "L2 halved: +0.002 (D3 vs D1); L2 hit ratio unchanged at 0.28,
  hit latency 9 -> 7 cycles". An entry without a number is not a finding.
- "unattributable": a list of strings, one per design whose outcome cannot be assigned to one knob,
  naming the knobs that moved together; empty if none.
- "failing": a list of strings: the knobs whose current value is not doing its job by its own
  counters - a prefetcher covering few misses at low accuracy, a level charging its hit latency
  for a low hit ratio, a policy that changed nothing - each with the number that says so and how
  many of its alternatives this run has measured at that level. A knob whose alternatives this
  run measured and lost is not failing, whatever its counters say. Knobs and levels, not the
  value to pick.
- "hypothesis": one sentence predicting what the current design's report will show changed after
  the next move, in report numbers. A prediction, not an experiment to run.
"""

OPENING_TASK = """
This is the first round and the current design is the stock chip. Besides the sheet, propose the
opening design: the one design measured before the specialists take turns, so it should carry
every move the stock report's evidence already supports, across as many concerns as the evidence
reaches, and leave every other knob at its stock value. The specialists refine from it one concern
at a time.

The design must fit the chip's budget as the Chip section states it, or it is not measured at all.

Add to the JSON:
- "opening": an object with every knob and one allowed value each.
- "opening_reasoning": one sentence per move, what evidence it acts on.
"""

ANALYST_DESIGN_TASK = """- "design": the whole design you would measure this round beside the specialists' moves: an
  object with every knob and one allowed value each. Here you name values. Write the design the
  evidence points to, not one notch from the current one: the best designs measured in full say
  which regime this workload rewards, the refusals say what broke a cap and where the power went,
  the ladder prices every shape in silicon and leakage, and a design over the silicon cap or
  leaking more than the power cap is dropped unmeasured. Move as many knobs as the evidence
  supports, across concerns.
- "design_reasoning": one sentence per knob moved, what evidence it acts on.
"""

def sheet_text(sheet):
    lines = ["  {}: {} ({})".format(level, sheet["levels"][level]["verdict"], sheet["levels"][level]["number"])
             for level in ALL_LEVELS]
    lines += ["  bottleneck at {}: {}".format(sheet["bottleneck"]["level"], sheet["bottleneck"]["cause"]),
              "  evidence: " + sheet["bottleneck"]["evidence"]]
    for key in ["learned", "unattributable", "failing"]:
        lines.append("  {}:".format(key) + ("" if sheet[key] else " none"))
        lines += ["    - " + item for item in sheet[key]]
    lines.append("  hypothesis: " + sheet["hypothesis"])
    if sheet.get("over_budget"):
        over = sheet["over_budget"]
        lines.append("  cap headroom: {} - {} ({})".format(over["cost"], over["driver"], over["number"]))
    return "\n".join(lines)


def previous_section(previous, history, objective):
    """The hypothesis written last round and what was measured since, so the analyst checks
    its prediction before writing the next sheet. Only the hypothesis is carried over: a
    whole previous sheet invited copying it."""
    last = history[-1]
    if last.get("against") is None:
        measured = "  nothing was measured since (the specialist held)"
    else:
        moved = last["moved"]
        measured = "  measured since: D{} {} moved {} | {:+.4f} against D{}".format(
            last["index"], last["source"].split(":")[-1],
            ", ".join("{} {}->{}".format(knob, moved[knob][0], moved[knob][1]) for knob in sorted(moved)),
            last["metrics"][objective] - last["baseline"], last["against"])
    return ["", "## Your previous hypothesis, and what happened", "  " + previous["hypothesis"], measured]


def diagnosis_prompt(problem, history, incumbent_entry, opening=False, previous=None, rounds=(), state=None, evaluated=None):
    """The analyst's prompt for one round. Kept apart from the call so it can be rendered
    without one (`preview`)."""
    objective = problem["objective"]
    parts = ["## Role",
             "You are the analyst of a team tuning one cache hierarchy. Four specialists own the knobs, "
             "one concern each; you own none and you propose no value. You read every design this run "
             "has measured and write the sheet the specialists read: what the measurements have "
             "established with its numbers, what they cannot tell apart, and which knobs are failing by "
             "their own counters.",
             "",
             "## Chip", chip_line(problem),
             "Objective: maximise {}, the geometric mean of IPC over {}.".format(
                 objective, " + ".join(labels_for(problem["workloads"]).values())),
             "Concerns and their knobs: " + "; ".join(
                 "{} ({})".format(name, ", ".join(CONCERNS[name]["knobs"])) for name in ORDER) + ".",
             ""]
    parts += knob_section(incumbent_entry["knobs"])
    parts += ["",
              current_heading(incumbent_entry, objective),
              "  knobs:  " + assignment(incumbent_entry["knobs"], list(SEARCH_SPACE)),
              cost_line(incumbent_entry, problem)]
    parts += refused_lines(incumbent_entry)
    parts += [level_report(incumbent_entry, ALL_LEVELS, problem["workloads"]),
              ("  derived from the counters and this chip:" if CHIP_VIEW else "  derived from the counters:"),
              derived_view(incumbent_entry, problem["workloads"])]
    if CHIP_VIEW:
        parts += ["", "## What each workload contributes",
                  workload_view(incumbent_entry, history, problem["workloads"])]
    if history[-1]["index"] != incumbent_entry["index"]:
        parts += last_round_section(problem, history, incumbent_entry["knobs"], ALL_LEVELS)
        parts += ["  derived from the counters:", derived_view(history[-1], problem["workloads"])]
    if previous is not None:
        parts += previous_section(previous, history, objective)
    parts += ["",
              "## Every move this run has made",
              moves_ledger(history, objective),
              "",
              "## What each round's wave measured (each design's gain against that round's incumbent; a design a cap refused says why)",
              sketch_ledger(rounds, objective),
              "",
              "## What each value of each knob has done across the run",
              value_ledger(history, list(SEARCH_SPACE), problem["workloads"], objective)]
    parts += ["",
              "## The team's recent designs",
              recent(history, problem),
              "",
              "## The best designs measured so far, in full",
              best_designs(history, problem)]
    parts += statistician_section(state)
    task = DIAGNOSIS_TASK.replace("{ranking}", RANK_BY_TIME if CHIP_VIEW else RANK_BY_MISSES)
    task = task.replace("{dram}", DRAM_IS_A_TIER if CHIP_VIEW else "")
    task = task.replace("{budget}", OVER_BUDGET_KEY if chip.card()["power_budget_w"] is not None else "")
    if state is not None:
        task += ANALYST_DESIGN_TASK
    parts += ["", task + (OPENING_TASK if opening else "")]
    return "\n".join(parts)


def diagnose(problem, history, incumbent_entry, tag, opening=False, previous=None, rounds=(), state=None, evaluated=None):
    """One call a round: the bottleneck sheet every specialist reads, in round 1 the opening
    design, and the analyst's own design for the round. It reads its previous sheet and what
    was measured since. `incumbent_entry` is the design the round refines from. None if the
    call fails or the answer is not a sheet; the round then runs without one."""
    prompt_text = diagnosis_prompt(problem, history, incumbent_entry, opening, previous, rounds, state, evaluated)
    note(tag, "\n---------------- diagnosis{} ----------------\n{}".format(
        " + opening" if opening else "", prompt_text))
    try:
        answer = analyst.ask(prompt_text, "analyst")
    except Exception as error:
        print("[{}] diagnosis failed ({})".format(tag, repr(error)[-120:]), flush=True)
        note(tag, "-> call failed: " + repr(error)[-160:])
        return None
    if isinstance(answer, list) and answer:
        answer = answer[0]
    if not isinstance(answer, dict) or not isinstance(answer.get("bottleneck"), dict):
        note(tag, "-> not a sheet: " + json.dumps(answer)[:300])
        return None
    levels = answer.get("levels") if isinstance(answer.get("levels"), dict) else {}

    def items(key, limit):
        value = answer.get(key)
        if isinstance(value, str):
            value = [value]
        return [str(item)[:300] for item in value[:limit]] if isinstance(value, list) else []

    sheet = {"levels": {level: {"verdict": str((levels.get(level) or {}).get("verdict", "?"))[:40],
                                "number": str((levels.get(level) or {}).get("number", ""))[:200]}
                        for level in ALL_LEVELS},
             "bottleneck": {key: str(answer["bottleneck"].get(key, ""))[:400] for key in ["level", "cause", "evidence"]},
             "learned": items("learned", 6), "unattributable": items("unattributable", 8),
             "failing": items("failing", 8),
             "hypothesis": str(answer.get("hypothesis", ""))[:400]}
    # A knob measured alone in some round is judged by its measurements, not by its counters: a
    # "failing" entry naming it is dropped (new4 kept calling prefetchers failing by their zero
    # counters after their removal had measured -0.16).
    tested = set()
    for record in rounds:
        for moved in (record.get("moves") or {}).values():
            if len(moved) == 1:
                tested.add(next(iter(moved)))
    sheet["failing"] = [item for item in sheet["failing"] if not any(knob in item for knob in tested)]
    if isinstance(answer.get("over_budget"), dict):
        sheet["over_budget"] = {key: str(answer["over_budget"].get(key, ""))[:300]
                                for key in ["cost", "driver", "number"]}
    note(tag, "-> sheet:\n" + sheet_text(sheet))
    if opening:
        sheet["opening"] = answer.get("opening")
        reasoning = answer.get("opening_reasoning", "")
        if isinstance(reasoning, list):
            reasoning = " ".join(str(item) for item in reasoning)
        sheet["opening_reasoning"] = str(reasoning)[:600]
    if state is not None:
        sheet["design"] = answer.get("design")
        sheet["design_reasoning"] = str(answer.get("design_reasoning", ""))[:600]
    return sheet


def opening_design(sheet, problem, tag, base=None, key="opening"):
    """The analyst's design: every knob at an allowed value, fitted to the budget, and
    different from the base, the stock chip in round 1 and the incumbent on the analyst's
    turn. None otherwise, and round 1 opens on the
    stock chip."""
    what = "opening" if base is None else "analyst" if key == "opening" else key
    base = problem["stock"] if base is None else base
    proposed = sheet.get(key) if sheet else None
    if not isinstance(proposed, dict):
        note(tag, "-> no {} design".format(what))
        return None
    design = {}
    for knob in SEARCH_SPACE:
        value = proposed.get(knob, base[knob])
        if str(value) not in [str(option) for option in SEARCH_SPACE[knob]]:
            note(tag, "-> {} off-contract at {}={}, dropped".format(what, knob, value))
            return None
        design[knob] = value
    design = typed_knobs(design)
    if not within_budget(design):
        note(tag, "-> the {} design needs {}, dropped".format(what, budget_text(design)))
        return None
    if not knobs_changed(design, base):
        note(tag, "-> the {} design is the current one".format(what))
        return None
    note(tag, "-> {}: {}\n   because: {}".format(what,
        ", ".join("{}={}".format(knob, design[knob]) for knob in sorted(knobs_changed(design, base))),
        sheet[key + "_reasoning"]))
    return design


# ---------------------------------------------------------------- the specialist ----

TASK = """## Task
Every specialist proposes this round. Each proposal is measured alone on the current design in
this round's wave, and the analyst's composition of the proposals is measured beside them.
Propose the move for your knobs that the evidence supports - one knob, or several of yours
together when one mechanism needs them; if it depends on another concern's move - a smaller level
that needs timely prefetch in front of it, a policy that pays only behind a prefetcher - say so in
your reasoning. Move as far as your levels' evidence supports in one design rather than one notch
at a time; the next round tells you if it was too far. A measured change in the objective
outranks a counter: a mechanism whose removal lost ipc is working, whatever its accuracy counter
says, and power matters only against its cap (the cost line says how close the design is). A gain
the power cap refused is not dead: it needs the move that frees the power it draws, and the
refusal says which level holds the power; if that level is yours, propose the shape or the
mechanism that frees it and name the refused gain you make room for. Read the best designs
measured in full: what they share is evidence about the regime this workload rewards. Hold
(return the current values) when the evidence at your levels shows nothing worth a simulation.

Do not re-propose an assignment of your knobs that this run has already measured below the
current design, unless that row says other knobs differed and those differences change your
reasoning.

Return JSON with exactly these keys:
- "reasoning": one sentence - the mechanism you are acting on and what it depends on, or why you
  are holding.
- "knobs": an object with exactly your knobs and one allowed value each.
"""


def specialist_prompt(problem, name, history, incumbent, incumbent_entry, sheet, rounds=(),
                      state=None, evaluated=None):
    """One specialist's prompt for one round. Kept apart from the call so it can be
    rendered without one (`preview`)."""
    concern = CONCERNS[name]
    objective = problem["objective"]
    parts = ["## Role",
             "You are the {} specialist in a team tuning one cache hierarchy.".format(name),
             "You choose only these knobs: " + ", ".join(concern["knobs"]) + ".",
             "Other specialists own every other knob and propose beside you this round.",
             "",
             "## Principles"]
    parts += ["- " + line for line in concern["principles"]]
    if name in POWER_PRINCIPLES and chip.card()["power_budget_w"] is not None:
        parts.append("- " + POWER_PRINCIPLES[name])
    parts += ["",
              "## Chip", chip_line(problem),
              "Objective: maximise {}, the geometric mean of IPC over {}.".format(
                  objective, " + ".join(labels_for(problem["workloads"]).values())),
              "",
              "## Your knobs and the values available (a geometry this chip's area budget forbids is left out)"
              if CHIP_VIEW else "## Your knobs and their allowed values"]
    if name == "prefetch":
        parts += ["One mechanism per level. The same mechanisms serve several levels, so this is one",
                  "placement decision across the levels, not one choice per level.",
                  placement_table(concern["knobs"])]
    else:
        parts += ["- {}: {}".format(knob, json.dumps(feasible_values(knob, incumbent)))
                  for knob in concern["knobs"]]
    if name == "geometry":
        parts += ["", "What a capacity costs at your levels on this chip, in cycles and in silicon:"
                  if CHIP_VIEW else "Capacity and the hit latency it costs:",
                  latency_ladder(concern["levels"], incumbent)]
        parts += ["", "What this run has measured at each capacity (what the capacity removes, "
                      "against what the ladder above says it costs):",
                  capacity_ladder(evaluated if evaluated is not None else history,
                                  concern["levels"], problem["workloads"], objective, incumbent)]
    parts += ["",
              "## The current design (the best this run has measured)",
              "  yours:  " + assignment(incumbent, concern["knobs"]),
              cost_line(incumbent_entry, problem)]
    parts += refused_lines(incumbent_entry)
    parts += [level_report(incumbent_entry, concern["levels"], problem["workloads"]),
              ("  derived from the counters and this chip:" if CHIP_VIEW else "  derived from the counters:"),
              derived_view(incumbent_entry, problem["workloads"], concern["levels"])]
    if CHIP_VIEW:
        parts += ["", "## What each workload contributes",
                  workload_view(incumbent_entry, history, problem["workloads"])]
    if sheet is not None:
        parts += ["",
                  "## The analyst's sheet, written this round from every design measured",
                  sheet_text(sheet)]
    if history[-1]["index"] != incumbent_entry["index"]:
        parts += last_round_section(problem, history, incumbent, concern["levels"])
        parts.insert(len(parts) - 1, "  yours:  " + assignment(history[-1]["knobs"], concern["knobs"]))
    parts += ["",
              "## What your knobs have scored",
              measured_view(history, concern["knobs"], objective, incumbent),
              "",
              "## What each value of your knobs has done across the run",
              value_ledger(history, concern["knobs"], problem["workloads"], objective)]
    parts += ["",
              "## Every move of your concern this run has made, and what it did",
              concern_moves(history, concern["knobs"], objective),
              "",
              "## What your concern's designs measured in each round's wave",
              concern_sketches(rounds, name),
              "",
              "## The team's recent designs",
              recent(history, problem),
              "",
              "## The best designs measured so far, in full",
              best_designs(history, problem)]
    parts += statistician_section(state)
    parts += ["", TASK]
    return "\n".join(parts)


def specialist(problem, name, history, incumbent, incumbent_entry, tag, sheet, log, rounds=(), state=None, evaluated=None):
    """The first version's specialist prompt with the analyst's sheet after the current
    design's report, the derived view of its levels, its concern's moves with their deltas
    and what each value of its knobs has done. One proposal for its own knobs, or None.
    Transcript lines go to `log`, so four specialists can answer at once and be written
    down in order."""
    concern = CONCERNS[name]
    prompt_text = specialist_prompt(problem, name, history, incumbent, incumbent_entry, sheet,
                                    rounds, state, evaluated)
    log.append("\n---------------- {} specialist ----------------\n{}".format(name, prompt_text))
    asked_for_area = False
    for attempt in range(3):
        try:
            answer = analyst.ask(prompt_text, name)
        except Exception as error:
            print("[{}] {} specialist failed ({})".format(tag, name, repr(error)[-120:]), flush=True)
            log.append("-> call failed: " + repr(error)[-160:])
            return None
        proposed = read_proposal(answer, concern["knobs"])
        if proposed is None:
            print("[{}] {} specialist answered off-contract, one retry".format(tag, name), flush=True)
            log.append("-> off-contract, retrying: " + json.dumps(answer)[:300])
            if attempt == 2:
                break
            continue
        held = proposed["knobs"] == {knob: incumbent[knob] for knob in concern["knobs"]}
        levels = levels_moved(proposed["knobs"], incumbent)
        log.append("-> {}{}\n   because: {}".format(
            assignment(proposed["knobs"], concern["knobs"]), "   (held)" if held else "",
            proposed["reasoning"]))
        candidate = with_parts(incumbent, [proposed["knobs"]])
        if not within_budget(candidate):
            if asked_for_area:
                log.append("-> still over the chip's budget ({}); taken as a hold".format(budget_text(candidate)))
                return None
            asked_for_area = True
            log.append("-> over the chip's budget ({}); asked once more".format(budget_text(candidate)))
            prompt_text += ("\n\n## Once more\nYour proposal needs {}, more than this chip has. "
                            "Propose one that fits.".format(budget_text(candidate)))
            continue
        if len(levels) > 1:
            log.append("-> moves {} levels ({}); taken as proposed".format(len(levels), ", ".join(levels)))
        return proposed
    log.append("-> off-contract, no proposal")
    return None


def levels_moved(proposal_knobs, incumbent):
    """The cache levels a proposal changes."""
    return sorted({LEVEL_OF[knob.split("_")[0]] for knob, value in proposal_knobs.items()
                   if str(value) != str(incumbent[knob])})



# ---------------------------------------------------------------- the sketches ----

def parts_of(design, incumbent):
    """The design's difference from the incumbent, split by concern."""
    changed = knobs_changed(design, incumbent)
    parts = {}
    for name in ORDER:
        part = {knob: design[knob] for knob in CONCERNS[name]["knobs"] if knob in changed}
        if part:
            parts[name] = part
    return parts


def with_parts(incumbent, parts):
    """The incumbent with these parts applied. Nothing is rewritten to fit the area
    budget: a design that does not fit is refused where it was proposed, so a move in the
    ledger is always the move that was made."""
    design = dict(incumbent)
    for part in parts:
        design.update(part)
    return typed_knobs(design)


def concern_sketches(rounds, name):
    """One concern's proposals round by round and what their sketches said, including the
    rounds where nothing was committed."""
    lines = []
    for record in rounds:
        probes = record.get("probes")
        moved = (record.get("moves") or {}).get(name)
        if not probes or not moved or name not in probes["parts"]:
            continue
        reason = probes.get("refused", {}).get(name)
        lines.append("  round {} against D{}: {} | measured {:+.4f} | {}".format(
            record["round"], record["against"],
            ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())),
            probes["parts"][name],
            "REFUSED by a cap: " + "; ".join(reason) if reason
            else "the new incumbent" if name in record["include"] else "not the best of its wave"))
    return "\n".join(lines) if lines else "  none yet"


def sketch_ledger(rounds, objective):
    """Every round's sketches, committed or not: which move each concern proposed, what its
    sketch said alone, what all of them said together, and what was committed. The sketches
    of a round that committed nothing are the only record that those moves lost."""
    lines = []
    for record in rounds:
        probes = record.get("probes")
        if not probes or not record.get("moves"):
            continue
        cells = []
        for name, moved in record["moves"].items():
            if name in probes["parts"]:
                # A sweep part is named by its move; the move alone says it.
                cells.append("{}{} {:+.4f}{}".format(
                    name + " " if name in CONCERNS or name in ("analyst", "statistician") else "",
                    ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())),
                    probes["parts"][name],
                    " REFUSED (" + "; ".join(probes["refused"][name]) + ")" if name in probes.get("refused", {}) else ""))
        for label, delta in probes.get("pairs", {}).items():
            cells.append("{} together {:+.4f}{}".format(
                label.replace("+", " and "), delta,
                " REFUSED (" + "; ".join(probes["refused"][label]) + ")" if label in probes.get("refused", {}) else ""))
        if "joint" in probes:
            cells.append("all together {:+.4f}{}".format(
                probes["joint"], " REFUSED (" + "; ".join(probes["refused"]["joint"]) + ")" if "joint" in probes.get("refused", {}) else ""))
        lines.append("  round {} against D{}: {} | new incumbent: {}".format(
            record["round"], record["against"], " | ".join(cells),
            " + ".join(record["include"]) if record["include"] else "none"))
    return "\n".join(lines) if lines else "  nothing measured yet"


def refused_moves(rounds, indices):
    """The moves sketched against any design in `indices` and not committed, with what the
    sketch said: the tabu list of a stall, and where the flat moves - free parts of a coupled
    design - are read from. `indices` is the current design and every design as good as it
    within the flat tolerance: a gain of a hair makes a new incumbent, and a move that lost
    against the design a hair below it has been measured."""
    refused = []
    for record in rounds:
        probes = record.get("probes")
        if record["against"] not in indices or not probes or not record.get("moves"):
            continue
        for name, moved in record["moves"].items():
            if any(moved == seen for _, seen, _, _ in refused):
                continue
            reason = probes.get("refused", {}).get(name)
            if name in probes.get("failed", []):
                refused.append((name, moved, None, reason))
            elif name in probes["parts"] and name not in record["include"]:
                refused.append((name, moved, probes["parts"][name], reason))
    return refused


def blocked(moved, refused):
    """Why a move against the current design is a hold before it is measured: this very move
    was measured against this design already. None when the move is worth a measurement."""
    if any(moved == seen for _, seen, _, _ in refused):
        return "a move already measured against this design"
    return None


def refused_text(refused, tolerance):
    lines = ["  {}: {} | {}{}".format(
        name if name in CONCERNS or name in ("analyst", "statistician") else "sweep",
        ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())),
        "the simulator could not measure it" if delta is None
        else "measured {:+.4f}{}".format(delta, " (flat)" if abs(delta) <= tolerance else ""),
        " | REFUSED by a cap: " + "; ".join(reason) if reason else "")
        for name, moved, delta, reason in refused]
    return "\n".join(lines) if lines else "  none"


def untried_moves(history, knobs, current, is_candidate=lambda knobs: True):
    """The values of these knobs never measured as a single change from the current design,
    within the area budget and runnable: the neighbourhood the run has not looked at, as
    (knob, value) pairs in knob order. A design that moved several knobs at once credits none
    of them alone, so a value carried only by such designs is untried here."""
    measured = {config_name(entry["knobs"]) for entry in history}
    moves = []
    for knob in knobs:
        for value in SEARCH_SPACE[knob]:
            design = dict(current, **{knob: value})
            if (value != current[knob] and within_budget(design)
                    and is_candidate(design) and config_name(design) not in measured):
                moves.append((knob, value))
    return moves


def fill_order(sheet):
    """The knobs in the order the wave's fill tries them: the level the sheet blames first."""
    blamed = None
    if sheet is not None:
        blamed = (sheet.get("bottleneck") or {}).get("level")
    first = []
    rest = []
    for knob in SEARCH_SPACE:
        if LEVEL_OF[knob.split("_")[0]] == blamed:
            first.append(knob)
        else:
            rest.append(knob)
    return first + rest


def statistician_section(state):
    """What the statistician's surrogate expects this round, for the analyst and the specialists
    to read beside the counters. Nothing before round 2."""
    if state is None or not state.get("statistician"):
        return []
    return ["", "## What the statistician expects (a surrogate fitted to every design this run measured; "
                "its five highest predictions as moves from the current design; predictions to test, not measurements)",
            state["statistician"]]


def current_heading(current_entry, objective):
    return "## The current design (the best this run has measured), D{} {}={:.4f}".format(
        current_entry["index"], objective, current_entry["metrics"][objective])


# ---------------------------------------------------------------- the synthesis ----

SYNTHESIS_TASK = """## Task
Compose this round's coupled design from the proposals above. Include the proposals that belong
together: a move that depends on another (say which), or independent moves that do not conflict.
Leave out a proposal your sheet or the ledgers contradict, and say why. A gain the power cap
refused is not dead: it belongs together with the proposal that frees the power it needs, and
each proposal's exact leakage and silicon are listed above, so compose the pair and say what it
frees. You may not change any proposed value; you choose which proposals are in. The parts are
measured alone whatever you compose; your design is the one extra measurement of the wave, so it
should be the combination whose interaction the parts alone cannot show. Include at least two
proposals, or nothing when no two belong together.

Return JSON with exactly these keys:
- "include": the list of concern names whose proposals form the design; empty to compose nothing.
- "hypothesis": one sentence: what the design should show against the current design, in report
  numbers, and why the included parts belong together.
- "excluded": an object with one sentence per excluded concern, empty if none.
"""


def synthesize(problem, sheet, incumbent, proposals, probes, tag, state=None, incumbent_entry=None):
    """The analyst's second call: which of the proposals form the round's design, with the
    sketches in front of it. Returns (included concern names, hypothesis). On any failure
    every proposal is in."""
    names = list(proposals)
    parts = ["## Role",
             "You are the analyst of a team tuning one cache hierarchy. The specialists have proposed; "
             "you compose the round's coupled design from the proposals, choosing among them and altering "
             "no value. Each proposal is measured alone on the current design in this round's wave, and "
             "the design you compose is measured beside them, so the wave tells the parts from their "
             "combination.",
             "",
             "## Chip", chip_line(problem),
             "",
             "## Your sheet this round",
             sheet_text(sheet) if sheet is not None else "  (no sheet this round)",
             "",
             "## The current design", "  " + assignment(incumbent, list(SEARCH_SPACE)),
             "",
             "## The proposals"]
    if incumbent_entry is not None:
        parts += [cost_line(incumbent_entry, problem)]
    base_silicon = silicon_mm2(typed_knobs(incumbent))
    base_leakage = leakage_w(typed_knobs(incumbent))
    for name in names:
        # A proposal may carry only the knobs it moves (the opening's parts do); show the whole
        # concern. A fill move is not a concern's: show what it moves.
        whole = dict(incumbent)
        whole.update(proposals[name]["knobs"])
        shown = CONCERNS[name]["knobs"] if name in CONCERNS else sorted(knobs_changed(whole, incumbent))
        typed = typed_knobs(whole)
        parts += ["  {}: {}".format(name, assignment(whole, shown)),
                  "    because: " + proposals[name]["reasoning"],
                  # The shape decides silicon and leakage before any simulation: exact, so the
                  # analyst can pair a gain a cap refused with the move that frees what it needs.
                  "    shape: silicon {:.2f} -> {:.2f} mm2, leakage {:.2f} -> {:.2f} W (exact; traffic power is measured)".format(
                      base_silicon, silicon_mm2(typed), base_leakage, leakage_w(typed))]
    parts += ["", SYNTHESIS_TASK]
    prompt_text = "\n".join(parts)
    note(tag, "\n---------------- synthesis ----------------\n" + prompt_text)
    try:
        answer = analyst.ask(prompt_text)
    except Exception as error:
        note(tag, "-> call failed, every proposal is in: " + repr(error)[-160:])
        return names, "every proposal, the synthesis call failed"
    if isinstance(answer, list) and answer:
        answer = answer[0]
    include = answer.get("include") if isinstance(answer, dict) else None
    if not isinstance(include, list):
        note(tag, "-> not a composition, every proposal is in: " + json.dumps(answer)[:300])
        return names, "every proposal, the synthesis answered off-contract"
    include = [name for name in names if name in include]
    hypothesis = str(answer.get("hypothesis", ""))[:400]
    excluded = answer.get("excluded") if isinstance(answer.get("excluded"), dict) else {}
    note(tag, "-> design: {}\n   hypothesis: {}{}".format(
        " + ".join(include) if include else "nothing this round", hypothesis,
        "".join("\n   left out {}: {}".format(name, str(why)[:200]) for name, why in excluded.items())))
    return include, hypothesis


# ---------------------------------------------------------------- the preview ----

def preview(problem):
    """Every prompt the council would write in its opening round on this chip, rendered
    and printed without a single model call, against the stock design.

    This is how a change to what the council reads gets checked. A run costs hours before
    it says whether the loop reads the chip right; this costs one simulation of the stock
    design, and nothing once that row is in the tables. Run it on each SoC in turn and the
    difference between the chips is the difference between the printouts."""
    knobs = typed_knobs(problem["stock"])
    entry = {"index": 0, "round": 0, "name": problem["name_of"](knobs), "knobs": knobs,
             "metrics": problem["evaluate"](knobs), "source": "stock", "hypothesis": None,
             "rung": list(problem["fidelity"]), "against": None, "baseline": None, "moved": {}}
    history = [entry]
    blocks = [("the analyst, opening round", diagnosis_prompt(problem, history, entry, opening=True))]
    for name in ORDER:
        blocks.append(("the {} specialist".format(name),
                       specialist_prompt(problem, name, history, knobs, entry, None)))
    for title, text in blocks:
        print("\n" + "=" * 96)
        print("== {} | {} characters, about {} tokens".format(title, len(text), len(text) // 4))
        print("=" * 96)
        print(text)
    print("\n{} prompts, about {} tokens a round before the sheet and the ledgers fill.".format(
        len(blocks), sum(len(text) for _, text in blocks) // 4))
    return blocks


# ---------------------------------------------------------------- the search ----

def search(problem, budget, seed, tag):
    """`budget` designs: every design the run evaluates counts once, whether the simulator ran
    or the table answered. Round 1 is an opening wave of OPENINGS designs; every later round is
    one wave of WAVE designs: the analyst's sheet, four proposals at once, the analyst's
    composition of them, untried one-knob moves filling the wave, all measured together. The
    incumbent is the best feasible design measured, whoever proposed it. PLATEAU designs without
    a new incumbent end the search."""
    analyst.RUN_TAG = tag
    objective = problem["objective"]
    fidelity = tuple(problem["fidelity"])
    started = time.time()
    history = []
    rounds = []
    by_name = {}            # (design name, rung) -> its entry: a design is evaluated once
    ledger_path = os.path.join(LEDGER_DIR, tag + ".jsonl")
    if os.path.exists(ledger_path):
        os.remove(ledger_path)          # a re-run under the same tag starts a fresh ledger
    ledger = Ledger(ledger_path, objective)

    def record(knobs, metrics, source, round_number, hypothesis, against, rung):
        entry = {"index": len(history), "round": round_number, "name": problem["name_of"](knobs),
                 "knobs": knobs, "metrics": metrics, "source": source,
                 "hypothesis": hypothesis(knobs) if callable(hypothesis) else hypothesis,
                 "rung": list(rung), "against": None, "baseline": None, "moved": {},
                 # The stock is the first design recorded and the reference for the floors.
                 "violations": violations(metrics, problem["workloads"], history[0]["metrics"]) if history else []}
        if against is not None:
            entry["against"] = against["index"]
            entry["baseline"] = against["metrics"][objective]
            entry["moved"] = knobs_changed(knobs, against["knobs"])
        history.append(entry)
        by_name[(entry["name"], tuple(rung))] = entry
        # The CHIA ledger block: one row per candidate as it is measured (who, why, what it
        # scored, what it broke), and a `candidate` event in the run's profiler log.
        ledger.record(knobs, {objective: metrics[objective], "mm2": metrics.get("mm2"), "watts": metrics.get("watts")},
                      entry["source"], round_number, entry["hypothesis"], entry["violations"])
        note(tag, "\n======== D{} [{}] {}={:.4f} | {}{} ========".format(
            entry["index"], source, objective, metrics[objective],
            ", ".join("{}={}".format(knob, knobs[knob]) for knob in sorted(knobs_changed(knobs, problem["stock"]))) or "stock",
            violation_note(entry)))
        print("[{}] round {} | D{} | {}={:.4f} | {} | {:.0f} min".format(
            tag, round_number, entry["index"], objective, metrics[objective], source, (time.time() - started) / 60), flush=True)
        return entry

    def measure(problem_, designs, source, round_number, against, hypothesis=None):
        """The designs' entries, in the order asked: the ones already evaluated at this rung
        are read back, the others are measured now and recorded. None for a design the
        simulator could not measure. `source` is a label or a function of the knobs."""
        rung = tuple(problem_["fidelity"])
        fresh, seen = [], set()
        for design in designs:
            name = problem_["name_of"](design)
            if (name, rung) not in by_name and name not in seen:
                seen.add(name)
                fresh.append(design)
        if fresh:
            knobs_list, metrics_list, _ = engine.measure_batch(problem_, fresh, tag)
            for knobs, metrics in zip(knobs_list, metrics_list):
                record(knobs, metrics, source(knobs) if callable(source) else source, round_number, hypothesis, against, rung)
        return [by_name.get((problem_["name_of"](design), rung)) for design in designs]

    def incumbent_of():
        """The best feasible design measured at the run's fidelity, whoever proposed it. A design
        that breaks a cap is measured, read and recorded; it is never stood on."""
        return max((entry for entry in history
                    if tuple(entry["rung"]) == fidelity and not entry.get("violations")),
                   key=lambda entry: entry["metrics"][objective])

    def wave(incumbent_entry, proposals, sheet, round_number, state, composed=(), hypothesis=""):
        """One round's wave, WAVE designs measured together at the run's fidelity: every
        proposal alone on the incumbent, the design the analyst composed from them beside
        the parts, and untried one-knob moves at the level the sheet blames filling the
        rest. The incumbent afterwards is whatever measured best, whoever proposed it.
        Returns the round's record; its `probes` hold each design's gain against the
        incumbent, which is what every ledger the team reads is written from."""
        incumbent = incumbent_entry["knobs"]
        is_candidate = problem.get("is_candidate", lambda knobs: True)
        parts = {name: proposals[name]["knobs"] for name in proposals}
        composed_label = None
        if len(composed) > 1:
            composed_label = "joint" if len(composed) == len(parts) else "+".join(composed)
        room = WAVE - len(parts) - (1 if composed_label else 0)
        fill_level = None
        if room > 0 and FILL:
            # Untried one-knob moves top the wave up, the knobs of the level the sheet blames first,
            # so every round measures WAVE designs and the neighbourhood gets read alone.
            for knob, value in untried_moves(history, fill_order(sheet), incumbent, is_candidate):
                if room <= 0:
                    break
                label = "{}={}".format(knob, value)
                design = dict(incumbent, **{knob: value})
                if label in parts or any(design == with_parts(incumbent, [part]) for part in parts.values()):
                    continue
                parts[label] = {knob: value}
                room -= 1
        moves = {label: knobs_changed(with_parts(incumbent, [part]), incumbent) for label, part in parts.items()}
        planned = [(label, with_parts(incumbent, [part])) for label, part in parts.items()]
        if composed_label is not None:
            planned.append((composed_label, with_parts(incumbent, [proposals[name]["knobs"] for name in composed])))
        label_of = {}
        for label, design in planned:
            label_of.setdefault(problem["name_of"](design), label)

        def source_of(knobs):
            label = label_of[problem["name_of"](knobs)]
            if label in CONCERNS or label in ("analyst", "statistician") or label == composed_label \
                    or label.startswith("repair:"):
                return "council:" + label
            return "sweep:" + label

        def reason_of(knobs):
            label = label_of[problem["name_of"](knobs)]
            if label == composed_label:
                return hypothesis
            return proposals.get(label, {}).get("reasoning")

        entries = measure(problem, [design for _, design in planned], source_of, round_number, incumbent_entry, reason_of)
        base = incumbent_entry["metrics"][objective]
        values = {}
        for (label, _), entry in zip(planned, entries):
            if entry is None:
                continue
            values.setdefault(label, entry["metrics"][objective])
        refused = {}
        for (label, _), entry in zip(planned, entries):
            if entry is not None and entry.get("violations"):
                refused.setdefault(label, entry["violations"])
        probes = {"rung": list(fidelity), "incumbent": base,
                  "parts": {label: values[label] - base for label in parts if label in values},
                  "failed": [label for label in parts if label not in values],
                  # A design refused by a cap after measurement, and why: shown in every ledger,
                  # so a gain the chip may not ship is not composed again (g5a-s0 did, three times).
                  "refused": refused,
                  "pairs": {}}
        if composed_label is not None and composed_label in values:
            probes["pairs" if composed_label != "joint" else "joint"] = (
                {composed_label: values[composed_label] - base} if composed_label != "joint" else values["joint"] - base)
            if composed_label == "joint":
                probes["interaction"] = probes["joint"] - sum(probes["parts"].get(name, 0.0) for name in composed)
        feasible = {label: values[label] for (label, _), entry in zip(planned, entries)
                    if entry is not None and not entry.get("violations") and label in values}
        best_label = max(feasible, key=feasible.get) if feasible else None
        include = [best_label] if best_label is not None and feasible[best_label] > base else []
        note(tag, "-> the wave against D{}: {}{}".format(
            incumbent_entry["index"],
            ", ".join("{} {:+.4f}".format(label, values[label] - base) for label, _ in planned if label in values),
            " | new incumbent: " + best_label if include else " | the incumbent stands"))
        record_ = {"round": round_number, "mode": state["mode"], "against": incumbent_entry["index"], "sheet": sheet,
                   "sweep": fill_level, "proposals": proposals, "moves": moves, "include": include,
                   "hypothesis": hypothesis, "probes": probes}
        rounds.append(record_)
        return record_

    stock_entry = measure(problem, [typed_knobs(problem["stock"])], "stock", 0, None)[0]
    # The statistician's candidates: a fixed random sample of the feasible designs, drawn with
    # the run's seed as the forest arm draws its own.
    pool = engine.candidate_pool(problem, seed)

    # Round 1: the analyst's sheet for the stock chip and its opening design, measured in one
    # wave with random feasible designs beside it, OPENINGS designs in all - the opening every
    # arm gets. Ten openings reached 90 % of the gap by round 4 to 6 where none never did (g5a).
    round_number = 1
    note(tag, "\n\n################ round 1 | incumbent D0 {}={:.4f} | opening ################".format(
        objective, stock_entry["metrics"][objective]))
    sheet = diagnose(problem, history, stock_entry, tag, opening=True, rounds=rounds)
    previous = sheet
    opening = opening_design(sheet, problem, tag)
    openers = []
    labels = {}
    if opening is not None:
        openers.append(opening)
        labels[problem["name_of"](opening)] = "council:opening"
    for design in random_feasible_designs(OPENINGS, seed=seed):
        if len(openers) >= OPENINGS:
            break
        name = problem["name_of"](design)
        if name not in labels:
            openers.append(design)
            labels[name] = "opening:random"
    def opening_reason(knobs):
        if opening is not None and problem["name_of"](knobs) == problem["name_of"](opening):
            return (sheet or {}).get("opening_reasoning")
        return None

    measure(problem, openers, lambda knobs: labels[problem["name_of"](knobs)], 1, stock_entry, opening_reason)

    # Every later round is one wave of WAVE designs. The budget, the round cap or PLATEAU
    # designs without a new incumbent end the search.
    max_rounds = MAX_ROUNDS if MAX_ROUNDS else 3 * budget
    since_incumbent = 0         # designs measured since the incumbent last moved
    fruitless = 0               # rounds in a row in which nobody proposed a design
    while len(history) - 1 < budget and round_number < max_rounds and since_incumbent < PLATEAU:
        analyst.ROUND = round_number + 1
        incumbent_entry = incumbent_of()
        current = incumbent_entry["knobs"]
        mode = "climb"
        is_candidate = problem.get("is_candidate", lambda knobs: True)
        tolerance = FLAT_SHARE * incumbent_entry["metrics"][objective]
        peers = {entry["index"] for entry in history
                 if tuple(entry["rung"]) == fidelity and not entry.get("violations")
                 and entry["metrics"][objective] >= incumbent_entry["metrics"][objective] - tolerance}
        state = {"mode": mode, "refused": refused_moves(rounds, peers), "tolerance": tolerance}
        # The statistician speaks first: what its surrogate expects is in front of the analyst
        # and the specialists, and its own proposal joins the wave beside theirs.
        statistician_proposal = statistician(problem, history, incumbent_entry, pool, tag) if STATISTICIAN else None
        state["statistician"] = statistician_proposal["view"] if statistician_proposal is not None else None
        round_number += 1
        note(tag, "\n\n################ round {} | {} | incumbent D{} {}={:.4f} | {} of {} designs ################".format(
            round_number, mode, incumbent_entry["index"], objective,
            incumbent_entry["metrics"][objective], len(history) - 1, budget))
        sheet = diagnose(problem, history, incumbent_entry, tag, previous=previous, rounds=rounds, state=state, evaluated=history)
        previous = sheet if sheet is not None else previous

        proposals = {}
        composed, hypothesis = (), ""
        verdicts = skip_verdicts(sheet, state, current, tag, round_number)
        consulted = consulted_specialists(verdicts, tag)
        logs = {name: [] for name in ORDER}
        answers = {name: None for name in ORDER}
        with ThreadPoolExecutor(max_workers=len(ORDER)) as workers:
            futures = {name: workers.submit(specialist, problem, name, history, current, incumbent_entry, tag, sheet, logs[name], rounds, state, history)
                       for name in consulted}
            for name in futures:
                answers[name] = futures[name].result()
        for name in ORDER:
            for text_ in logs[name]:
                note(tag, text_)
        # A proposal counts as a move only if it changes the design once typed and fitted; a
        # move already measured against this design, or one the simulator could not measure,
        # is a hold and costs nothing.
        for name in ORDER:
            if answers[name] is None:
                continue
            design = with_parts(current, [answers[name]["knobs"]])
            moved = knobs_changed(design, current)
            if not moved:
                continue
            if not within_budget(design):
                note(tag, "-> {}'s design needs {}; taken as a hold".format(name, budget_text(design)))
                continue
            if not is_candidate(design):
                note(tag, "-> {}'s design is one the simulator could not measure before; taken as a hold".format(name))
                continue
            why = blocked(moved, state["refused"])
            if why:
                note(tag, "-> {} proposed {}; taken as a hold".format(name, why))
                continue
            proposals[name] = answers[name]
        # The analyst's whole design for the round, written in its sheet call from the best
        # designs measured in full, the refusals and the ladder, is measured beside the
        # specialists' moves: the one design a round that may change regime. When the analyst
        # wrote none, the statistician's second candidate takes the slot, and the ledger says so.
        design = opening_design(sheet, problem, tag, base=current, key="design") if sheet is not None else None
        reasoning = sheet.get("design_reasoning", "") if sheet is not None else ""
        if design is None and statistician_proposal is not None and statistician_proposal.get("runner_up") is not None:
            design = statistician_proposal["runner_up"]
            reasoning = "the statistician's second candidate, taken because the analyst wrote no design"
            note(tag, "-> the analyst's slot goes to the statistician's second candidate")
        if design is not None:
            moved = knobs_changed(design, current)
            why = blocked(moved, state["refused"])
            if why:
                note(tag, "-> the analyst's design is {}; dropped".format(why))
            elif not is_candidate(design):
                note(tag, "-> the analyst's design is one the simulator could not measure before; dropped")
            else:
                proposals["analyst"] = {"knobs": {knob: design[knob] for knob in moved}, "reasoning": reasoning}
        # A gain the power cap refused last round comes back repaired to fit.
        for label, repair in repair_candidates(rounds[-1] if rounds else None, history, problem, is_candidate).items():
            moved = knobs_changed(with_parts(current, [repair["knobs"]]), current)
            if moved and not blocked(moved, state["refused"]):
                proposals[label] = repair
                note(tag, "-> {}: {}".format(label, repair["reasoning"]))
        # The statistician's own pick joins the wave.
        if statistician_proposal is not None:
            moved = knobs_changed(statistician_proposal["knobs"], current)
            if moved and within_budget(typed_knobs(statistician_proposal["knobs"])) and is_candidate(statistician_proposal["knobs"]) \
                    and not blocked(moved, state["refused"]):
                proposals["statistician"] = statistician_proposal
        # At most WAVE designs: the statistician, the analyst's design, the repairs, then the
        # specialists in order. The fill tops the wave up when fewer propose.
        ordered = {}
        repairs_first = [label for label in proposals if label.startswith("repair:")]
        for label in ["statistician", "analyst"] + repairs_first + ORDER + list(proposals):
            if label in proposals and label not in ordered and len(ordered) < WAVE:
                ordered[label] = proposals[label]
        record_ = wave(incumbent_entry, ordered, sheet, round_number, state, composed, hypothesis)
        log_skip(tag, round_number, verdicts, ordered, record_["probes"])

        measured_now = sum(1 for entry in history if entry["round"] == round_number)
        if measured_now == 0:
            # Nobody proposed and nothing was left to fill with. One more sheet may propose; a
            # second empty round in a row means nothing is left to measure.
            fruitless += 1
            if fruitless >= 2:
                note(tag, "-> nothing left to measure; stopping")
                print("[{}] round {} | nothing left to measure, stopping".format(tag, round_number), flush=True)
                break
            print("[{}] round {} | nobody proposed; asking once more".format(tag, round_number), flush=True)
            continue
        fruitless = 0
        best_now = incumbent_of()
        improved = best_now["index"] != incumbent_entry["index"]
        since_incumbent = 0 if improved else since_incumbent + measured_now
        print("[{}] round {} | {} | {} designs | incumbent D{} {}={:.4f}{}".format(
            tag, round_number, mode, len(history) - 1, best_now["index"], objective, best_now["metrics"][objective],
            "" if improved else " | did not move ({} designs since)".format(since_incumbent)), flush=True)
    if since_incumbent >= PLATEAU:
        print("[{}] plateau: {} designs without a new incumbent, stopping".format(tag, since_incumbent), flush=True)
    if round_number >= max_rounds:
        print("[{}] {} rounds: stopping".format(tag, round_number), flush=True)
    print("[{}] {} designs evaluated, best D{} {}={:.4f}".format(
        tag, len(history) - 1, incumbent_of()["index"], objective, incumbent_of()["metrics"][objective]), flush=True)
    return {"designs": history, "rounds": rounds}
