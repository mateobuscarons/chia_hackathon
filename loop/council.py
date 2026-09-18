"""The council: an analyst and four specialists tuning one cache hierarchy from the stock
chip, in team rounds, sketching before they commit.

A round: the analyst reads every design measured and writes the sheet (a verdict per level,
the level that misses most never "fine", each finding with its numbers, what is
unattributable, which knobs fail by their own counters, a prediction); all four specialists
read it and propose at once, one move each on their own knobs - one knob, or several when one
mechanism needs them - or hold; every proposal is sketched at the probe rung - the incumbent,
each proposal alone on it, all of them together - one wave of parallel simulations; the analyst
composes the round's design from the proposals with the sketches in front of it, choosing among
them and altering no value; a composed design the wave did not sketch is measured then. Every
design evaluated is a design of the run. Round 1 is the same with the analyst's
opening design from the stock report, split by concern, in place of proposals. The ledger keeps
what the sketches predicted next to what was measured. Every design the run evaluates, sketch
or composition, is a design of the run and counts against the budget; the council reads and
stands on the composed designs, and the incumbent is the best of them. When every specialist holds, the analyst takes the turn
with one design of its own, split by concern like the opening.

The council only climbs, so it can stall in a local optimum with budget left. STALL_ROUNDS
rounds in a row that do not move the incumbent make the next round a jump round: everyone reads
the moves the sketches refused against the incumbent and, per knob, the values never measured as
a one-knob change from it; each specialist proposes the untried one-knob move it expects most,
alone, so the sketch is attributable, and only a concern whose neighbourhood is exhausted
proposes the coupled move that puts its levels in a different regime; the analyst writes which
other level's evidence would motivate a different design and composes from the parts that
gained. A jump may not carry a part the sketches refused unless the parts together are not a
loss; a jump the sketches do not predict to gain is not measured, and the next round jumps
again with a smaller neighbourhood, sweeping the level the jump rounds have blamed and sketched
least, so a stall does not spend its budget on the levels the sheet keeps naming. The search stops when the budget is spent or after three
rounds per design of budget; a stalled search keeps jumping, its refused list growing.

The concerns split the thirteen knobs by what they are - what prefetches, how big, what
policy, how many misses in flight. The principles say what a knob does, never which value to
pick, and name no workload, trace or chip; a workload is shown by position (W1, W2, ...).
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from loop import analyst
from loop import search as engine
from loop.simulate import derived_latency
from loop.space import SEARCH_SPACE, cache_area_kb, config_name, fit_to_budget, knobs_changed, typed_knobs, within_budget

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

# A stall is STALL_ROUNDS rounds in a row that did not move the incumbent; the next round is a
# jump round. FLAT_SHARE of the incumbent is the tolerance below which a sketch counts as a loss
# in the jump guard and in the refused list.
STALL_ROUNDS = 2
FLAT_SHARE = 0.005

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


def labels_for(workloads):
    """Workloads are shown by position (W1, W2, ...), never by name: a trace's file
    name can say what the program is, and the specialist is not told."""
    return {workload: "W{}".format(index + 1) for index, workload in enumerate(workloads)}


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
            cell = "{} mpki {:.2f}, hit {:.2f}".format(labels[workload], got("mpki"), got("hit_ratio") or 0.0)
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



ALL_LEVELS = ["L1D", "L2C", "LLC"]


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
        lines.append(line)
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
            stats = seen.setdefault(str(entry["knobs"][knob]), {"n": 0, "cov": [], "acc": [], "best": None})
            stats["n"] += 1
            if knob.endswith("_prefetcher"):
                for workload in workloads:
                    coverage = entry["metrics"].get("{}:{}_pf_coverage".format(workload, level))
                    if coverage is not None:
                        stats["cov"].append(coverage)
                        stats["acc"].append(entry["metrics"].get("{}:{}_pf_accuracy".format(workload, level)) or 0.0)
            value = entry["metrics"][objective]
            if stats["best"] is None or value > stats["best"]:
                stats["best"] = value
        cells = []
        for value in sorted(seen, key=lambda v: -seen[v]["best"]):
            stats = seen[value]
            cell = "{}: {} design{}, best {:.4f}".format(value, stats["n"], "" if stats["n"] == 1 else "s", stats["best"])
            if stats["cov"]:
                cell += ", cov {:.2f} acc {:.2f}".format(
                    sum(stats["cov"]) / len(stats["cov"]), sum(stats["acc"]) / len(stats["acc"]))
            cells.append(cell)
        lines.append("  {}: {}".format(knob, " | ".join(cells)))
    return "\n".join(lines)


LEVEL_SHAPE = {"L1D": ("l1d_sets", "l1d_ways"), "L2C": ("l2_sets", "l2_ways"), "LLC": ("llc_sets", "llc_ways")}
VERDICTS = ["capacity-bound", "latency-heavy", "prefetch-uncovered", "prefetch-polluting", "bandwidth-bound", "fine"]


def derived_view(entry, workloads, levels=ALL_LEVELS):
    """The quantities an architect reasons with, computed from the counters alone so they
    read the same on any chip and any suite. Per workload and level: the share of the
    accesses reaching the level that miss it, the share of the level above's misses that
    miss again here, the hit cycles paid per useful hit, the prefetches issued and the
    useless ones per thousand instructions; then off-chip traffic in KB per thousand
    instructions and the DRAM row-buffer hit rate."""
    labels = labels_for(workloads)
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
            if level not in levels:
                above_misses = misses
                continue
            accesses = hits + misses
            sets_knob, ways_knob = LEVEL_SHAPE[level]
            latency = derived_latency(entry["knobs"][sets_knob], entry["knobs"][ways_knob])
            hit_ratio = hits / accesses if accesses else 0.0
            cell = "{}: miss share {:.2f}".format(level, misses / accesses if accesses else 0.0)
            if above_misses:
                cell += ", of the level above's misses {:.2f} miss again".format(min(misses / above_misses, 9.99))
            cell += ", {:.1f} hit cycles per useful hit".format(latency / hit_ratio if hit_ratio else float("inf"))
            issued, useless = got(level + "_pf_issued") or 0, got(level + "_pf_useless") or 0
            if issued:
                cell += ", prefetch issued {:.1f}/ki useless {:.1f}/ki".format(issued / kilo, useless / kilo)
            cells.append(cell)
            above_misses = misses
        llc_misses = (got("LLC_misses") or 0) + (got("LLC_pf_misses") or 0)
        cells.append("off-chip {:.1f} KB/ki".format(llc_misses * 64 / 1024 / kilo))
        row_hit, row_miss = got("dram_rq_row_buffer_hit") or 0, got("dram_rq_row_buffer_miss") or 0
        if row_hit + row_miss:
            cells.append("DRAM row-buffer hit {:.2f}".format(row_hit / (row_hit + row_miss)))
        lines.append("  {} | {}".format(labels[workload], " | ".join(cells)))
    return "\n".join(lines)


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


def knob_section():
    """Every knob and its allowed values, the prefetch placement table and the latency ladder."""
    lines = ["## Knobs and allowed values"]
    lines += ["- {}: {}".format(knob, json.dumps(SEARCH_SPACE[knob])) for knob in SEARCH_SPACE]
    lines += ["Prefetch mechanisms and the levels that accept each:", placement_table(CONCERNS["prefetch"]["knobs"]),
              "Capacity and the hit latency it costs (latency follows capacity alone):", latency_ladder(ALL_LEVELS)]
    return lines


# ---------------------------------------------------------------- the diagnosis ----

DIAGNOSIS_TASK = """## Task
Write the sheet for the current design. You choose no knob and you name no knob value: the
specialists decide, from what you establish. Read every design measured, not only the last, and
read the counters, not only the objective. One design is one measurement: write what it shows, not
what it confirms. A design whose outcome cannot be assigned to one knob because several moved
together teaches nothing about any one of them: say so, and draw no per-knob lesson from it.

Return JSON with exactly these keys:
- "levels": an object with one entry each for L1D, L2C and LLC: {"verdict": one of capacity-bound,
  latency-heavy, prefetch-uncovered, prefetch-polluting, bandwidth-bound, fine; "number": the one
  figure from the report or the derived view that says so}. Rank the levels by misses per
  kilo-instruction first: the level that misses most is where the objective is lost, and it is
  never fine, whatever its hit ratio; its verdict names what would cut those misses.
- "bottleneck": an object {"level": one of L1D, L2C, LLC, DRAM, "cause": one sentence naming the
  mechanism, "evidence": the numbers that show it}.
- "learned": a list of up to six strings. Each states one finding with its numbers: the knob or
  level, what moved, the change in the objective with its sign and the design ids, and the counter
  that explains it - of the form "L2 halved: +0.002 (D3 vs D1); L2 hit ratio unchanged at 0.28,
  hit latency 9 -> 7 cycles". An entry without a number is not a finding.
- "unattributable": a list of strings, one per design whose outcome cannot be assigned to one knob,
  naming the knobs that moved together; empty if none.
- "failing": a list of strings: the knobs whose current value is not doing its job by its own
  counters - a prefetcher covering few misses at low accuracy, a level charging its hit latency
  for a low hit ratio, a policy that changed nothing - each with the number that says so and how
  many of its alternatives this run has measured at that level. Knobs and levels, not the value
  to pick.
- "hypothesis": one sentence predicting what the current design's report will show changed after
  the next move, in report numbers. A prediction, not an experiment to run.
"""

OPENING_TASK = """
This is the first round and the current design is the stock chip. Besides the sheet, propose the
opening design: the one design measured before the specialists take turns, so it should carry
every move the stock report's evidence already supports, across as many concerns as the evidence
reaches, and leave every other knob at its stock value. The specialists refine from it one concern
at a time.

Add to the JSON:
- "opening": an object with every knob and one allowed value each.
- "opening_reasoning": one sentence per move, what evidence it acts on.
"""

JUMP_SHEET_TASK = """
This is a jump round: the search is stalled on the current design and the moves refused against it
are listed above. The sheets so far have placed the bottleneck at {named}. Add to the JSON:
- "counterfactual": an object {{"level": the level named least, or never, whose evidence would most
  change the design if it were read as the bottleneck; "cause": one sentence naming the mechanism;
  "evidence": its numbers; "would_change": the knobs, across concerns, the specialists would then
  have to move together}}. The specialists read it beside your verdicts.
"""

ANALYST_TURN_TASK = """
Every specialist held on the current design this round. Besides the sheet, propose the one design
you would measure: every move the current report's evidence supports, across as many concerns as
it reaches, and every other knob at its current value. Look where the misses are, not only where
a verdict points; a hold that waits for another concern is not evidence. The design is sketched
and vetoed like any proposal.

Add to the JSON:
- "opening": an object with every knob and one allowed value each.
- "opening_reasoning": one sentence per move, what evidence it acts on.
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
    if sheet.get("counterfactual"):
        counter = sheet["counterfactual"]
        lines.append("  counterfactual bottleneck at {}: {} | evidence: {} | would move together: {}".format(
            counter["level"], counter["cause"], counter["evidence"], counter["would_change"]))
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


def diagnose(problem, history, incumbent_entry, tag, opening=False, previous=None, rounds=(), turn=False, state=None, evaluated=None):
    """One call a round: the bottleneck sheet every specialist reads, in round 1 the opening
    design, and when every specialist held, the analyst's own design for the round. It reads
    its previous sheet and what was measured since; on a jump round it also names the
    counterfactual bottleneck. `incumbent_entry` is the design the round refines from. None if
    the call fails or the answer is not a sheet; the round then runs without one."""
    jump = state is not None and state["mode"] == "jump"
    objective = problem["objective"]
    parts = ["## Role",
             "You are the analyst of a team tuning one cache hierarchy. Four specialists own the knobs, "
             "one concern each; you own none and you propose no value. You read every design this run "
             "has measured and write the sheet the specialists read: what the measurements have "
             "established with its numbers, what they cannot tell apart, and which knobs are failing by "
             "their own counters.",
             "",
             "## Chip", problem["chip_text"],
             "Objective: maximise {}, the geometric mean of IPC over {}.".format(
                 objective, " + ".join(labels_for(problem["workloads"]).values())),
             "Concerns and their knobs: " + "; ".join(
                 "{} ({})".format(name, ", ".join(CONCERNS[name]["knobs"])) for name in ORDER) + ".",
             ""]
    parts += knob_section()
    parts += ["",
              current_heading(incumbent_entry, objective),
              "  knobs:  " + assignment(incumbent_entry["knobs"], list(SEARCH_SPACE)),
              "  area:   {:.0f} of {} KB of the budget in use by L2 + LLC".format(
                  cache_area_kb(incumbent_entry["knobs"]), problem["area_budget_kb"]),
              level_report(incumbent_entry, ALL_LEVELS, problem["workloads"]),
              "  derived from the counters:",
              derived_view(incumbent_entry, problem["workloads"])]
    if history[-1]["index"] != incumbent_entry["index"]:
        parts += last_round_section(problem, history, incumbent_entry["knobs"], ALL_LEVELS)
        parts += ["  derived from the counters:", derived_view(history[-1], problem["workloads"])]
    if previous is not None:
        parts += previous_section(previous, history, objective)
    parts += ["",
              "## Every move this run has made",
              moves_ledger(history, objective),
              "",
              "## What the sketches showed, round by round (cheap simulations; the parts alone and together)",
              sketch_ledger(rounds, objective),
              "",
              "## What each value of each knob has done across the run",
              value_ledger(history, list(SEARCH_SPACE), problem["workloads"], objective)]
    if jump:
        parts += ["  (a design's ipc is credited to every value it carries; knobs that moved together are not separated here)",
                  "  never measured as a one-knob change from the current design:",
                  untried_text(evaluated if evaluated is not None else history, list(SEARCH_SPACE), incumbent_entry["knobs"])]
    parts += ["",
              "## The team's recent designs",
              recent(history, problem)]
    parts += state_section(state)
    task = DIAGNOSIS_TASK
    if jump:
        counts = {}
        for record in rounds:
            if record.get("sheet"):
                level = record["sheet"]["bottleneck"]["level"]
                counts[level] = counts.get(level, 0) + 1
        task += JUMP_SHEET_TASK.format(named=", ".join(
            "{} ({} sheets)".format(level, n) for level, n in sorted(counts.items(), key=lambda item: -item[1])) or "no level yet")
    parts += ["", task + (OPENING_TASK if opening else ANALYST_TURN_TASK if turn else "")]
    prompt_text = "\n".join(parts)
    note(tag, "\n---------------- diagnosis{} ----------------\n{}".format(
        " + opening" if opening else " + the analyst's turn" if turn else "", prompt_text))
    try:
        answer = analyst.ask(prompt_text)
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
    if jump and isinstance(answer.get("counterfactual"), dict):
        sheet["counterfactual"] = {key: str(answer["counterfactual"].get(key, ""))[:300]
                                   for key in ["level", "cause", "evidence", "would_change"]}
    note(tag, "-> sheet:\n" + sheet_text(sheet))
    if opening or turn:
        sheet["opening"] = answer.get("opening")
        sheet["opening_reasoning"] = str(answer.get("opening_reasoning", ""))[:600]
    return sheet


def opening_design(sheet, problem, tag, base=None):
    """The analyst's design: every knob at an allowed value, fitted to the budget, and
    different from the base, the stock chip in round 1 and the incumbent on the analyst's
    turn. None otherwise, and round 1 opens on the stock chip."""
    what = "opening" if base is None else "analyst"
    base = problem["stock"] if base is None else base
    proposed = sheet.get("opening") if sheet else None
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
    design = typed_knobs(fit_to_budget(typed_knobs(design)))
    if not knobs_changed(design, base):
        note(tag, "-> the {} design is the current one".format(what))
        return None
    note(tag, "-> {}: {}\n   because: {}".format(what,
        ", ".join("{}={}".format(knob, design[knob]) for knob in sorted(knobs_changed(design, base))),
        sheet["opening_reasoning"]))
    return design


# ---------------------------------------------------------------- the specialist ----

TASK = """## Task
Every specialist proposes this round. Each proposal is sketched cheaply, alone and together with
the others, before the analyst composes the round's design from the proposals that belong
together; one design is then measured for real. Propose the move for your knobs that the evidence
supports - one knob, or several of yours together when one mechanism needs them; if it depends on
another concern's move - a smaller level that needs timely prefetch in front of it, a policy that
pays only behind a prefetcher - say so in your reasoning. Move as far as your levels' evidence
supports in one design rather than one notch at a time; the next round tells you if it was too
far. Hold (return the current values) when the evidence at your levels shows nothing worth a
simulation.

Do not re-propose an assignment of your knobs that this run has already measured below the
current design, unless that row says other knobs differed and those differences change your
reasoning.

Return JSON with exactly these keys:
- "reasoning": one sentence - the mechanism you are acting on and what it depends on, or why you
  are holding.
- "knobs": an object with exactly your knobs and one allowed value each.
"""


JUMP_TASK = """## Task
The search is stalled: the rounds against the current design did not move it, and the moves the
sketches refused against it are listed above. Your candidates are the values of your knobs never
measured as a one-knob change from the current design, listed above: a move refused on an earlier
design was never measured on this one, and a move of yours that changed several knobs at once
credited none of them alone. Propose the one you expect most, alone: nothing else of yours moves,
so the sketch is attributable; the level the state names to sweep first, else the levels the
sheet blames first, but any of your levels. Only
when that list is empty for your knobs, propose the coupled move that puts your levels in a
different regime, and say which hypothesis about the bottleneck it tests. Do not re-propose a
move refused against the current design. Hold (return the current values) only if you can name
the measurement that would make you move.

Return JSON with exactly these keys:
- "reasoning": one sentence - the hypothesis the move tests and the mechanism it acts on, or why
  you are holding and what would make you move.
- "knobs": an object with exactly your knobs and one allowed value each.
"""


def specialist(problem, name, history, incumbent, incumbent_entry, tag, sheet, log, rounds=(), state=None, evaluated=None):
    """The first version's specialist prompt with the analyst's sheet after the current
    design's report, the derived view of its levels, its concern's moves with their deltas
    and what each value of its knobs has done. One proposal for its own knobs, or None.
    Transcript lines go to `log`, so four specialists can answer at once and be written
    down in order."""
    concern = CONCERNS[name]
    objective = problem["objective"]
    jump = state is not None and state["mode"] == "jump"
    parts = ["## Role",
             "You are the {} specialist in a team tuning one cache hierarchy.".format(name),
             "You choose only these knobs: " + ", ".join(concern["knobs"]) + ".",
             "Other specialists own every other knob and propose beside you this round.",
             "",
             "## Principles"]
    parts += ["- " + line for line in concern["principles"]]
    parts += ["",
              "## Chip", problem["chip_text"],
              "Objective: maximise {}, the geometric mean of IPC over {}.".format(
                  objective, " + ".join(labels_for(problem["workloads"]).values())),
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
              level_report(incumbent_entry, concern["levels"], problem["workloads"]),
              "  derived from the counters:",
              derived_view(incumbent_entry, problem["workloads"], concern["levels"])]
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
    if jump:
        parts += ["  (a design's ipc is credited to every value it carries; knobs that moved together are not separated here)",
                  "  never measured as a one-knob change from the current design:",
                  untried_text(evaluated if evaluated is not None else history, concern["knobs"], incumbent_entry["knobs"])]
    parts += ["",
              "## Every move of your concern this run has made, and what it did",
              concern_moves(history, concern["knobs"], objective),
              "",
              "## What your concern's sketches showed, committed or not",
              concern_sketches(rounds, name),
              "",
              "## The team's recent designs",
              recent(history, problem)]
    parts += state_section(state)
    parts += ["", JUMP_TASK if jump else TASK]
    prompt_text = "\n".join(parts)
    log.append("\n---------------- {} specialist ----------------\n{}".format(name, prompt_text))
    asked_once_more = False
    for attempt in range(3):
        try:
            answer = analyst.ask(prompt_text)
        except Exception as error:
            print("[{}] {} specialist failed ({})".format(tag, name, repr(error)[-120:]), flush=True)
            log.append("-> call failed: " + repr(error)[-160:])
            return None
        proposed = read_proposal(answer, concern["knobs"])
        if proposed is None:
            print("[{}] {} specialist answered off-contract, one retry".format(tag, name), flush=True)
            log.append("-> off-contract, retrying: " + json.dumps(answer)[:300])
            if attempt == 2 or asked_once_more:
                break
            continue
        held = proposed["knobs"] == {knob: incumbent[knob] for knob in concern["knobs"]}
        levels = levels_moved(proposed["knobs"], incumbent)
        log.append("-> {}{}\n   because: {}".format(
            assignment(proposed["knobs"], concern["knobs"]), "   (held)" if held else "",
            proposed["reasoning"]))
        if len(levels) > 1 and not asked_once_more:
            # The task asks for one level a round; ask once more, then take what comes.
            asked_once_more = True
            log.append("-> moves {} levels ({}); asked once more for one level".format(len(levels), ", ".join(levels)))
            prompt_text += ("\n\n## Once more\nYour proposal moved {} levels ({}). Propose the one level whose move "
                            "the evidence supports most, and leave the others at their current values."
                            .format(len(levels), ", ".join(levels)))
            continue
        if len(levels) > 1:
            log.append("-> still {} levels; taken as proposed".format(len(levels)))
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
    """The incumbent with these parts applied, fitted to the budget."""
    design = dict(incumbent)
    for part in parts:
        design.update(part)
    return typed_knobs(fit_to_budget(typed_knobs(design)))


def probe_wave(probe_problem, incumbent_entry, parts, tag, measure, round_number):
    """Sketches at the probe rung, measured together: the incumbent, each part alone on it,
    and all parts together. Every sketch is a design of the run. Deltas against the
    incumbent's own sketch, so the rung's bias cancels. None if the incumbent's sketch failed."""
    objective = probe_problem["objective"]
    incumbent = incumbent_entry["knobs"]
    planned = [("incumbent", typed_knobs(incumbent))]
    planned += [(name, with_parts(incumbent, [part])) for name, part in parts.items()]
    if len(parts) > 1:
        planned.append(("joint", with_parts(incumbent, list(parts.values()))))
    label_of = {}
    for label, design in planned:
        label_of.setdefault(probe_problem["name_of"](design), label)
    entries = measure(probe_problem, [design for _, design in planned],
                      lambda knobs: "sketch:" + label_of[probe_problem["name_of"](knobs)], round_number, incumbent_entry)
    values = {}
    for (label, _), entry in zip(planned, entries):
        if entry is not None:
            values.setdefault(label, entry["metrics"][objective])
    if "incumbent" not in values:
        note(tag, "-> the incumbent's sketch failed; no sketches this round")
        return None
    base = values["incumbent"]
    probes = {"rung": list(probe_problem["fidelity"]), "incumbent": base,
              "parts": {name: values[name] - base for name in parts if name in values},
              "failed": [name for name in parts if name not in values]}
    if "joint" in values:
        probes["joint"] = values["joint"] - base
        probes["interaction"] = probes["joint"] - sum(probes["parts"].values())
    note(tag, "-> sketches at {}: {}{}".format(
        "/".join("{}M".format(n // 1_000_000) for n in probes["rung"]),
        ", ".join("{} {:+.4f}".format(name, delta) for name, delta in probes["parts"].items()),
        (" | together {:+.4f}, interaction {:+.4f}".format(probes["joint"], probes["interaction"]) if "joint" in probes else "")
        + (" | could not be simulated: " + ", ".join(probes["failed"]) if probes["failed"] else "")))
    return probes


def concern_sketches(rounds, name):
    """One concern's proposals round by round and what their sketches said, including the
    rounds where nothing was committed."""
    lines = []
    for record in rounds:
        probes = record.get("probes")
        moved = (record.get("moves") or {}).get(name)
        if not probes or not moved or name not in probes["parts"]:
            continue
        lines.append("  round {} against D{}: {} | alone {:+.4f} | {}".format(
            record["round"], record["against"],
            ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())),
            probes["parts"][name], "committed" if name in record["include"] else "not committed"))
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
                cells.append("{} {} {:+.4f}".format(
                    name, ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())),
                    probes["parts"][name]))
        if "joint" in probes:
            cells.append("together {:+.4f}".format(probes["joint"]))
        lines.append("  round {} against D{}: {} | committed: {}".format(
            record["round"], record["against"], " | ".join(cells),
            " + ".join(record["include"]) if record["include"] else "nothing"))
    return "\n".join(lines) if lines else "  no sketches yet"


def refused_moves(rounds, index):
    """The moves sketched against design `index` and not committed, with what the sketch said:
    the tabu list of a stall, and where the flat moves - free parts of a coupled design - are
    read from."""
    refused = []
    for record in rounds:
        probes = record.get("probes")
        if record["against"] != index or not probes or not record.get("moves"):
            continue
        for name, moved in record["moves"].items():
            if name in probes.get("failed", []):
                refused.append((name, moved, None))
            elif name in probes["parts"] and name not in record["include"]:
                refused.append((name, moved, probes["parts"][name]))
    return refused


def refused_text(refused, tolerance):
    lines = ["  {}: {} | {}".format(
        name, ", ".join("{} {}->{}".format(knob, a, b) for knob, (a, b) in sorted(moved.items())),
        "the simulator could not measure it" if delta is None
        else "sketched {:+.4f}{}".format(delta, " (flat)" if abs(delta) <= tolerance else ""))
        for name, moved, delta in refused]
    return "\n".join(lines) if lines else "  none"


def sweep_level(rounds, index):
    """After a jump round against design `index` found nothing, the level the next one sweeps:
    the level those jump rounds have blamed least and sketched least, so a stall does not spend
    its budget on the levels the sheet keeps naming. None before the first empty jump."""
    jumps = [record for record in rounds if record["mode"] == "jump" and record["against"] == index]
    if not jumps:
        return None
    score = {level: 0 for level in ALL_LEVELS}
    for record in jumps:
        sheet = record.get("sheet") or {}
        for key in ("bottleneck", "counterfactual"):
            level = (sheet.get(key) or {}).get("level")
            if level in score:
                score[level] += 1
        for moved in record.get("moves", {}).values():
            for knob in moved:
                score[LEVEL_OF[knob.split("_")[0]]] += 1
    return min(ALL_LEVELS, key=lambda level: score[level])


def untried_text(history, knobs, current):
    """The values of these knobs never measured as a single change from the current design,
    sketches included, within the area budget: the neighbourhood the run has not looked at. A
    design that moved several knobs at once credits none of them alone, so a value carried only
    by such designs is untried here."""
    measured = {config_name(entry["knobs"]) for entry in history}
    lines = []
    for knob in knobs:
        untried = [str(value) for value in SEARCH_SPACE[knob]
                   if value != current[knob] and within_budget(dict(current, **{knob: value}))
                   and config_name(dict(current, **{knob: value})) not in measured]
        if untried:
            lines.append("  {}: {}".format(knob, ", ".join(untried)))
    return "\n".join(lines) if lines else "  every one-knob change from the current design has been measured"


def state_section(state):
    """What a jump round tells everyone before the task. Nothing in a climbing round."""
    if state is None or state["mode"] != "jump":
        return []
    lines = ["", "## The search is stalled",
             "  {} rounds in a row did not move the current design.".format(state["stalled"]),
             "  moves the sketches refused against it (none is worth re-proposing alone):",
             refused_text(state["refused"], state["tolerance"])]
    if state.get("sweep"):
        lines += ["  the jump rounds against this design found nothing at the levels the sheet blamed; this round "
                  "sweeps the {}: propose your untried one-knob move there if you have one, else at any of your "
                  "levels.".format(state["sweep"])]
    return lines


def current_heading(current_entry, objective):
    return "## The current design (the best this run has measured), D{} {}={:.4f}".format(
        current_entry["index"], objective, current_entry["metrics"][objective])


def sketch_text(probes):
    if probes is None:
        return "  no sketches this round"
    lines = ["  {} alone: {:+.4f}".format(name, delta) for name, delta in probes["parts"].items()]
    if "joint" in probes:
        lines.append("  all together: {:+.4f} | interaction (together minus the sum of the parts): {:+.4f}".format(
            probes["joint"], probes["interaction"]))
    return "\n".join(lines)


# ---------------------------------------------------------------- the synthesis ----

SYNTHESIS_TASK = """## Task
Compose this round's design from the proposals above. Include the proposals that belong together:
a move that depends on another (say which), or independent moves that do not conflict. Leave out a
proposal that the sketches or your sheet contradict, and say why. The sketches are cheap
simulations that show direction, not the number the measurement will give. You may not change any
proposed value; you choose which proposals are in. The design you compose is measured for real
and is the round's one counted design; when every sketch loses and no proposal is worth a real
measurement, include nothing, and the round costs no design.

Return JSON with exactly these keys:
- "include": the list of concern names whose proposals form the design; empty to commit nothing.
- "hypothesis": one sentence: what the design should show against the current design, in report
  numbers, and why the included parts belong together.
- "excluded": an object with one sentence per excluded concern, empty if none.
"""


JUMP_SYNTHESIS = """## This is a jump round
The search is stalled on the current design and the proposals are one-knob moves never measured
from it. The sketches decide: a part that lost by more than {tolerance:.4f} may be in the design
only if the "all together" sketch is not a loss, and the code enforces this. Compose from the parts
that gained: several attributable gains may go together when the "all together" sketch confirms
them. A design the sketches do not predict to gain is not measured, and the next round jumps
again. Include nothing if no part gained.

"""


def synthesize(problem, sheet, incumbent, proposals, probes, tag, state=None):
    """The analyst's second call: which of the proposals form the round's design, with the
    sketches in front of it. Returns (included concern names, hypothesis). On any failure
    every proposal is in."""
    names = list(proposals)
    parts = ["## Role",
             "You are the analyst of a team tuning one cache hierarchy. The specialists have proposed "
             "and every proposal has been sketched; you compose the round's design from the proposals, "
             "choosing among them and altering no value.",
             "",
             "## Chip", problem["chip_text"],
             "",
             "## Your sheet this round",
             sheet_text(sheet) if sheet is not None else "  (no sheet this round)",
             "",
             "## The current design", "  " + assignment(incumbent, list(SEARCH_SPACE)),
             "",
             "## The proposals"]
    for name in names:
        # A proposal may carry only the knobs it moves (the opening's parts do); show the whole concern.
        whole = dict(incumbent)
        whole.update(proposals[name]["knobs"])
        parts += ["  {}: {}".format(name, assignment(whole, CONCERNS[name]["knobs"])),
                  "    because: " + proposals[name]["reasoning"]]
    parts += ["",
              "## What the sketches say (change in {} against the current design, at the probe rung)".format(problem["objective"]),
              sketch_text(probes)]
    parts += ["", (JUMP_SYNTHESIS.format(tolerance=state["tolerance"]) if state is not None and state["mode"] == "jump" else "")
              + SYNTHESIS_TASK]
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


def predicted(probes, include):
    """What the sketches predict for the composed design: its own sketch when it was
    sketched (one part, or all of them), else the sum of its parts' sketches, which `commit`
    replaces with a sketch of the composed design itself."""
    if probes is None:
        return None
    if len(include) == 1 and include[0] in probes["parts"]:
        return probes["parts"][include[0]], True
    if len(include) == len(probes["parts"]) and "joint" in probes:
        return probes["joint"], True
    return sum(probes["parts"].get(name, 0.0) for name in include), False


# ---------------------------------------------------------------- the search ----

def search(problem, budget, seed, tag, probe_problem=None):
    """`budget` designs, sketches included: every design the run evaluates counts once, whether
    the simulator ran or the table answered. A round: the analyst's sheet, four proposals at
    once, the sketch wave, the analyst's composition, and the composed design measured if the
    wave did not. The incumbent is the best design measured at the run's fidelity, from a
    sketch or a composition alike."""
    objective = problem["objective"]
    fidelity = tuple(problem["fidelity"])
    started = time.time()
    history = []
    rounds = []
    by_name = {}            # (design name, rung) -> its entry: a design is evaluated once

    def record(knobs, metrics, source, round_number, hypothesis, against, rung):
        entry = {"index": len(history), "round": round_number, "name": problem["name_of"](knobs),
                 "knobs": knobs, "metrics": metrics, "source": source, "hypothesis": hypothesis,
                 "rung": list(rung), "against": None, "baseline": None, "moved": {}}
        if against is not None:
            entry["against"] = against["index"]
            entry["baseline"] = against["metrics"][objective]
            entry["moved"] = knobs_changed(knobs, against["knobs"])
        history.append(entry)
        by_name[(entry["name"], tuple(rung))] = entry
        note(tag, "\n======== D{} [{}] {}={:.4f} | {} ========".format(
            entry["index"], source, objective, metrics[objective],
            ", ".join("{}={}".format(knob, knobs[knob]) for knob in sorted(knobs_changed(knobs, problem["stock"]))) or "stock"))
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

    def committed_designs():
        """The record the council reads and stands on: the composed designs, not the sketches. Every sketch still counts against the budget and is in the report, and the
        jump round's neighbourhood list reads every design evaluated."""
        return [entry for entry in history if not entry["source"].startswith("sketch:")]

    def incumbent_of():
        return max((entry for entry in committed_designs() if tuple(entry["rung"]) == fidelity),
                   key=lambda entry: entry["metrics"][objective])

    def commit(incumbent_entry, proposals, sheet, source, round_number, state=None):
        """Sketch the proposals, compose, measure the composed design. Returns its entry, or
        None when nothing was composed or nothing new could be measured."""
        incumbent = incumbent_entry["knobs"]
        mode = state["mode"] if state is not None else "climb"
        parts = {name: proposals[name]["knobs"] for name in proposals}
        moves = {name: knobs_changed(with_parts(incumbent, [parts[name]]), incumbent) for name in parts}
        probes = probe_wave(probe_problem, incumbent_entry, parts, tag, measure, round_number) if probe_problem is not None else None
        if probes is not None and probes["failed"]:
            # A part the simulator could not measure is out of the round and on the refused list.
            proposals = {name: proposals[name] for name in proposals if name not in probes["failed"]}
            parts = {name: parts[name] for name in parts if name not in probes["failed"]}
        if not proposals:
            include, hypothesis = [], "nothing could be simulated"
        elif len(proposals) > 1:
            include, hypothesis = synthesize(problem, sheet, incumbent, proposals, probes, tag, state)
        else:
            # One proposal: nothing to compose, the sketch alone decides below.
            include, hypothesis = list(proposals), next(iter(proposals.values()))["reasoning"]
        if probes is not None and include and mode == "jump":
            # A jump may not carry a part the sketches refused, unless all the parts together
            # are not a loss.
            tolerance = FLAT_SHARE * probes["incumbent"]
            losers = [name for name in include if probes["parts"].get(name, 0.0) < -tolerance]
            together_ok = ("joint" in probes and probes["joint"] >= -tolerance
                           and len(include) == len(probes["parts"]))
            if losers and not together_ok:
                note(tag, "-> a jump may not carry parts the sketches refused ({}); left out".format(", ".join(losers)))
                include = [name for name in include if name not in losers]
        if probes is not None and include:
            value, exact = predicted(probes, include)
            if not exact:
                # A composed subset nobody sketched: sketch it now, so the veto reads a
                # measurement rather than the sum of its parts. It is a design of the run.
                composed = with_parts(incumbent, [parts[name] for name in include])
                sketched = measure(probe_problem, [composed], "sketch:" + "+".join(include), round_number, incumbent_entry)[0]
                if sketched is not None:
                    value, exact = sketched["metrics"][objective] - probes["incumbent"], True
                    note(tag, "-> sketched the composed design {}: {:+.4f}".format(" + ".join(include), value))
            probes = dict(probes, predicted=value, exact=exact, include=include)
            # The sketches veto: a design they predict to lose is not measured, whoever composed
            # it.
            if value <= 0.0:
                note(tag, "-> the sketches predict {:+.4f} for {}: vetoed, nothing composed".format(value, " + ".join(include)))
                include = []
        rounds.append({"round": round_number, "mode": mode, "against": incumbent_entry["index"], "sheet": sheet,
                       "proposals": proposals, "moves": moves, "include": include,
                       "hypothesis": hypothesis, "probes": probes})
        if not include:
            note(tag, "-> nothing composed this round; the sketches stay in the ledger")
            return None
        design = with_parts(incumbent, [parts[name] for name in include])
        if not knobs_changed(design, incumbent):
            note(tag, "-> the composed design is the current design")
            return None
        label = source if len(include) > 1 else "council:" + include[0]
        entry = measure(problem, [design], label, round_number, incumbent_entry, hypothesis)[0]
        if entry is None:
            print("[{}] round {} | the composed design could not be measured".format(tag, round_number), flush=True)
            return None
        if entry["source"].startswith("sketch:"):
            entry["source"] = label          # the wave measured it; the analyst chose it
            entry["hypothesis"] = hypothesis
        note(tag, "-> composed: D{} {}={:.4f}, {:+.4f} against D{}".format(
            entry["index"], objective, entry["metrics"][objective],
            entry["metrics"][objective] - incumbent_entry["metrics"][objective], incumbent_entry["index"]))
        return entry

    stock_entry = measure(problem, [typed_knobs(problem["stock"])], "stock", 0, None)[0]

    # Round 1: the analyst's sheet for the stock chip and its opening, split by concern.
    round_number = 1
    note(tag, "\n\n################ round 1 | incumbent D0 {}={:.4f} | opening ################".format(
        objective, stock_entry["metrics"][objective]))
    sheet = diagnose(problem, history, stock_entry, tag, opening=True, rounds=rounds)
    previous = sheet
    opening = opening_design(sheet, problem, tag)
    if opening is not None:
        proposals = {name: {"knobs": part, "reasoning": sheet["opening_reasoning"]}
                     for name, part in parts_of(opening, problem["stock"]).items()}
        commit(stock_entry, proposals, sheet, "council:opening", 1)

    # A round that improves nothing costs its sketches, so the budget ends the search, or the
    # cap on rounds. A stalled search keeps jumping, its refused list growing.
    max_rounds = 3 * budget
    stalled = 0             # rounds in a row in which the incumbent did not move
    while len(history) - 1 < budget and round_number < max_rounds:
        incumbent_entry = incumbent_of()
        mode = "jump" if stalled >= STALL_ROUNDS else "climb"
        current_entry = incumbent_entry
        current = current_entry["knobs"]
        state = {"mode": mode, "stalled": stalled,
                 "refused": refused_moves(rounds, current_entry["index"]),
                 "sweep": sweep_level(rounds, current_entry["index"]) if mode == "jump" else None,
                 "tolerance": FLAT_SHARE * current_entry["metrics"][objective]}
        round_number += 1
        note(tag, "\n\n################ round {} | {}{} | current D{} {}={:.4f} | incumbent D{} | {} of {} designs ################".format(
            round_number, mode, " " + state["sweep"] if state["sweep"] else "", current_entry["index"], objective,
            current_entry["metrics"][objective], incumbent_entry["index"], len(history) - 1, budget))
        committed = committed_designs()
        sheet = diagnose(problem, committed, current_entry, tag, previous=previous, rounds=rounds, state=state, evaluated=history)
        previous = sheet if sheet is not None else previous

        logs = {name: [] for name in ORDER}
        with ThreadPoolExecutor(max_workers=len(ORDER)) as pool:
            futures = {name: pool.submit(specialist, problem, name, committed, current, current_entry, tag, sheet, logs[name], rounds, state, history)
                       for name in ORDER}
            answers = {name: future.result() for name, future in futures.items()}
        for name in ORDER:
            for text in logs[name]:
                note(tag, text)
        # A proposal counts as a move only if it changes the design once typed and fitted; a
        # move the sketches already refused against this design, or a design the simulator
        # could not measure, is a hold and costs no sketch.
        proposals = {}
        for name in ORDER:
            if answers[name] is None:
                continue
            design = with_parts(current, [answers[name]["knobs"]])
            moved = knobs_changed(design, current)
            if not moved:
                continue
            if not problem.get("is_candidate", lambda knobs: True)(design):
                note(tag, "-> {}'s design is one the simulator could not measure before; taken as a hold".format(name))
                continue
            if any(other == name and refused == moved for other, refused, _ in state["refused"]):
                note(tag, "-> {} re-proposed a move the sketches refused against D{}; taken as a hold".format(
                    name, current_entry["index"]))
                continue
            proposals[name] = answers[name]
        source = "council:joint"
        if not proposals:
            # Holds can wait on one another; the analyst, who reads every level, takes the turn
            # with one design of its own.
            note(tag, "-> every specialist held; the analyst takes the turn")
            sheet = diagnose(problem, committed, current_entry, tag, previous=previous, rounds=rounds, turn=True, state=state, evaluated=history)
            previous = sheet if sheet is not None else previous
            design = opening_design(sheet, problem, tag, base=current) if sheet is not None else None
            if design is not None:
                proposals = {name: {"knobs": part, "reasoning": sheet["opening_reasoning"]}
                             for name, part in parts_of(design, current).items()}
                source = "council:analyst"
        if proposals:
            commit(current_entry, proposals, sheet, source, round_number, state)

        # What the round did: the incumbent is the best design measured, a sketch included.
        best_now = incumbent_of()
        improved = best_now["index"] != incumbent_entry["index"]
        stalled = 0 if improved else stalled + 1
        if not improved:
            print("[{}] round {} | {} | the incumbent did not move".format(tag, round_number, mode), flush=True)
        if stalled >= STALL_ROUNDS:
            note(tag, "-> {} rounds did not move the incumbent D{}: the next round is a jump round".format(
                stalled, incumbent_entry["index"]))
    if round_number >= max_rounds:
        print("[{}] {} rounds: stopping".format(tag, round_number), flush=True)
    print("[{}] {} designs evaluated, best D{} {}={:.4f}".format(
        tag, len(history) - 1, incumbent_of()["index"], objective, incumbent_of()["metrics"][objective]), flush=True)
    return {"designs": history, "rounds": rounds}
