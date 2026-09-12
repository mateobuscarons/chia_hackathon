"""The memory: one shelf of CASES, one per workload the chip has been searched on.

A case is what a past search left behind, written by the code from the result
table, never by the LLM: the workload's descriptors, the stock and best designs
with their IPC, and every measured single-knob effect (controlled pairs: two
measured designs that differ in exactly one knob). Retrieval is by descriptor
distance. The agent never reads the cases raw: `digest` turns the nearest ones
into conclusions: the moves that paid everywhere, the traps, where the best
designs disagree, and the pooled design to copy and adapt (the one that did best
across every remembered workload).

  python -m loop.memory build <out.json> <trace> [trace ...]   # cases from the cached tables
  python -m loop.memory leave_one_out <memory traces...> -- <test traces...>   # free check: do remembered effects keep their sign elsewhere?
"""

import json
import math
import os
import sys

from loop import champsim_problem
from loop.configs import CHIP, KNOB_PRIORITY, SEARCH_SPACE, knobs_changed, same_knobs, typed_knobs, within_budget

NEAREST_CASES = 3        # per workload of the suite
MIN_PAIRS = 2            # an effect needs this many controlled pairs to count


def empty():
    return {"cases": [], "pooled": None}


def load(path):
    if path is None or not os.path.exists(path):
        return empty()
    with open(path) as memory_file:
        return json.load(memory_file)


def save(memory, path):
    with open(path, "w") as memory_file:
        json.dump(memory, memory_file, indent=1)


# ---------------------------------------------------------------- effects ----

def one_knob_effects(rows, objective="ipc"):
    """Every controlled pair among `rows` (two designs differing in exactly one
    knob), grouped by (knob, from, to): mean effect in percent and pair count,
    largest first. `rows` carry "knobs" and the objective under "ipc" or in "metrics"."""
    groups = {}
    for index_a in range(len(rows)):
        for index_b in range(index_a + 1, len(rows)):
            knobs_a = rows[index_a]["knobs"]
            knobs_b = rows[index_b]["knobs"]
            differing = []
            for knob in SEARCH_SPACE:
                if str(knobs_a.get(knob)) != str(knobs_b.get(knob)):
                    differing.append(knob)
                    if len(differing) > 1:
                        break
            if len(differing) != 1:
                continue
            knob = differing[0]
            value_a = row_value(rows[index_a], objective)
            value_b = row_value(rows[index_b], objective)
            if value_a <= 0 or value_b <= 0:
                continue
            # The step is reported in the space's own value order (from -> to).
            order = []
            for value in SEARCH_SPACE[knob]:
                order.append(str(value))
            if order.index(str(knobs_a[knob])) < order.index(str(knobs_b[knob])):
                key = (knob, str(knobs_a[knob]), str(knobs_b[knob]))
                gain = 100.0 * (value_b / value_a - 1.0)
            else:
                key = (knob, str(knobs_b[knob]), str(knobs_a[knob]))
                gain = 100.0 * (value_a / value_b - 1.0)
            groups.setdefault(key, []).append(gain)
    effects = []
    for (knob, low, high), gains in groups.items():
        effects.append({"knob": knob, "from": low, "to": high, "effect_pct": sum(gains) / len(gains), "pairs": len(gains)})
    effects.sort(key=lambda effect: -abs(effect["effect_pct"]))
    return effects


def row_value(row, objective):
    if objective in row:
        return row[objective]
    return row["metrics"][objective]


# ---------------------------------------------------------------- build ----

def case_from_table(trace_path, only_names=None):
    """One workload's case from its cached result table (None if the table is empty).

    `only_names` keeps a case to the designs one build asked for, so two memories
    built by different procedures can share the simulation cache without mixing."""
    name = champsim_problem.trace_short_name(trace_path)
    table = champsim_problem.load_table(champsim_problem.table_path(name))
    rows = champsim_problem.measured_rows(table)
    if only_names is not None:
        rows = keep_named(rows, only_names)
    if len(rows) == 0:
        return None
    stock = champsim_problem.stock_design()
    stock_ipc = None
    best = rows[0]
    for row in rows:
        if same_knobs(row["knobs"], stock):
            stock_ipc = row["ipc"]
        if row["ipc"] > best["ipc"]:
            best = row
    return {"chip": CHIP, "workload": name,
            "descriptors": champsim_problem.workload_descriptors(trace_path),
            "stock_design": stock, "stock_ipc": stock_ipc,
            "best_design": best["knobs"], "best_ipc": best["ipc"],
            "designs_measured": len(rows),
            "effects": one_knob_effects(rows)}


def build(trace_paths, output_path, only_names=None):
    """Cases from the tables, plus the pooled default design, frozen here so every
    run of the cell reads the same one."""
    memory = empty()
    rows_by_workload = {}
    for trace_path in trace_paths:
        case = case_from_table(trace_path, only_names)
        if case is None:
            print("[memory] no table for {}, skipped".format(trace_path), flush=True)
            continue
        case["id"] = "CASE-{:02d}".format(len(memory["cases"]) + 1)
        memory["cases"].append(case)
        rows_by_workload[case["workload"]] = table_rows(case["workload"], only_names)
        print("[memory] {} {}: {} designs, stock {} -> best {:.4f}, {} effects ({} with >= {} pairs)".format(
            case["id"], case["workload"], case["designs_measured"], format_ipc(case["stock_ipc"]), case["best_ipc"],
            len(case["effects"]), count_solid(case["effects"]), MIN_PAIRS), flush=True)
    pooled = pooled_design(memory["cases"], rows_by_workload, pooled_guard(memory["cases"]))
    if pooled is not None:
        memory["pooled"] = {"knobs": pooled[1], "mean_share": pooled[0], "workloads_measured": pooled[2]}
        print("[memory] pooled default: mean share {:.0f}% over {} workloads".format(100.0 * pooled[0], pooled[2]), flush=True)
    save(memory, output_path)
    print("[memory] {} cases -> {}".format(len(memory["cases"]), output_path), flush=True)
    return memory


def table_rows(workload, only_names=None):
    rows = champsim_problem.measured_rows(champsim_problem.load_table(champsim_problem.table_path(workload)))
    if only_names is None:
        return rows
    return keep_named(rows, only_names)


def keep_named(rows, only_names):
    kept = []
    for row in rows:
        if row["name"] in only_names:
            kept.append(row)
    return kept


def pooled_guard(cases):
    """A pooled design must be measured on all the memory workloads but one (at least two)."""
    return max(2, len(cases) - 1)


def format_ipc(value):
    if value is None:
        return "n/a"
    return "{:.4f}".format(value)


def count_solid(effects):
    count = 0
    for effect in effects:
        if effect["pairs"] >= MIN_PAIRS:
            count += 1
    return count


# ---------------------------------------------------------------- retrieval ----

def transformed(name, value):
    """Counts and sizes are compared on a log scale; ratios and fractions as they are."""
    if value is None:
        return 0.0
    if abs(value) > 1.5 and value >= 0:
        return math.log(value + 1.0)
    return value


def standardization(cases):
    """Mean and spread per descriptor over the cases in memory: the population the
    distances are measured in."""
    stats = {}
    for name in champsim_problem.DESCRIPTOR_METRICS:
        values = []
        for case in cases:
            values.append(transformed(name, case["descriptors"].get(name)))
        mean = sum(values) / len(values)
        spread = 0.0
        for value in values:
            spread += (value - mean) ** 2
        spread = math.sqrt(spread / len(values))
        if spread <= 0:
            spread = 1.0
        stats[name] = (mean, spread)
    return stats


def distance(descriptors_a, descriptors_b, stats):
    total = 0.0
    for name in stats:
        mean, spread = stats[name]
        a = (transformed(name, descriptors_a.get(name)) - mean) / spread
        b = (transformed(name, descriptors_b.get(name)) - mean) / spread
        total += (a - b) ** 2
    return math.sqrt(total)


def pairwise_distances(cases, stats):
    values = []
    for index_a in range(len(cases)):
        for index_b in range(index_a + 1, len(cases)):
            values.append(distance(cases[index_a]["descriptors"], cases[index_b]["descriptors"], stats))
    return sorted(values)


def percentile_of(value, sorted_values):
    if len(sorted_values) == 0:
        return None
    below = 0
    for other in sorted_values:
        if other < value:
            below += 1
    return 100.0 * below / len(sorted_values)


def retrieve(memory, problem):
    """The nearest cases per workload of the suite. Returns None on an empty memory,
    else {"stats", "pairwise", "by_workload": {workload: [(distance, case)]},
    "cases": {case id: (smallest distance, the workload it is nearest to)}}."""
    if len(memory["cases"]) == 0:
        return None
    stats = standardization(memory["cases"])
    by_workload = {}
    nearest_of = {}
    for workload in problem["workloads"]:
        scored = []
        for case in memory["cases"]:
            scored.append((distance(problem["descriptors"][workload], case["descriptors"], stats), case))
        scored.sort(key=lambda pair: pair[0])
        by_workload[workload] = scored[:NEAREST_CASES]
        for case_distance, case in by_workload[workload]:
            if case["id"] not in nearest_of or nearest_of[case["id"]][0] > case_distance:
                nearest_of[case["id"]] = (case_distance, workload)
    return {"stats": stats, "pairwise": pairwise_distances(memory["cases"], stats),
            "by_workload": by_workload, "cases": nearest_of}


# ---------------------------------------------------------------- the digest the agent reads ----

CONSENSUS_SHOWN = 6
TRAPS_SHOWN = 4
DISAGREEMENTS_SHOWN = 5
MIXED_SHOWN = 4
CONTRASTS_PER_WORKLOAD = 2
EFFECT_FLOOR_PCT = 1.0


def similarity_word(case_distance, pairwise):
    percentile = percentile_of(case_distance, pairwise)
    if percentile is None or percentile < 25.0:
        return "high"
    if percentile < 60.0:
        return "medium"
    return "low"


def nearest_cases_in_order(memory, retrieval):
    """The retrieved cases, nearest first."""
    ordered = []
    for case in memory["cases"]:
        if case["id"] in retrieval["cases"]:
            ordered.append((retrieval["cases"][case["id"]][0], case))
    ordered.sort(key=lambda pair: pair[0])
    cases = []
    for case_distance, case in ordered:
        cases.append(case)
    return cases


def effects_from_stock(cases, stock):
    """Every remembered single-knob step that starts at the stock value, grouped by
    (knob, target value): [(workload, effect_pct, pairs)], the reverse step read backwards."""
    grouped = {}
    for case in cases:
        for effect in case["effects"]:
            if effect["pairs"] < MIN_PAIRS:
                continue
            knob = effect["knob"]
            if effect["from"] == str(stock[knob]):
                key = (knob, effect["to"])
                value = effect["effect_pct"]
            elif effect["to"] == str(stock[knob]):
                key = (knob, effect["from"])
                value = -effect["effect_pct"]
            else:
                continue
            grouped.setdefault(key, []).append((case["workload"], value, effect["pairs"]))
    return grouped


def effect_line(rank, key, entries):
    knob, value = key
    parts = []
    least_pairs = None
    for workload, effect, pairs in entries:
        parts.append("{:+.0f}% ({})".format(effect, short_workload(workload)))
        if least_pairs is None or pairs < least_pairs:
            least_pairs = pairs
    solid = "solid"
    if least_pairs < 3:
        solid = "weak evidence"
    return "{}. {} -> {}: {}   {}".format(rank, knob, value, ", ".join(parts), solid)


def short_workload(workload):
    """"605.mcf_s-665B" -> "mcf"; "sierra.a.4_0000" -> "sierra.a.4"; "bfs.urand-36B" -> "bfs.urand"."""
    name = workload
    if "." in name and name.split(".")[0].isdigit():
        name = name.split(".", 1)[1]
    for separator in ["_s-", "-", "_0"]:
        if separator in name:
            name = name.split(separator)[0]
    return name


def digest(memory, problem, retrieval):
    """The memory as the agent reads it: conclusions first, computed from the
    nearest cases, anchored on the stock chip the agent starts from, ending with
    the pooled design to copy and adapt."""
    if retrieval is None:
        return ""
    stock = problem["stock"]
    stats = retrieval["stats"]
    lines = ["## What earlier searches on this chip say about workloads like yours"]
    similarity = []
    contrasts = []
    for workload in problem["workloads"]:
        case_distance, case = retrieval["by_workload"][workload][0]
        similarity.append("{} is closest to {} (similarity {})".format(
            short_workload(workload), short_workload(case["workload"]), similarity_word(case_distance, retrieval["pairwise"])))
        gaps = []
        for name in stats:
            mean, spread = stats[name]
            here = problem["descriptors"][workload].get(name)
            there = case["descriptors"].get(name)
            gap = abs(transformed(name, here) - transformed(name, there)) / spread
            gaps.append((gap, name, here, there))
        gaps.sort(key=lambda item: -item[0])
        for gap, name, here, there in gaps[:CONTRASTS_PER_WORKLOAD]:
            contrasts.append("{} {} {:.3g} vs {:.3g} for {}".format(short_workload(workload), name, here, there, short_workload(case["workload"])))
    lines.append("Similarity: " + "; ".join(similarity) + ".")
    lines.append("Where yours differ most from them: " + "; ".join(contrasts) + ".")
    cases = nearest_cases_in_order(memory, retrieval)
    # Consensus and traps are read from each workload's CLOSEST case only; the
    # wider circle of retrieved cases supplies the workload-dependent moves.
    closest_ids = set()
    for workload in problem["workloads"]:
        closest_ids.add(retrieval["by_workload"][workload][0][1]["id"])
    closest = []
    for case in cases:
        if case["id"] in closest_ids:
            closest.append(case)
    consensus = []
    traps = []
    for key, entries in effects_from_stock(closest, stock).items():
        # One workload's word counts only when it rests on three or more controlled pairs.
        if len(entries) < 2 and entries[0][2] < 3:
            continue
        values = []
        for workload, effect, pairs in entries:
            values.append(effect)
        mean = sum(values) / len(values)
        if min(values) > EFFECT_FLOOR_PCT:
            consensus.append((mean, key, entries))
        elif max(values) < -EFFECT_FLOOR_PCT:
            traps.append((mean, key, entries))
    consensus.sort(key=lambda item: -item[0])
    traps.sort(key=lambda item: item[0])
    mixed = []
    for key, entries in effects_from_stock(cases, stock).items():
        if len(entries) < 2:
            continue
        values = []
        for workload, effect, pairs in entries:
            values.append(effect)
        if max(values) > EFFECT_FLOOR_PCT and min(values) < -EFFECT_FLOOR_PCT:
            mixed.append((max(values) - min(values), key, entries))
    mixed.sort(key=lambda item: -item[0])
    lines.append("")
    lines.append("Moves from the stock chip that paid on the closest remembered workloads (per-workload IPC change):")
    if len(consensus) == 0:
        lines.append("(none agreed on)")
    for rank, (mean, key, entries) in enumerate(consensus[:CONSENSUS_SHOWN]):
        lines.append(effect_line(rank + 1, key, entries))
    lines.append("")
    lines.append("Traps: moves from the stock chip that hurt on the closest remembered workloads:")
    if len(traps) == 0:
        lines.append("(none)")
    for rank, (mean, key, entries) in enumerate(traps[:TRAPS_SHOWN]):
        lines.append(effect_line(rank + 1, key, entries))
    disagreements = []
    for knob in KNOB_PRIORITY:
        values = {}
        for case in cases:
            values[short_workload(case["workload"])] = str(case["best_design"].get(knob, stock[knob]))
        distinct = set(values.values())
        if len(distinct) < 2:
            continue
        deviates = False
        for value in distinct:
            if value != str(stock[knob]):
                deviates = True
        if not deviates:
            continue
        parts = []
        for workload in values:
            parts.append("{} {}".format(workload, values[workload]))
        disagreements.append("{}: {} (stock {})".format(knob, ", ".join(parts), stock[knob]))
    lines.append("")
    lines.append("Where their best designs DISAGREE (the questions your search should settle):")
    if len(disagreements) == 0:
        lines.append("(they agree on every knob)")
    for line in disagreements[:DISAGREEMENTS_SHOWN]:
        lines.append("- " + line)
    for spread, key, entries in mixed[:MIXED_SHOWN]:
        knob, value = key
        parts = []
        for workload, effect, pairs in entries:
            parts.append("{:+.0f}% ({})".format(effect, short_workload(workload)))
        lines.append("- {} -> {} is workload-dependent: {}".format(knob, value, ", ".join(parts)))
    pooled = memory.get("pooled")
    if pooled is not None:
        lines.append("")
        lines.append("The remembered design that did best across the remembered workloads (measured on {} of them, "
                     "reaching on average {:.0f}% of each one's stock-to-best gap), to copy and adapt:".format(
                         pooled["workloads_measured"], 100.0 * pooled["mean_share"]))
        lines.append(json.dumps(fit_to_budget(pooled["knobs"])))
    return "\n".join(lines)


# ---------------------------------------------------------------- the pooled design ----

def fit_to_budget(knobs):
    """A remembered design over this chip's budget: shrink the LLC one step at a
    time, then the L2, until it fits."""
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


# ---------------------------------------------------------------- offline check ----

def leave_one_out(memory_traces, test_traces):
    """Free check on the cached tables: do the single-knob effects a case records
    keep their sign on a workload it was not measured on? Each memory workload is
    held out against the rest, each test workload against the whole memory, and
    then every ordered pair of memory workloads is scored by descriptor distance,
    which is the transfer-versus-distance curve."""
    cases = []
    for trace_path in memory_traces:
        case = case_from_table(trace_path)
        if case is None:
            continue
        case["id"] = "CASE-{:02d}".format(len(cases) + 1)
        cases.append(case)
    tests = []
    for trace_path in test_traces:
        case = case_from_table(trace_path)
        if case is not None:
            tests.append(case)
    if len(cases) < 2:
        print("need at least two memory workloads with tables")
        return
    stats = standardization(cases)
    print("== do a remembered workload's measured effects keep their sign on another workload?")
    print("   {:<20s} {:<20s} {:>8s} {:>9s} {:>9s}".format("workload", "closest remembered", "distance", "effects", "sign held"))
    for held in cases + tests:
        others = []
        for case in cases:
            if case["id"] != held.get("id"):
                others.append(case)
        closest = None
        closest_distance = None
        for case in others:
            case_distance = distance(held["descriptors"], case["descriptors"], stats)
            if closest_distance is None or case_distance < closest_distance:
                closest_distance = case_distance
                closest = case
        checked, survived = sign_survival(closest, held)
        print("   {:<20s} {:<20s} {:8.2f} {:9d} {:9d}".format(
            held["workload"][:20], closest["workload"][:20], closest_distance, checked, survived))
    print()
    print("== sign survival of remembered effects against distance, over every ordered pair of memory workloads")
    points = []
    for source in cases:
        for target in cases:
            if source["id"] == target["id"]:
                continue
            pair_distance = distance(source["descriptors"], target["descriptors"], stats)
            checked, survived = sign_survival(source, target)
            for index in range(checked):
                points.append((pair_distance, index < survived))
    points.sort(key=lambda point: point[0])
    if len(points) == 0:
        print("   no shared controlled pairs between any two workloads")
        return
    thirds = 3
    size = len(points) // thirds
    for third in range(thirds):
        chunk = points[third * size:(third + 1) * size]
        if third == thirds - 1:
            chunk = points[third * size:]
        held_count = 0
        for point in chunk:
            if point[1]:
                held_count += 1
        print("   distance {:.2f} .. {:.2f}: {} of {} effects kept their sign ({:.0f}%)".format(
            chunk[0][0], chunk[-1][0], held_count, len(chunk), 100.0 * held_count / len(chunk)))


def pooled_design(cases, rows_by_workload, min_cases):
    """The design with the best mean gap share over the remembered workloads that
    measured it, counting only designs measured on at least `min_cases` of them.
    Returns (mean share, knobs, how many workloads) or None."""
    shares = {}
    knobs_of = {}
    for case in cases:
        if case["stock_ipc"] is None or case["best_ipc"] <= case["stock_ipc"]:
            continue
        gap = case["best_ipc"] - case["stock_ipc"]
        for row in rows_by_workload[case["workload"]]:
            shares.setdefault(row["name"], []).append((row["ipc"] - case["stock_ipc"]) / gap)
            knobs_of[row["name"]] = row["knobs"]
    best = None
    for name, values in shares.items():
        if len(values) < min_cases:
            continue
        mean = sum(values) / len(values)
        if best is None or mean > best[0]:
            best = (mean, knobs_of[name], len(values))
    return best


def sign_survival(source, target):
    """How many of the source case's solid effects the target workload has also
    measured, and how many of those kept their sign."""
    target_effects = {}
    for effect in target["effects"]:
        target_effects[(effect["knob"], effect["from"], effect["to"])] = effect["effect_pct"]
    checked = 0
    survived = 0
    for effect in source["effects"]:
        if effect["pairs"] < MIN_PAIRS:
            continue
        key = (effect["knob"], effect["from"], effect["to"])
        if key not in target_effects:
            continue
        checked += 1
        if (target_effects[key] > 0) == (effect["effect_pct"] > 0):
            survived += 1
    return checked, survived


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[3:], sys.argv[2])
    elif sys.argv[1] == "leave_one_out":
        arguments = sys.argv[2:]
        test_traces = []
        if "--" in arguments:
            test_traces = arguments[arguments.index("--") + 1:]
            arguments = arguments[:arguments.index("--")]
        leave_one_out(arguments, test_traces)
