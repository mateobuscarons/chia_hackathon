"""The agent's memory: written entirely by the system from experience, nothing
seeded by us, so an empty memory makes the memory arms behave exactly like the
plain agent. Simulator-agnostic: knob names, descriptor names and workload names
are read from the problem dict and never written into this file.

Shelves (memory.json):
  cases      one per (chip, workload) of a run: descriptors, baseline and best
             design, best metric, top single-knob effects. Provenance only in
             its provenance field.
  facts      one per controlled pair measured on one workload: knob, from, to,
             effect in percent, pairs, descriptors of where it was measured, a
             record (cited / improved / worsened / direction_wrong_at) and a
             similarity gate that tightens when the fact's direction fails somewhere.
  strategies procedural notes written by the reflection step after a run:
             kind "procedure", "failure" or "structure"; qualitative, knob and
             descriptor names only, no numbers, no workload or chip names (a
             linter enforces it); evidence points at designs of the run.
  forecast_log  every forecast the agent made (run_id, design, predicted,
             confidence, measured).
Nothing is deleted; losing items are re-scoped (facts) or demoted (strategies).

Read path: the prompt is assembled from data only. Distances and records are
printed as numbers; the prompt never says what to trust. The plain agent
(llm_direct) is this agent with no shelf open; an empty memory gives the same prompt.

  python -m loop.memory build <memory.json> <chip,chip> <trace.xz> [more]   # cases + facts from the tables, strategies written from the cases
  python -m loop.memory consolidate <memory.json> <out.json> <cell>         # merge the cell's run memories for the next cell
  python -m loop.memory check                                               # the offline acceptance checks
"""

import json
import math
import os
import re

from loop import analyst, forecast, playbook
from loop.configs import typed_knobs, within_budget
from loop.champsim_problem import BASE_CONFIG, aggregate_suite, enrich_metrics, trace_short_name

NEAREST_CASES = 3
FACTS_IN_PROMPT = 10
STRATEGIES_IN_PROMPT = 12
TOP_EFFECTS = 5
# A case shows a single-knob effect only when at least this many controlled pairs back it.
MIN_PAIRS_SHOWN = 3
BUCKETS = {"small": (0.0, 3.0), "medium": (3.0, 10.0), "large": (10.0, 1000.0)}
DEMOTE_AFTER_CITATIONS = 4
FAMILY_WORDS = ["SPEC", "GAP", "CPU2017", "SPEC2017", "SPEC2006"]


def empty():
    return {"cases": [], "facts": [], "strategies": [], "forecast_log": []}


def load(path):
    if path is None or not os.path.exists(path):
        return empty()
    with open(path) as memory_file:
        return json.load(memory_file)


def save(memory, path):
    with open(path, "w") as memory_file:
        json.dump(memory, memory_file, indent=1)


def next_id(memory, shelf, prefix):
    return "{}-{:04d}".format(prefix, len(memory[shelf]) + 1)


# ---------------------------------------------------------------- per-workload metrics ----

def workload_metric_key(problem, workload):
    """The metric of one workload inside a suite result ("mcf:ipc"), or the plain
    objective for a single-workload problem."""
    if "holders" in problem:
        return workload + ":" + problem["objective"]
    return problem["objective"]


def bucket_of(effect_pct):
    magnitude = abs(effect_pct)
    for name in ["small", "medium", "large"]:
        low, high = BUCKETS[name]
        if low <= magnitude < high:
            return name
    return "large"


# ---------------------------------------------------------------- distances ----

def descriptor_names(memory, problem):
    names = set(problem["descriptors"].keys())
    for case in memory["cases"]:
        names = names | set(case["descriptors"].keys())
    kept = []
    for name in sorted(names):
        if not name.startswith("max_"):
            kept.append(name)
    return kept


def standardization(memory, problem):
    """Mean and spread per descriptor over the cases in memory (the population the
    distances are measured in). Log scale for descriptors that span decades."""
    names = descriptor_names(memory, problem)
    stats = {}
    for name in names:
        values = []
        for case in memory["cases"]:
            if name in case["descriptors"]:
                values.append(transformed(name, case["descriptors"][name]))
        if len(values) == 0:
            values = [transformed(name, problem["descriptors"].get(name, 0.0))]
        mean = sum(values) / len(values)
        spread = 0.0
        for value in values:
            spread += (value - mean) ** 2
        spread = math.sqrt(spread / len(values))
        if spread <= 0:
            spread = 1.0
        stats[name] = (mean, spread)
    return stats


def transformed(name, value):
    """Descriptors that are counts or sizes are compared on a log scale; ratios and fractions as they are."""
    if value is None:
        return 0.0
    if abs(value) > 1.5 and value >= 0:
        return math.log(value + 1.0)
    return value


def distance(descriptors_a, descriptors_b, stats):
    total = 0.0
    for name in stats:
        mean, spread = stats[name]
        a = (transformed(name, descriptors_a.get(name, mean)) - mean) / spread
        b = (transformed(name, descriptors_b.get(name, mean)) - mean) / spread
        total += (a - b) ** 2
    return math.sqrt(total)


def pairwise_case_distances(memory, stats):
    values = []
    cases = memory["cases"]
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


def nearest_cases(memory, descriptors, stats, how_many=NEAREST_CASES):
    scored = []
    for case in memory["cases"]:
        scored.append((distance(descriptors, case["descriptors"], stats), case))
    scored.sort(key=lambda pair: pair[0])
    return scored[:how_many]


# ---------------------------------------------------------------- linter ----

def provenance_names(memory, problem):
    """Every chip and workload name the memory or the problem knows: the words a
    strategy may not contain."""
    names = set()
    for case in memory["cases"]:
        names.add(case["provenance"]["chip"])
        names.add(case["provenance"]["workload"])
    for fact in memory["facts"]:
        names.add(fact["provenance"]["chip"])
        names.add(fact["provenance"]["workload"])
    names.add(problem["soc_name"])
    for workload in problem["workload_descriptors"]:
        names.add(workload)
    expanded = set()
    for name in names:
        expanded.add(name)
        for piece in re.split(r"[_./-]", name):
            if len(piece) >= 3 and not piece.isdigit():
                expanded.add(piece)
    return expanded


def lint_strategy(text, memory, problem):
    """None when the strategy is admissible, else the reason. A strategy may name
    knobs and descriptors only: no chip, workload or family name, no number."""
    lowered = text.lower()
    for name in provenance_names(memory, problem):
        if re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])", lowered):
            return "names '{}'".format(name)
    for word in FAMILY_WORDS:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", text):
            return "names the family '{}'".format(word)
    stripped = text
    for knob in problem["search_space"]:
        stripped = stripped.replace(knob, " ")
    for descriptor in problem["descriptors"]:
        stripped = stripped.replace(descriptor, " ")
    stripped = re.sub(r"(?i)\b(l1d|l1i|l1|l2c|l2|llc|l3)\b", " ", stripped)
    if re.search(r"\d", stripped):
        return "contains a number"
    return None


# ---------------------------------------------------------------- write path ----

def case_from_run(problem, history, workload, run_id):
    key = workload_metric_key(problem, workload)
    best = None
    baseline_value = None
    for entry in history:
        if forecast.same_knobs(entry["knobs"], problem["baseline"]):
            baseline_value = entry["metrics"][key]
        if best is None or entry["metrics"][key] > best["metrics"][key]:
            best = entry
    effects = []
    for effect in forecast.one_knob_effects(history, key):
        if effect["pairs"] < MIN_PAIRS_SHOWN:
            continue
        effects.append({"knob": effect["knob"], "from": effect["from"], "to": effect["to"],
                        "effect": effect["gain_pct"], "pairs": effect["pairs"]})
        if len(effects) == TOP_EFFECTS:
            break
    return {"provenance": {"chip": problem["soc_name"], "workload": workload},
            "descriptors": problem["workload_descriptors"][workload],
            "baseline_design": problem["baseline"], "baseline_metric": baseline_value,
            "best_design": best["knobs"], "best_metric": best["metrics"][key],
            "top_effects": effects, "run_id": run_id, "designs_measured": len(history)}


def facts_from_run(problem, history, workload, run_id):
    """One fact per (knob, from, to) with at least one controlled pair measured on
    this workload in this run: two evaluated designs that differ in exactly one knob."""
    key = workload_metric_key(problem, workload)
    facts = []
    for effect in forecast.one_knob_effects(history, key):
        facts.append({"provenance": {"chip": problem["soc_name"], "workload": workload, "run_id": run_id},
                      "descriptors": problem["workload_descriptors"][workload],
                      "knob": effect["knob"], "from": effect["from"], "to": effect["to"],
                      "effect": effect["gain_pct"], "pairs": effect["pairs"], "bucket": bucket_of(effect["gain_pct"]),
                      "gate_distance": None,
                      "record": {"cited": 0, "improved": 0, "worsened": 0, "direction_wrong_at": []}})
    return facts


def record_run(memory, problem, history, forecast_log, reflection, run_id):
    """Mechanical write path plus the linted strategies from the reflection call."""
    for workload in problem["workload_descriptors"]:
        case = case_from_run(problem, history, workload, run_id)
        case["id"] = next_id(memory, "cases", "CASE")
        memory["cases"].append(case)
        for fact in facts_from_run(problem, history, workload, run_id):
            fact["id"] = next_id(memory, "facts", "FACT")
            memory["facts"].append(fact)
    for item in forecast_log:
        logged = dict(item)
        logged["run_id"] = run_id
        memory["forecast_log"].append(logged)
    admitted = []
    rejected = []
    for proposal in reflection:
        if not isinstance(proposal, dict) or not isinstance(proposal.get("text"), str):
            continue
        kind = proposal.get("kind")
        if kind not in ["procedure", "failure", "structure"]:
            kind = "procedure"
        reason = lint_strategy(proposal["text"], memory, problem)
        if reason is not None:
            rejected.append((proposal["text"][:90], reason))
            continue
        evidence = proposal.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        memory["strategies"].append({"id": next_id(memory, "strategies", "STRAT"), "text": proposal["text"], "kind": kind,
                                     "evidence": [{"run_id": run_id, "designs": evidence}],
                                     "record": {"cited": 0, "improved": 0, "worsened": 0, "runs_supporting": 1}})
        admitted.append(kind)
    return admitted, rejected


# ---------------------------------------------------------------- read path ----

def section_workloads(problem, history):
    lines = ["## Workloads (descriptors profiled from the traces, one row per workload)"]
    objective = problem["objective"]
    gains = {}
    if len(history) > 1:
        # Each workload's share of the suite gain so far: its log gain over the start
        # divided by the sum over workloads (the suite is a geometric mean).
        start = history[0]["metrics"]
        best = max(history, key=lambda entry: entry["metrics"][objective])["metrics"]
        total = 0.0
        for workload in problem["workload_descriptors"]:
            key = workload_metric_key(problem, workload)
            if key in start and key in best and start[key] > 0 and best[key] > 0:
                gains[workload] = math.log(best[key] / start[key])
                total += gains[workload]
        for workload in gains:
            if total > 0:
                gains[workload] = 100.0 * gains[workload] / total
            else:
                gains[workload] = 0.0
    for workload in problem["workload_descriptors"]:
        rounded = {}
        for name, value in problem["workload_descriptors"][workload].items():
            rounded[name] = round(value, 4)
        line = "- {}: {}".format(workload, json.dumps(rounded))
        if workload in gains:
            line += " | share of the suite gain so far: {:.0f}%".format(gains[workload])
        lines.append(line)
    return "\n".join(lines)


def section_distances(memory, problem, stats):
    if len(memory["cases"]) == 0:
        return ""
    pairwise = pairwise_case_distances(memory, stats)
    lines = ["## Distance to memory (standardized descriptor space; percentile among all pairwise distances between remembered cases)"]
    for workload in problem["workload_descriptors"]:
        nearest = nearest_cases(memory, problem["workload_descriptors"][workload], stats, 1)
        if len(nearest) == 0:
            continue
        nearest_distance, case = nearest[0]
        percentile = percentile_of(nearest_distance, pairwise)
        if percentile is None:
            percentile_text = "n/a (one case in memory)"
        elif percentile >= 100.0:
            percentile_text = "larger than all {} pairwise distances between remembered cases".format(len(pairwise))
        else:
            percentile_text = "{:.0f}th percentile of {} pairwise distances".format(percentile, len(pairwise))
        lines.append("- {}: nearest case {} at distance {:.2f} ({})".format(workload, case["id"], nearest_distance, percentile_text))
    return "\n".join(lines)


def case_text(case, this_distance=None):
    rounded = {}
    for name, value in case["descriptors"].items():
        if not name.startswith("max_"):
            rounded[name] = round(value, 3)
    baseline = "n/a"
    if case.get("baseline_metric") is not None:
        baseline = "{:.4f}".format(case["baseline_metric"])
    header = "{} ({} / {}, run {}, {} designs): baseline {} -> best {:.4f}".format(
        case["id"], case["provenance"]["chip"], case["provenance"]["workload"], case.get("run_id", "?"),
        case.get("designs_measured", "?"), baseline, case["best_metric"])
    if this_distance is not None:
        header += " | distance {:.2f}".format(this_distance)
    lines = [header, "  descriptors: " + json.dumps(rounded), "  best design: " + json.dumps(case["best_design"])]
    for effect in case["top_effects"]:
        lines.append("  measured: {} {} -> {}: {:+.1f}% ({} pairs)".format(
            effect["knob"], effect["from"], effect["to"], effect["effect"] + 0.0 if abs(effect["effect"]) >= 0.05 else 0.0,
            effect.get("pairs", "?")))
    return "\n".join(lines)


def section_cases(memory, problem, stats):
    if len(memory["cases"]) == 0:
        return ""
    shown = {}
    for workload in problem["workload_descriptors"]:
        for case_distance, case in nearest_cases(memory, problem["workload_descriptors"][workload], stats):
            if case["id"] not in shown or shown[case["id"]] > case_distance:
                shown[case["id"]] = case_distance
    lines = ["## Nearest remembered cases (k={} per workload, deduplicated)".format(NEAREST_CASES)]
    for case in memory["cases"]:
        if case["id"] in shown:
            lines.append(case_text(case, shown[case["id"]]))
    return "\n".join(lines)


def similar_facts(memory, problem, stats):
    """Facts measured on a workload closer to one of the current workloads than the
    median pairwise distance between remembered cases, and within the fact's own
    gate. Sorted by pairs / (1 + distance)."""
    pairwise = pairwise_case_distances(memory, stats)
    if len(pairwise) == 0:
        median = float("inf")
    else:
        median = pairwise[len(pairwise) // 2]
    scored = []
    for fact in memory["facts"]:
        best_distance = None
        for workload in problem["workload_descriptors"]:
            fact_distance = distance(problem["workload_descriptors"][workload], fact["descriptors"], stats)
            if best_distance is None or fact_distance < best_distance:
                best_distance = fact_distance
        if best_distance is None or best_distance > median:
            continue
        if fact.get("gate_distance") is not None and best_distance > fact["gate_distance"]:
            continue
        scored.append((fact["pairs"] / (1.0 + best_distance), best_distance, fact))
    scored.sort(key=lambda item: -item[0])
    return scored[:FACTS_IN_PROMPT]


def section_facts(memory, problem, stats):
    facts = similar_facts(memory, problem, stats)
    if len(facts) == 0:
        if len(memory["facts"]) == 0:
            return ""
        return ("## Remembered facts\nFacts retrieved: 0 of {} (no remembered workload within the similarity gate "
                "of the current ones)".format(len(memory["facts"])))
    lines = ["## Remembered facts (measured elsewhere; effect, pairs, where, record): {} of {} pass the similarity gate".format(
        len(facts), len(memory["facts"]))]
    for score, fact_distance, fact in facts:
        record = fact["record"]
        lines.append("- {}: {} {} -> {}: {:+.1f}% over {} pair(s) on {}/{} (distance {:.2f}) | cited {} improved {} worsened {}".format(
            fact["id"], fact["knob"], fact["from"], fact["to"], fact["effect"], fact["pairs"],
            fact["provenance"]["chip"], fact["provenance"]["workload"], fact_distance,
            record["cited"], record["improved"], record["worsened"]))
    return "\n".join(lines)


def strategy_order(strategy):
    record = strategy["record"]
    demoted = record["cited"] >= DEMOTE_AFTER_CITATIONS and record["worsened"] > record["improved"]
    kind_rank = 0
    if strategy["kind"] != "failure":
        kind_rank = 1
    return (1 if demoted else 0, kind_rank, -record["runs_supporting"], -(record["improved"] - record["worsened"]))


def section_strategies(memory):
    if len(memory["strategies"]) == 0:
        return ""
    ordered = sorted(memory["strategies"], key=strategy_order)[:STRATEGIES_IN_PROMPT]
    lines = ["## Remembered strategies (kind; runs supporting; record)"]
    for strategy in ordered:
        record = strategy["record"]
        lines.append("- {} [{}; supported by {} run(s); cited {} improved {} worsened {}]: {}".format(
            strategy["id"], strategy["kind"], record["runs_supporting"], record["cited"], record["improved"],
            record["worsened"], strategy["text"]))
    return "\n".join(lines)


def section_calibration(memory, forecast_log):
    """The agent's own forecasting record in THIS run (per run, so every agent arm sees
    the same kind of record; earlier runs' forecasts stay in memory as the artifact)."""
    settled = []
    for item in forecast_log:
        if item.get("measured") is not None:
            settled.append(item)
    if len(settled) == 0:
        return "## Your own forecasting record\nNo settled forecasts yet in this search."
    signed = 0.0
    brier = 0.0
    buckets = {}
    for item in settled:
        error = item["measured"] - item["predicted"]
        signed += error
        hit = item["measured"] >= 0.95 * item["predicted"]
        probability = (1.0 + item["confidence"]) / 2.0
        brier += (probability - (1.0 if hit else 0.0)) ** 2
        bucket = "{:.1f}".format(round(item["confidence"], 1))
        counts = buckets.setdefault(bucket, [0, 0])
        counts[1] += 1
        if hit:
            counts[0] += 1
    lines = ["## Your own forecasting record in this search ({} settled forecasts): mean signed error {:+.4f}, Brier {:.3f}".format(
        len(settled), signed / len(settled), brier / len(settled))]
    for bucket in sorted(buckets):
        hits, total = buckets[bucket]
        lines.append("- stated confidence {}: {} of {} landed within 5% of the forecast or above".format(bucket, hits, total))
    return "\n".join(lines)


def assemble_prompt(problem, history, memory, shelves, forecast_log, how_many):
    """The sections, in order, each generated from data. Empty shelves and an
    empty memory produce exactly the plain agent's prompt."""
    from loop.loop import format_table
    stats = None
    if len(memory["cases"]) > 0:
        stats = standardization(memory, problem)
    sections = []
    sections.append(analyst.section_problem(problem["search_space"], problem["objective"], problem["area_budget_kb"],
                                            problem.get("chip_text", "")))
    sections.append(section_workloads(problem, history))
    if ("facts" in shelves or "strategies" in shelves) and stats is not None:
        sections.append(section_distances(memory, problem, stats))
    if "facts" in shelves and stats is not None:
        sections.append(section_cases(memory, problem, stats))
        sections.append(section_facts(memory, problem, stats))
    if "strategies" in shelves:
        sections.append(section_strategies(memory))
    sections.append(section_calibration(memory, forecast_log))
    sections.append("## Simulation results so far on this chip and workloads (one row per design)\n" +
                    format_table(history, problem["table_metrics"]))
    with_memory = len(shelves) > 0 and (len(memory["cases"]) > 0 or len(memory["strategies"]) > 0)
    sections.append(analyst.section_pick_task(problem["objective"], how_many, with_memory))
    kept = []
    for section in sections:
        if section:
            kept.append(section)
    return "\n\n".join(kept)


# ---------------------------------------------------------------- opening move ----

def opening_move(memory, problem, stats):
    """Round 1 of a memory arm, one slot, mechanical: the START design with the
    decisive knob flipped. The decisive knob is the one with the largest mean
    absolute measured effect across the cases retrieved for the current workloads;
    it is set to the strongest alternative value those cases measured that the start
    does not already use. Tested on the start's own geometry, so the answer is about
    the knob, not about a remembered design. Empty memory: nothing."""
    if len(memory["cases"]) == 0 or stats is None:
        return []
    retrieved = {}
    for workload in problem["workload_descriptors"]:
        for case_distance, case in nearest_cases(memory, problem["workload_descriptors"][workload], stats):
            retrieved[case["id"]] = case
    effect_sum = {}
    effect_count = {}
    better_values = {}          # knob -> {value: strongest effect in its favour}
    for case in retrieved.values():
        for effect in case["top_effects"]:
            knob = effect["knob"]
            effect_sum[knob] = effect_sum.get(knob, 0.0) + abs(effect["effect"])
            effect_count[knob] = effect_count.get(knob, 0) + 1
            if effect["effect"] > 0:
                favoured = effect["to"]
            else:
                favoured = effect["from"]
            strength = abs(effect["effect"])
            if strength > better_values.setdefault(knob, {}).get(str(favoured), 0.0):
                better_values[knob][str(favoured)] = strength
    ranked_knobs = sorted(effect_sum, key=lambda knob: -(effect_sum[knob] / effect_count[knob]))
    start_design = problem["baseline"]
    for knob in ranked_knobs:
        alternatives = {}
        for value, strength in better_values[knob].items():
            if value != str(start_design[knob]):
                alternatives[value] = strength
        if len(alternatives) == 0:
            continue
        chosen_value = max(alternatives, key=alternatives.get)
        design = dict(start_design)
        for allowed in problem["search_space"][knob]:
            if str(allowed) == chosen_value:
                design[knob] = allowed
        if not problem["is_candidate"](design):
            continue
        design = typed_knobs(design, problem["search_space"])
        return [(problem["name_of"](design), design)]
    return []


def perturbation(problem, history, measured):
    """The fallback when the LLM cannot name a new design: the incumbent with one
    knob moved to an untried value (categorical: a value nobody has tried on this
    problem; ordinal: one step up or down), preferring the knob least explored so
    far. Deterministic; never a random design."""
    objective = problem["objective"]
    incumbent = history[0]
    for entry in history:
        if entry["metrics"][objective] > incumbent["metrics"][objective]:
            incumbent = entry
    tried = {}
    for entry in history:
        for knob, value in entry["knobs"].items():
            tried.setdefault(knob, set()).add(str(value))
    knobs_by_exploration = sorted(problem["search_space"], key=lambda knob: len(tried.get(knob, set())))
    for knob in knobs_by_exploration:
        values = problem["search_space"][knob]
        candidates = []
        if isinstance(values[0], str):
            for value in values:
                if str(value) not in tried.get(knob, set()):
                    candidates.append(value)
        else:
            for direction in ["up", "down"]:
                stepped = forecast.stepped_value(problem["search_space"], knob, incumbent["knobs"][knob], direction)
                if stepped is not None:
                    candidates.append(stepped)
        for value in candidates:
            design = dict(incumbent["knobs"])
            design[knob] = value
            if not problem["is_candidate"](design):
                continue
            design = typed_knobs(design, problem["search_space"])
            name = problem["name_of"](design)
            if name not in measured:
                return name, design
    return None


# ---------------------------------------------------------------- scoreboard ----

def count_citations(memory, relied_on):
    for item_id in relied_on:
        for shelf in ["facts", "strategies"]:
            for item in memory[shelf]:
                if item["id"] == item_id:
                    item["record"]["cited"] += 1


def settle_citations_by_percentile(memory, reasoning_log):
    """Once per run: an item is scored by the mean percentile (within this run's
    picks) of the picks that cited it, against the mean percentile of the picks that
    did not (the run's median when every pick cited something). Above: improved += 1;
    below: worsened += 1. "Below the incumbent" would punish every probe."""
    values = sorted(item["measured"] for item in reasoning_log)
    if len(values) < 2:
        return {}
    def percentile(value):
        below = 0
        for other in values:
            if other < value:
                below += 1
        return 100.0 * below / (len(values) - 1)
    cited_by = {}
    uncited = []
    for item in reasoning_log:
        if len(item["relies_on"]) == 0:
            uncited.append(percentile(item["measured"]))
        for item_id in item["relies_on"]:
            cited_by.setdefault(item_id, []).append(percentile(item["measured"]))
    baseline = 50.0
    if len(uncited) > 0:
        baseline = sum(uncited) / len(uncited)
    verdicts = {}
    for item_id, percentiles in cited_by.items():
        mean = sum(percentiles) / len(percentiles)
        for shelf in ["facts", "strategies"]:
            for item in memory[shelf]:
                if item["id"] == item_id:
                    if mean >= baseline:
                        item["record"]["improved"] += 1
                    else:
                        item["record"]["worsened"] += 1
        verdicts[item_id] = round(mean - baseline, 1)
    return verdicts


def settle_facts_by_pairs(memory, problem, history, stats, store, run_id):
    """A remembered fact whose (knob, from, to) has a controlled pair measured on a
    workload of THIS problem is settled on direction and bucket, per workload."""
    settled = 0
    for workload in problem["workload_descriptors"]:
        key = workload_metric_key(problem, workload)
        effects = forecast.one_knob_effects(history, key)
        here = problem["workload_descriptors"][workload]
        for fact in memory["facts"]:
            for effect in effects:
                same = effect["knob"] == fact["knob"] and effect["from"] == fact["from"] and effect["to"] == fact["to"]
                if not same:
                    continue
                fact_distance = distance(here, fact["descriptors"], stats)
                direction_ok = (effect["gain_pct"] > 0) == (fact["effect"] > 0)
                bucket_ok = bucket_of(effect["gain_pct"]) == fact["bucket"]
                event = "{} {} -> {} on {}: sign {} bucket {}".format(fact["knob"], fact["from"], fact["to"], workload,
                                                                      "held" if direction_ok else "failed",
                                                                      "held" if bucket_ok else bucket_of(effect["gain_pct"]))
                bet_id = playbook.place_bet(store, fact["id"], run_id, event, 0.5 + 0.5 * min(1.0, fact["pairs"] / 5.0),
                                            kind="sign", level="workload")
                playbook.settle_bet(store, bet_id, direction_ok)
                settled += 1
                if not direction_ok:
                    fact["record"]["direction_wrong_at"].append({"descriptors": here, "distance": fact_distance, "run_id": run_id})
                    if fact.get("gate_distance") is None or fact_distance < fact["gate_distance"]:
                        fact["gate_distance"] = fact_distance
                elif not bucket_ok:
                    fact["bucket"] = bucket_of(effect["gain_pct"])
    return settled


# ---------------------------------------------------------------- the agent ----

def run_agent(problem, rounds, per_round, store, tag, memory_path, seed=0, shelves=()):
    """One agent for every LLM arm: the plain agent (llm_direct) is `shelves=()`;
    the memory arms open the facts and/or strategies shelves. Same budget, same
    start, same seeds. Memory arms open with the mechanical pair; the plain agent
    does not. Each pick bets that the measured objective lands within 5% of its
    forecast (P = (1 + confidence) / 2); a cited fact or strategy is scored on
    whether the pick beat the incumbent."""
    from loop.loop import valid_pick
    objective = problem["objective"]
    # The plain agent reads nothing (an empty memory), but its run is still
    # recorded next to the cell's memory: every search feeds the next cell.
    memory = empty()
    if len(shelves) > 0:
        memory = load(memory_path)
    stats = None
    if len(memory["cases"]) > 0:
        stats = standardization(memory, problem)
    is_candidate = problem["is_candidate"]
    name_of = problem["name_of"]
    baseline_name = name_of(problem["baseline"])
    baseline_metrics = problem["evaluate"](problem["baseline"])
    history = [{"name": baseline_name, "knobs": problem["baseline"], "metrics": baseline_metrics,
                "reference": baseline_metrics[objective], "source": "start"}]
    measured = {baseline_name}
    knobs_of = {}
    forecast_log = []
    reasoning_log = []
    round_logs = []
    for round_number in range(1, rounds + 1):
        chosen = []
        picks = {}
        slots = {}
        if round_number == 1 and len(shelves) > 0:
            for name, knobs in opening_move(memory, problem, stats):
                if name not in measured and len(chosen) < per_round:
                    chosen.append(name)
                    knobs_of[name] = knobs
                    slots[name] = "opening move"
        proposals = []
        prompt = None
        rejected = []
        if len(chosen) < per_round:
            prompt = assemble_prompt(problem, history, memory, shelves, forecast_log, per_round - len(chosen))
            proposals = analyst.pick_from_prompt(prompt)
        # Two attempts: the second repeats the prompt with the designs the first one
        # re-proposed listed as already tested. LLM output is untrusted: a pick must be
        # a real, in-budget, unmeasured design.
        for attempt in range(2):
            for index, proposal in enumerate(proposals):
                if len(chosen) == per_round:
                    break
                if not valid_pick(proposal) or not is_candidate(proposal["knobs"]):
                    rejected.append(json.dumps(proposal.get("knobs")))
                    continue
                knobs = typed_knobs(proposal["knobs"], problem["search_space"])
                candidate_name = name_of(knobs)
                if candidate_name in measured or candidate_name in chosen:
                    rejected.append(candidate_name)
                    continue
                chosen.append(candidate_name)
                knobs_of[candidate_name] = knobs
                relied_on = proposal.get("relies_on")
                if not isinstance(relied_on, list):
                    relied_on = []
                picks[candidate_name] = {"id": "{}-PICK-r{}-{}-{}".format(tag, round_number, attempt + 1, index + 1),
                                         "predicted": float(proposal["predicted"]),
                                         "confidence": float(proposal["confidence"]),
                                         "relies_on": [str(item) for item in relied_on],
                                         "reasoning": str(proposal.get("reasoning", ""))}
            if len(chosen) == per_round or attempt == 1 or len(rejected) == 0:
                break
            retry_prompt = prompt + "\n\n## Already tested or not allowed (do not propose these again)\n" + "\n".join(rejected)
            proposals = analyst.pick_from_prompt(retry_prompt)
        perturbed = 0
        while len(chosen) < per_round:
            fallback = perturbation(problem, history, measured | set(chosen))
            if fallback is None:
                break
            name, knobs = fallback
            chosen.append(name)
            knobs_of[name] = knobs
            slots[name] = "perturbation"
            perturbed += 1
        # The whole prompt is kept in the report, so a reader can see exactly what the agent saw.
        round_logs.append({"round": round_number, "chosen": chosen, "slots": slots, "proposals": proposals,
                           "rejected": rejected, "perturbations": perturbed, "shelves": list(shelves), "prompt": prompt})

        incumbent = max(entry["metrics"][objective] for entry in history)
        bets_by_name = {}
        for name in chosen:
            bets_by_name[name] = []
            if name in picks:
                pick = picks[name]
                event = "{} >= {:.4f}".format(objective, 0.95 * pick["predicted"])
                bets_by_name[name].append(playbook.place_bet(store, pick["id"], name, event,
                                                             (1.0 + pick["confidence"]) / 2.0, kind="forecast"))
        knobs_list = []
        for name in chosen:
            knobs_list.append(knobs_of[name])
        metrics_list = problem["evaluate_many"](knobs_list)
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            source = "own pick"
            if name in slots:
                source = slots[name]
            history.append({"name": name, "knobs": knobs, "metrics": metrics, "reference": baseline_metrics[objective],
                            "source": source})
            measured.add(name)
            value = metrics[objective]
            for bet_id in bets_by_name[name]:
                for bet in store["bets"]:
                    if bet["id"] == bet_id:
                        metric_name, operator, threshold = bet["event"].split()
                        playbook.settle_bet(store, bet_id, value >= float(threshold))
            label = "pick"
            if name in slots:
                label = slots[name]
            elif name not in picks:
                label = "random fill"
            if name in picks:
                pick = picks[name]
                forecast_log.append({"design": name, "predicted": pick["predicted"], "confidence": pick["confidence"],
                                     "measured": value})
                reasoning_log.append({"round": round_number, "design": name, "reasoning": pick["reasoning"],
                                      "relies_on": pick["relies_on"], "measured": value})
                count_citations(memory, pick["relies_on"])
            print("[{}] round {} | {} | {}={:.4f} | {}".format(tag, round_number, name, objective, value, label), flush=True)

    fact_bets = 0
    if stats is not None:
        fact_bets = settle_facts_by_pairs(memory, problem, history, stats, store, tag)
    verdicts = settle_citations_by_percentile(memory, reasoning_log)
    if len(verdicts) > 0:
        print("[{}] citation scoreboard (mean percentile of citing picks minus the others): {}".format(tag, verdicts), flush=True)
    if memory_path is not None:
        reflect_and_write(memory, problem, history, forecast_log, reasoning_log, tag, memory_path)
    return {"history": history, "rounds": round_logs, "fact_bets": fact_bets}


def reflect_and_write(memory, problem, history, forecast_log, reasoning_log, run_id, memory_path):
    """After the run: one reflection call, then the mechanical write path, into a
    file next to the memory (seeds stay independent until `consolidate`)."""
    from loop.loop import format_table
    reflection = analyst.reflect_on_run(format_table(history, problem["table_metrics"]),
                                        section_workloads(problem, history), json.dumps(reasoning_log, indent=0)[:12000])
    admitted, rejected = record_run(memory, problem, history, forecast_log, reflection, run_id)
    run_path = memory_path.replace(".json", "_run-{}.json".format(run_id.replace("/", "_")))
    save(memory, run_path)
    print("[{}] memory written -> {} | strategies admitted {} rejected {}".format(
        run_id, run_path, admitted, [reason for text, reason in rejected]), flush=True)


def consolidate(memory_path, run_paths, output_path, problem):
    """Merge the runs' memories for the next cell: union of cases, facts and
    forecast logs; strategies grouped by the LLM (which ids say the same thing, the
    more general wording), records summed and evidence unioned by the code, every
    result linted. A strategy the LLM leaves out of every group is kept as it is."""
    memory = load(memory_path)
    base_counts = {shelf: len(memory[shelf]) for shelf in ["cases", "facts", "strategies", "forecast_log"]}
    pool = list(memory["strategies"])
    for run_path in run_paths:
        run_memory = load(run_path)
        for shelf in ["cases", "facts", "forecast_log"]:
            for item in run_memory[shelf][base_counts[shelf]:]:
                if shelf != "forecast_log":
                    item["id"] = next_id(memory, shelf, {"cases": "CASE", "facts": "FACT"}[shelf])
                memory[shelf].append(item)
        for strategy in run_memory["strategies"][base_counts["strategies"]:]:
            copied = dict(strategy)
            copied["id"] = "{}-{:04d}".format("STRAT", len(pool) + 1)
            pool.append(copied)
    by_id = {}
    compact = []
    for strategy in pool:
        by_id[strategy["id"]] = strategy
        compact.append({"id": strategy["id"], "kind": strategy["kind"], "text": strategy["text"]})
    groups = analyst.merge_strategies(json.dumps(compact, indent=0))
    kept = []
    covered = set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("text"), str):
            continue
        members = []
        for member_id in group.get("merged_ids") or []:
            if str(member_id) in by_id and str(member_id) not in covered:
                members.append(by_id[str(member_id)])
                covered.add(str(member_id))
        if len(members) == 0:
            continue
        reason = lint_strategy(group["text"], memory, problem)
        if reason is not None:
            # The wording failed the linter: keep the members as they were.
            print("[memory] consolidation rejected a wording ({}): {}".format(reason, group["text"][:90]), flush=True)
            for member in members:
                kept.append(member)
            continue
        kind = group.get("kind")
        if kind not in ["procedure", "failure", "structure"]:
            kind = members[0]["kind"]
        record = {"cited": 0, "improved": 0, "worsened": 0, "runs_supporting": 0}
        evidence = []
        for member in members:
            for key in record:
                record[key] += int(member["record"].get(key, 0))
            evidence = evidence + member.get("evidence", [])
        kept.append({"text": group["text"], "kind": kind, "evidence": evidence, "record": record})
    for strategy in pool:
        if strategy["id"] not in covered:
            kept.append(strategy)
    memory["strategies"] = []
    for strategy in kept:
        entry = dict(strategy)
        entry["id"] = "STRAT-{:04d}".format(len(memory["strategies"]) + 1)
        memory["strategies"].append(entry)
    save(memory, output_path)
    print("[memory] consolidated: {} cases, {} facts, {} strategies (from {}) -> {}".format(
        len(memory["cases"]), len(memory["facts"]), len(memory["strategies"]), len(pool), output_path), flush=True)
    return memory


# ---------------------------------------------------------------- build from tables ----

def measured_rows(problem):
    """Every design measured on a single-workload problem, from its result table."""
    history = []
    table = problem["holder"].sweep_table
    for name in table:
        if table[name]["metrics"] is None or "soc" in table[name]["knobs"]:
            continue
        history.append({"name": name, "knobs": table[name]["knobs"], "metrics": enrich_metrics(dict(table[name]["metrics"]))})
    return history


def build(soc_names, trace_paths, memory_path):
    """The initial memory of a cell, from result tables alone: one case and its
    facts per (chip, workload) measured there (mechanical), then strategies the LLM
    writes from those cases (no search trajectory exists yet), linted. Nothing is
    typed in by us."""
    from loop import champsim_problem
    memory = empty()
    last_problem = None
    for soc_name in soc_names:
        for trace_path in trace_paths:
            problem = champsim_problem.make_problem(soc_name, trace_path, allow_simulation=False)
            history = measured_rows(problem)
            if len(history) < 2:
                print("[memory] {} has no measured rows; skipped".format(problem["name"]), flush=True)
                continue
            workload = list(problem["workload_descriptors"].keys())[0]
            case = case_from_run(problem, history, workload, "tables")
            case["id"] = next_id(memory, "cases", "CASE")
            memory["cases"].append(case)
            for fact in facts_from_run(problem, history, workload, "tables"):
                fact["id"] = next_id(memory, "facts", "FACT")
                memory["facts"].append(fact)
            last_problem = problem
    if last_problem is None:
        save(memory, memory_path)
        return memory
    cases_text = "\n".join(case_text(case) for case in memory["cases"])
    admitted = 0
    for proposal in analyst.write_strategies(cases_text):
        if not isinstance(proposal, dict) or not isinstance(proposal.get("text"), str):
            continue
        reason = lint_strategy(proposal["text"], memory, last_problem)
        if reason is not None:
            print("[memory] build rejected a strategy ({}): {}".format(reason, proposal["text"][:90]), flush=True)
            continue
        kind = proposal.get("kind")
        if kind not in ["procedure", "failure", "structure"]:
            kind = "procedure"
        evidence = proposal.get("evidence")
        if not isinstance(evidence, list):
            evidence = []
        memory["strategies"].append({"id": next_id(memory, "strategies", "STRAT"), "text": proposal["text"], "kind": kind,
                                     "evidence": [{"run_id": "tables", "designs": evidence}],
                                     "record": {"cited": 0, "improved": 0, "worsened": 0, "runs_supporting": 1}})
        admitted += 1
    save(memory, memory_path)
    print("[memory] built from tables: {} cases, {} facts, {} strategies -> {}".format(
        len(memory["cases"]), len(memory["facts"]), admitted, memory_path), flush=True)
    return memory


# ---------------------------------------------------------------- import earlier searches ----

def history_from_report(run, problem):
    """A finished run's trajectory with per-workload metrics, rebuilt from the
    report's design names and the result tables (the report keeps only the suite
    objective per design)."""
    holders = problem["holders"]
    short_names = []
    for holder in holders:
        short_names.append(trace_short_name(holder.trace_path))
    history = []
    for row in run["history"]:
        name = row["name"]
        per_trace = []
        complete = True
        for holder in holders:
            entry = holder.sweep_table.get(name)
            if entry is None or entry["metrics"] is None:
                complete = False
                break
            per_trace.append(enrich_metrics(dict(entry["metrics"])))
        if not complete:
            continue
        knobs = holders[0].sweep_table[name]["knobs"]
        history.append({"name": name, "knobs": knobs, "metrics": aggregate_suite(per_trace, short_names)})
    return history


def reasoning_from_report(run, history, problem):
    """The agent's reasoning per pick, matched to the designs it chose."""
    objective = problem["objective"]
    measured = {}
    for entry in history:
        measured[entry["name"]] = entry["metrics"][objective]
    log = []
    forecasts = []
    for round_log in run["rounds"]:
        proposals = round_log.get("proposals") or []
        for proposal in proposals:
            if not isinstance(proposal, dict) or not isinstance(proposal.get("knobs"), dict):
                continue
            for name in round_log["chosen"]:
                if name in measured and forecast.same_knobs(problem["candidates"].get(name, {}), proposal["knobs"]):
                    log.append({"round": round_log["round"], "design": name, "reasoning": str(proposal.get("reasoning", "")),
                                "relies_on": [], "measured": measured[name]})
                    try:
                        forecasts.append({"design": name, "predicted": float(proposal["predicted"]),
                                          "confidence": float(proposal["confidence"]), "measured": measured[name]})
                    except (KeyError, TypeError, ValueError):
                        pass
    return log, forecasts


def import_report(memory, report_path, problem, arms):
    """Strategies and forecasts from the searches of an earlier report on this
    problem (one reflection call per run). Cases and facts are not re-derived here:
    the tables the runs wrote into already feed `build`."""
    from loop.loop import format_table
    with open(report_path) as report_file:
        report = json.load(report_file)
    imported = 0
    for problem_name in report["test"]:
        for arm in report["test"][problem_name]["arms"]:
            if arm not in arms:
                continue
            for seed, run in report["test"][problem_name]["arms"][arm].items():
                history = history_from_report(run, problem)
                if len(history) < 4:
                    continue
                reasoning_log, forecasts = reasoning_from_report(run, history, problem)
                run_id = "{}:{}-s{}".format(os.path.basename(report_path).replace(".json", ""), arm, seed)
                reflection = analyst.reflect_on_run(format_table(history, problem["table_metrics"]),
                                                    section_workloads(problem, history), json.dumps(reasoning_log, indent=0)[:12000])
                admitted = []
                rejected = []
                for proposal in reflection:
                    if not isinstance(proposal, dict) or not isinstance(proposal.get("text"), str):
                        continue
                    reason = lint_strategy(proposal["text"], memory, problem)
                    if reason is not None:
                        rejected.append(reason)
                        continue
                    kind = proposal.get("kind")
                    if kind not in ["procedure", "failure", "structure"]:
                        kind = "procedure"
                    evidence = proposal.get("evidence")
                    if not isinstance(evidence, list):
                        evidence = []
                    memory["strategies"].append({"id": next_id(memory, "strategies", "STRAT"), "text": proposal["text"], "kind": kind,
                                                 "evidence": [{"run_id": run_id, "designs": evidence}],
                                                 "record": {"cited": 0, "improved": 0, "worsened": 0, "runs_supporting": 1}})
                    admitted.append(kind)
                for item in forecasts:
                    logged = dict(item)
                    logged["run_id"] = run_id
                    memory["forecast_log"].append(logged)
                imported += 1
                print("[memory] imported {} ({} designs): strategies {} rejected {}".format(
                    run_id, len(history), admitted, rejected), flush=True)
    return imported


# ---------------------------------------------------------------- acceptance checks ----

def check(problem):
    """Offline acceptance checks: empty memory reduces to the plain agent's prompt;
    the linter rejects names and numbers; the distance section prints a percentile
    for every workload once memory holds cases."""
    from loop.loop import format_table
    history = [{"name": "start", "knobs": problem["baseline"], "metrics": {problem["objective"]: 1.0}, "reference": 1.0}]
    for workload in problem["workload_descriptors"]:
        history[0]["metrics"][workload_metric_key(problem, workload)] = 1.0
    for metric in problem["table_metrics"]:
        history[0]["metrics"].setdefault(metric, 1.0)
    plain = assemble_prompt(problem, history, empty(), (), [], 2)
    both = assemble_prompt(problem, history, empty(), ("facts", "strategies"), [], 2)
    print("check 1 (empty memory, memory prompt == llm_direct prompt):", "PASS" if plain == both else "FAIL")
    memory = empty()
    workload = list(problem["workload_descriptors"].keys())[0]
    bad_name = lint_strategy("Always try the other prefetcher on {} first".format(workload), memory, problem)
    bad_number = lint_strategy("Set llc_sets to 4096 before anything else", memory, problem)
    good = lint_strategy("Test the two strongest l2_prefetcher candidates against each other in the first round; the winner is workload-specific.", memory, problem)
    print("check 2 (linter): name ->", bad_name, "| number ->", bad_number, "| clean ->", good,
          "|", "PASS" if bad_name and bad_number and good is None else "FAIL")
    fact = facts_from_run(problem, history + [dict(history[0], name="other",
                                                    knobs=dict(problem["baseline"], l2_prefetcher="spp_dev"))],
                          workload, "check")
    named = len(fact) > 0 and fact[0]["provenance"]["workload"] == workload
    print("check 3 (a fact carries the workload name in its provenance and is admitted):", "PASS" if named else "FAIL")
    print("check 4 (llm_direct opens with no mechanical move):", "PASS" if opening_move(memory, problem, None) == [] else "FAIL")
    print("check 6 (a proposal outside the space or budget is refused):",
          "PASS" if not problem["is_candidate"](dict(problem["baseline"], l2_prefetcher="magic")) and
          not problem["is_candidate"](dict(problem["baseline"], l2_sets=2048, l2_ways=16, llc_sets=8192, llc_ways=16)) else "FAIL")
    for index, workload in enumerate(problem["workload_descriptors"]):
        memory["cases"].append({"id": "CASE-{:04d}".format(index + 1), "provenance": {"chip": "x", "workload": "w{}".format(index)},
                                "descriptors": problem["workload_descriptors"][workload], "baseline_design": problem["baseline"],
                                "baseline_metric": 1.0, "best_design": problem["baseline"], "best_metric": 1.1,
                                "top_effects": [], "run_id": "check", "designs_measured": 1})
    stats = standardization(memory, problem)
    text = section_distances(memory, problem, stats)
    percentiles = text.count("th percentile")
    print("check 5 (a percentile per workload):", "PASS" if percentiles == len(problem["workload_descriptors"]) else "FAIL", "\n" + text)


if __name__ == "__main__":
    import glob
    import sys
    from loop import champsim_problem, run
    if sys.argv[1] == "check":
        problem = champsim_problem.make_suite_problem("C_server", run.GAP_SET_1, allow_simulation=False)
        check(problem)
    elif sys.argv[1] == "build":
        build(sys.argv[3].split(","), sys.argv[4:], sys.argv[2])
    elif sys.argv[1] == "import":
        # python -m loop.memory import <memory.json> <cell of the report's problem> <report.json> <arm,arm>
        cell = run.CELLS[sys.argv[3]]
        problem = champsim_problem.make_suite_problem(cell["test_soc"], cell["test_traces"], allow_simulation=False)
        memory = load(sys.argv[2])
        import_report(memory, sys.argv[4], problem, sys.argv[5].split(","))
        save(memory, sys.argv[2])
    elif sys.argv[1] == "consolidate":
        base_path = sys.argv[2]
        output_path = sys.argv[3]
        cell = run.CELLS[sys.argv[4]]
        run_paths = sorted(glob.glob(base_path.replace(".json", "_run-*.json")))
        problem = champsim_problem.make_suite_problem(cell["test_soc"], cell["test_traces"], allow_simulation=False)
        print("consolidating {} run memories into {}".format(len(run_paths), output_path), flush=True)
        consolidate(base_path, run_paths, output_path, problem)
