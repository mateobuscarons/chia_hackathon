"""The memory: what the agent carries from chip to chip and class to class,
written for an LLM reader, not for a GP.

Three shelves and a scoreboard, one JSON file that grows and never deletes:
  cases    one card per (chip, workload): its descriptors, the start design, the
           best design found, the five largest single-knob effects measured.
           Retrieved by descriptor distance ("this looks like mcf; mcf liked X").
  cards    mechanism cards: when (clauses in descriptor words), do (knob and
           value or direction), because (the mechanism), size (a bucket: small
           < 3%, medium 3-10%, large > 10%), record (held on / failed on).
           Interactions are cards too. The mechanism is what lets the LLM judge
           whether a card applies to a class it never saw.
  recipe   an ordered decision list that turns a chip's budget and a workload's
           descriptors into a starting design.
  ledger   every bet a card placed, per workload, on direction and bucket,
           Brier-scored; lose on bucket -> re-bucket, lose on direction ->
           sharpen the condition.

A run reads the memory as it was when the cell started and writes its own
additions to a file next to it (seeds stay independent); `consolidate` merges
those files into the memory for the next cell.
"""

import json
import math
import os
import random

from loop import analyst, champsim_problem, forecast, playbook
from loop.loop import check_event, claim_events, event_text, format_table, tightened_condition, valid_pick

BUCKETS = {"small": (0.0, 3.0), "medium": (3.0, 10.0), "large": (10.0, 1000.0)}
NEAREST_CASES = 3
CARDS_IN_PROMPT = 12
TOP_EFFECTS = 5
RESCOPE_AFTER_LOSSES = 3
# Descriptors that place a workload for retrieval; log-scaled ones vary over decades.
DISTANCE_LOG = ["footprint_kb", "mem_accesses_per_kinstr", "movable_l2_mpki", "movable_llc_mpki"]
DISTANCE_LINEAR = ["stride_regular_fraction", "reuse_local_fraction", "write_fraction"]


def empty():
    return {"cases": [], "cards": [], "recipe": [], "trajectories": [], "ledger": []}


def load(path):
    if not os.path.exists(path):
        return empty()
    with open(path) as memory_file:
        return json.load(memory_file)


def save(memory, path):
    with open(path, "w") as memory_file:
        json.dump(memory, memory_file, indent=1)


# ---------------------------------------------------------------- cases ----

def bucket_of(gain_pct):
    magnitude = abs(gain_pct)
    for name in ["small", "medium", "large"]:
        low, high = BUCKETS[name]
        if low <= magnitude < high:
            return name
    return "large"


def case_card(problem, evidence, workload):
    """What happened on one (chip, workload): read from every design measured there."""
    key = workload + ":" + problem["objective"]
    if "holders" not in problem:
        key = problem["objective"]          # a single-workload problem keeps the plain name
    descriptors = problem["workload_descriptors"][workload]
    start_value = None
    best = None
    for entry in evidence:
        if forecast.same_knobs(entry["knobs"], problem["baseline"]):
            start_value = entry["metrics"][key]
        if best is None or entry["metrics"][key] > best["metrics"][key]:
            best = entry
    effects = forecast.one_knob_effects(evidence, key)[:TOP_EFFECTS]
    return {"chip": problem["soc_name"], "workload": workload, "descriptors": descriptors,
            "start_design": problem["baseline"], "start_value": start_value,
            "best_design": best["knobs"], "best_value": best["metrics"][key],
            "top_effects": effects, "designs_measured": len(evidence)}


def case_text(case):
    lines = ["case {} / {} ({} designs measured): start {:.4f} -> best {:.4f}".format(
        case["chip"], case["workload"], case["designs_measured"],
        case["start_value"] if case["start_value"] is not None else float("nan"), case["best_value"])]
    rounded = {}
    for key in DISTANCE_LOG + DISTANCE_LINEAR:
        if key in case["descriptors"]:
            rounded[key] = round(case["descriptors"][key], 3)
    lines.append("  descriptors: " + json.dumps(rounded))
    lines.append("  best design: " + json.dumps(case["best_design"]))
    for effect in case["top_effects"]:
        lines.append("  effect: {} {} -> {}: {:+.1f}% ({} pairs)".format(
            effect["knob"], effect["from"], effect["to"], effect["gain_pct"], effect["pairs"]))
    return "\n".join(lines)


def descriptor_vector(descriptors, scales):
    vector = []
    for key in DISTANCE_LOG:
        vector.append(math.log(max(descriptors.get(key, 0.0), 0.0) + 1.0) / scales[key])
    for key in DISTANCE_LINEAR:
        vector.append(descriptors.get(key, 0.0) / scales[key])
    return vector


def distance_scales(memory):
    """One scale per descriptor: the spread over the cases, so no descriptor dominates."""
    scales = {}
    for key in DISTANCE_LOG + DISTANCE_LINEAR:
        values = []
        for case in memory["cases"]:
            value = case["descriptors"].get(key, 0.0)
            if key in DISTANCE_LOG:
                value = math.log(max(value, 0.0) + 1.0)
            values.append(value)
        spread = 1.0
        if len(values) > 1:
            spread = max(values) - min(values)
        if spread <= 0:
            spread = 1.0
        scales[key] = spread
    return scales


def nearest_cases(memory, descriptors, how_many=NEAREST_CASES):
    scales = distance_scales(memory)
    here = descriptor_vector(descriptors, scales)
    scored = []
    for case in memory["cases"]:
        there = descriptor_vector(case["descriptors"], scales)
        distance = 0.0
        for index in range(len(here)):
            distance += (here[index] - there[index]) ** 2
        scored.append((math.sqrt(distance), case))
    scored.sort(key=lambda pair: pair[0])
    nearest = []
    for distance, case in scored[:how_many]:
        nearest.append(case)
    return nearest


# ---------------------------------------------------------------- cards ----

def valid_card(card, search_space, condition_metrics):
    """LLM output is untrusted: a card may only name real metrics, knobs and values."""
    try:
        for clause in card["when"]:
            if clause["metric"] not in condition_metrics or clause["op"] not in [">=", ">", "<", "<="]:
                return False
            float(clause["value"])
        knob = card["do"]["knob"]
        values = search_space[knob]
        value = card["do"]["value"]
        if str(value) in ["up", "down"]:
            for allowed in values:
                if not isinstance(allowed, (int, float)) or isinstance(allowed, bool):
                    return False
        else:
            allowed_strings = []
            for allowed in values:
                allowed_strings.append(str(allowed))
            if str(value) not in allowed_strings:
                return False
        if card["size"] not in BUCKETS:
            return False
        return isinstance(card["because"], str) and isinstance(card["when_text"], str)
    except (KeyError, TypeError, ValueError):
        return False


def add_card(memory, card):
    clean = {"id": "CARD-{:03d}".format(len(memory["cards"]) + 1),
             "when": [], "when_text": card["when_text"], "do": dict(card["do"]),
             "because": card["because"], "size": card["size"],
             "record": {"held_on": [], "failed_on": [], "bucket_missed_on": []},
             "origin": []}
    for clause in card["when"]:
        clean["when"].append({"metric": clause["metric"], "op": clause["op"], "value": float(clause["value"])})
    memory["cards"].append(clean)
    return clean


def card_score(card):
    return len(card["record"]["held_on"]) - len(card["record"]["failed_on"])


def applicable_cards(memory, descriptors):
    """Cards whose every clause holds on these descriptors, best record first."""
    applicable = []
    for card in memory["cards"]:
        holds = True
        for clause in card["when"]:
            if not forecast.clause_holds(clause, descriptors):
                holds = False
        if holds:
            applicable.append(card)
    applicable.sort(key=card_score, reverse=True)
    return applicable


def card_text(card):
    record = card["record"]
    return "{} | when {} | do {} = {} | because {} | size {} | held on {} | failed on {}".format(
        card["id"], card["when_text"], card["do"]["knob"], card["do"]["value"], card["because"], card["size"],
        ", ".join(record["held_on"][:6]) or "-", ", ".join(record["failed_on"][:6]) or "-")


def card_claim(card):
    """A card's 'do' read as a claim the loop's sibling logic understands."""
    low, high = BUCKETS[card["size"]]
    typical = (low + min(high, 30.0)) / 2.0
    return {"knob": card["do"]["knob"], "value": card["do"]["value"], "gain_pct": typical}


# ---------------------------------------------------------------- build ----

def build_initial(soc_names, traces, path):
    """The first memory, from the training chips' tables: one case per (chip,
    workload), mechanism cards written by the LLM over the cases, and the recipe."""
    from loop.experiment import measured_designs, problem_for
    memory = empty()
    problems = []
    for soc_name in soc_names:
        problem = problem_for(soc_name, traces, "C")
        evidence = measured_designs(problem)
        problems.append(problem)
        for workload in problem["workload_descriptors"]:
            memory["cases"].append(case_card(problem, evidence, workload))
    cases_text = "\n".join(case_text(case) for case in memory["cases"])
    search_space = problems[0]["search_space"]
    condition_metrics = problems[0]["condition_metrics"]
    proposals = analyst.write_cards(search_space, condition_metrics, cases_text)
    for proposal in proposals:
        if valid_card(proposal, search_space, condition_metrics):
            add_card(memory, proposal)
        else:
            print("[memory] dropped malformed card:", json.dumps(proposal)[:160], flush=True)
    memory["recipe"] = analyst.write_recipe(search_space, cases_text, "\n".join(card_text(card) for card in memory["cards"]))
    save(memory, path)
    print("[memory] {} cases, {} cards, recipe of {} steps -> {}".format(
        len(memory["cases"]), len(memory["cards"]), len(memory["recipe"]), path), flush=True)
    return memory


# ---------------------------------------------------------------- the agent ----

def recipe_design(memory, problem):
    """The recipe applied to this chip and workload by the LLM: a design in the
    candidate set, or None when its answer is not one."""
    answer = analyst.apply_recipe(problem["search_space"], memory["recipe"], problem["area_budget_kb"],
                                  json.dumps(problem["descriptors"], indent=1), problem["baseline"])
    if not isinstance(answer, dict):
        return None
    for name in problem["candidates"]:
        if forecast.same_knobs(problem["candidates"][name], answer):
            return name
    return None


def run_memory_agent(problem, rounds, per_round, store, tag, memory_path, seed=0):
    """The agent with the memory. Round 1: the recipe's design plus one pick. Every
    round: the results table, the nearest cases and the applicable cards go into
    the prompt; the agent picks and forecasts; every applicable card whose 'do' the
    design carries bets on direction and size bucket, per workload. After the run
    the agent writes new cards and the case, into a file next to the memory."""
    objective = problem["objective"]
    memory = load(memory_path)
    candidates = dict(problem["candidates"])
    baseline_name = None
    for name in candidates:
        if forecast.same_knobs(candidates[name], problem["baseline"]):
            baseline_name = name
    baseline_metrics = problem["evaluate"](problem["baseline"])
    history = [{"name": baseline_name, "knobs": problem["baseline"], "metrics": baseline_metrics,
                "reference": baseline_metrics[objective]}]
    del candidates[baseline_name]
    descriptors = problem["descriptors"]
    cases = nearest_cases(memory, descriptors)
    cases_text = "\n".join(case_text(case) for case in cases)
    filler = random.Random(seed)
    round_logs = []
    for round_number in range(1, rounds + 1):
        cards = applicable_cards(memory, descriptors)[:CARDS_IN_PROMPT]
        cards_text = "\n".join(card_text(card) for card in cards)
        chosen = []
        forecasts_of = {}
        slots = {}
        if round_number == 1:
            recipe_name = recipe_design(memory, problem)
            if recipe_name is not None and recipe_name in candidates:
                chosen.append(recipe_name)
                slots[recipe_name] = "recipe design"
        proposals = analyst.pick_with_memory(problem["search_space"], objective,
                                             format_table(history, problem["table_metrics"]),
                                             json.dumps(descriptors, indent=1), cases_text, cards_text,
                                             "\n".join(memory["recipe"]), per_round - len(chosen), problem["area_budget_kb"])
        for index, proposal in enumerate(proposals):
            if len(chosen) == per_round:
                break
            if not valid_pick(proposal):
                continue
            for candidate_name in candidates:
                if candidate_name in chosen:
                    continue
                if forecast.same_knobs(candidates[candidate_name], proposal["knobs"]):
                    chosen.append(candidate_name)
                    forecasts_of[candidate_name] = {"id": "{}-MEM-r{}-{}".format(tag, round_number, index + 1),
                                                    "predicted": float(proposal["predicted"]),
                                                    "confidence": float(proposal["confidence"])}
                    break
        filled = 0
        names = list(candidates.keys())
        while len(chosen) < per_round:
            name = names[filler.randrange(len(names))]
            if name not in chosen:
                chosen.append(name)
                filled += 1
        round_logs.append({"round": round_number, "chosen": chosen, "slots": slots, "proposals": proposals,
                           "random_fill": filled, "cards_shown": [card["id"] for card in cards]})

        # Bets: the agent on its own forecast; every applicable card on direction and bucket.
        bets_by_name = {}
        for name in chosen:
            bets_by_name[name] = []
            if name in forecasts_of:
                pick = forecasts_of[name]
                event = "{} >= {:.4f}".format(objective, 0.95 * pick["predicted"])
                bets_by_name[name].append(playbook.place_bet(store, pick["id"], name, event,
                                                             (1.0 + pick["confidence"]) / 2.0, kind="forecast"))
            for card in cards:
                claim = card_claim(card)
                sibling = forecast.claim_sibling(claim, candidates[name], problem["baseline"], problem["search_space"])
                if sibling is None:
                    continue
                sibling_metrics = forecast.find_measured(history, sibling)
                if sibling_metrics is None:
                    continue          # cards bet only as controlled comparisons
                credibility = card_credibility(card)
                for key in sibling_metrics:
                    if key != objective and not key.endswith(":" + objective):
                        continue
                    level = "suite"
                    if key != objective:
                        level = "workload"
                    sibling_value = sibling_metrics[key]
                    direction = 1.0
                    low, high = BUCKETS[card["size"]]
                    sign_event = "{} >= {:.4f}".format(key, sibling_value * (1.0 + 0.001))
                    bets_by_name[name].append(playbook.place_bet(store, card["id"], name, sign_event, credibility,
                                                                 kind="sign", level=level))
                    bucket_event = "{} in {:.4f},{:.4f}".format(key, sibling_value * (1.0 + low / 100.0),
                                                                sibling_value * (1.0 + high / 100.0))
                    bets_by_name[name].append(playbook.place_bet(store, card["id"], name, bucket_event,
                                                                 credibility * 0.8, kind="bucket", level=level))
        knobs_list = []
        for name in chosen:
            knobs_list.append(candidates[name])
        metrics_list = problem["evaluate_many"](knobs_list)
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            history.append({"name": name, "knobs": knobs, "metrics": metrics, "reference": baseline_metrics[objective]})
            del candidates[name]
            for bet_id in bets_by_name[name]:
                bet = None
                for candidate in store["bets"]:
                    if candidate["id"] == bet_id:
                        bet = candidate
                happened = settle_memory_event(bet["event"], metrics)
                playbook.settle_bet(store, bet_id, happened)
                for card in cards:
                    if card["id"] != bet["forecaster"]:
                        continue
                    update_record(card, bet, happened, problem["name"], metrics, descriptors)
            label = "memory pick"
            if name in slots:
                label = slots[name]
            elif name not in forecasts_of:
                label = "random fill"
            print("[{}] round {} | {} | {}={:.4f} | {} | {} bets".format(
                tag, round_number, name, objective, metrics[objective], label, len(bets_by_name[name])), flush=True)

    write_back(memory, problem, history, store, tag, memory_path)
    return {"history": history, "rounds": round_logs}


def card_credibility(card):
    """Like a rule: a prior of 0.7 pulled toward the card's own direction record."""
    held = len(card["record"]["held_on"])
    failed = len(card["record"]["failed_on"])
    return (0.7 * 2 + held) / (2 + held + failed)


def settle_memory_event(event, metrics):
    """'metric in low,high' is a bucket; everything else is a plain comparison."""
    if " in " in event:
        metric_name, bounds = event.split(" in ")
        low, high = bounds.split(",")
        return float(low) <= metrics[metric_name] <= float(high)
    return check_event(event, metrics)


def update_record(card, bet, happened, problem_name, metrics, descriptors):
    """Direction held or failed -> the record; bucket missed -> re-bucket to the
    measured one; three direction failures here -> sharpen the first clause."""
    workload = bet["event"].split(" ")[0]
    where = "{}:{}".format(problem_name, workload)
    if bet["kind"] == "sign":
        if happened:
            card["record"]["held_on"].append(where)
        else:
            card["record"]["failed_on"].append(where)
            failures_here = 0
            for entry in card["record"]["failed_on"]:
                if entry.startswith(problem_name):
                    failures_here += 1
            if failures_here == RESCOPE_AFTER_LOSSES and len(card["when"]) > 0:
                old = list(card["when"])
                card["when"][0] = tightened_condition(card["when"][0], descriptors)
                card["origin"].append({"sharpened_from": old, "on": problem_name})
    elif bet["kind"] == "bucket" and not happened:
        card["record"]["bucket_missed_on"].append(where)
        metric_name = bet["event"].split(" in ")[0]
        low_text, high_text = bet["event"].split(" in ")[1].split(",")
        # The sibling value is the bucket's lower bound divided by (1 + low/100).
        low_pct, high_pct = BUCKETS[card["size"]]
        sibling_value = float(low_text) / (1.0 + low_pct / 100.0)
        measured_gain = 100.0 * (metrics[metric_name] / sibling_value - 1.0)
        new_bucket = bucket_of(measured_gain)
        if new_bucket != card["size"]:
            card["origin"].append({"rebucketed_from": card["size"], "to": new_bucket, "on": where})
            card["size"] = new_bucket


def write_back(memory, problem, history, store, tag, memory_path):
    """After the run: the agent writes new cards and updated records, the case for
    this problem and the trajectory, into a file next to the memory (seeds stay
    independent; `consolidate` merges them for the next cell)."""
    objective = problem["objective"]
    for workload in problem["workload_descriptors"]:
        memory["cases"].append(case_card(problem, history, workload))
    trajectory = []
    for entry in history:
        trajectory.append({"name": entry["name"], objective: entry["metrics"][objective]})
    memory["trajectories"].append({"problem": problem["name"], "tag": tag, "trajectory": trajectory})
    settled = []
    for bet in store["bets"]:
        if bet["outcome"] is not None and bet["forecaster"].startswith("CARD"):
            settled.append("{} {} {} p={:.2f}: {}".format(bet["forecaster"], bet["kind"], bet["event"],
                                                        bet["probability"], "happened" if bet["outcome"] else "did not happen"))
    proposals = analyst.revise_cards(problem["search_space"], problem["condition_metrics"],
                                     format_table(history, problem["table_metrics"]),
                                     json.dumps(problem["descriptors"], indent=1),
                                     "\n".join(card_text(card) for card in memory["cards"]),
                                     "\n".join(settled[-80:]))
    added = 0
    for proposal in proposals:
        if valid_card(proposal, problem["search_space"], problem["condition_metrics"]):
            add_card(memory, proposal)
            added += 1
    memory["ledger"] = memory["ledger"] + settled
    run_path = memory_path.replace(".json", "_{}.json".format(tag.replace("/", "_")))
    save(memory, run_path)
    print("[{}] memory written: {} new cards -> {}".format(tag, added, run_path), flush=True)


def consolidate(memory_path, run_paths, output_path, search_space, condition_metrics):
    """Merge the runs' memories into one for the next cell: union of cases and
    trajectories, cards merged by the LLM (duplicates folded, records kept)."""
    memory = load(memory_path)
    all_cards = list(memory["cards"])
    for run_path in run_paths:
        run_memory = load(run_path)
        memory["cases"] = memory["cases"] + run_memory["cases"][len(memory["cases"]):]
        memory["trajectories"] = memory["trajectories"] + run_memory["trajectories"]
        memory["ledger"] = memory["ledger"] + run_memory["ledger"]
        all_cards = all_cards + run_memory["cards"][len(memory["cards"]):]
    merged = analyst.merge_cards(search_space, condition_metrics, "\n".join(card_text(card) for card in all_cards))
    memory["cards"] = []
    for proposal in merged:
        if valid_card(proposal, search_space, condition_metrics):
            card = add_card(memory, proposal)
            if isinstance(proposal.get("record"), dict):
                for field in ["held_on", "failed_on", "bucket_missed_on"]:
                    if isinstance(proposal["record"].get(field), list):
                        card["record"][field] = list(proposal["record"][field])
    memory["recipe"] = analyst.write_recipe(search_space, "\n".join(case_text(case) for case in memory["cases"]),
                                            "\n".join(card_text(card) for card in memory["cards"]),
                                            "\n".join(json.dumps(item) for item in memory["trajectories"][-10:]))
    save(memory, output_path)
    print("[memory] consolidated: {} cases, {} cards, recipe {} steps -> {}".format(
        len(memory["cases"]), len(memory["cards"]), len(memory["recipe"]), output_path), flush=True)
    return memory


if __name__ == "__main__":
    import glob
    import sys
    from loop.configs import SEARCH_SPACE
    from loop.champsim_problem import DESCRIPTOR_METRICS
    if sys.argv[1] == "consolidate":
        base_path = sys.argv[2]
        output_path = sys.argv[3]
        run_paths = sorted(glob.glob(base_path.replace(".json", "_memory-*.json")))
        print("consolidating {} run memories into {}".format(len(run_paths), output_path), flush=True)
        condition_metrics = list(DESCRIPTOR_METRICS)
        for metric in DESCRIPTOR_METRICS:
            condition_metrics.append("max_" + metric)
        consolidate(base_path, run_paths, output_path, SEARCH_SPACE, condition_metrics)
