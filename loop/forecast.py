"""Forecasters and experiment selection.

Three kinds of forecaster give a predicted objective for a candidate config:
  - the surrogate (always has an opinion),
  - playbook rules (only when their condition holds on this workload and chip,
    the candidate carries the knob value they talk about, and their record is
    still credible),
  - the analyst's reply (only for the one design it proposed on a stall).
Designs are picked by expected improvement over the surrogate; a speaking rule
tilts the surrogate's mean toward what it claims (a bounded shift that decays
as real observations arrive), and every forecaster bets on every chosen design
so the simulator can score it.

A rule is a DELTA claim: "switching knob K to value V changes the objective
by gain_pct %". Its forecast for a candidate is therefore relative to the
candidate's SIBLING (same config with K at the baseline value): the measured
sibling when we have it, the surrogate's estimate of it otherwise. Forecasting
baseline * (1 + gain) for every candidate would give the same number to
hundreds of different configs and make rule bets a coin flip.
"""

import math
import random

PRIOR_CONFIDENCE = 0.7   # a brand-new rule's confidence, before any bets
PRIOR_WEIGHT = 2         # how many imaginary bets that prior is worth
CREDIBLE_CONFIDENCE = 0.5   # below this a rule stays silent on the current problem


def clause_holds(clause, baseline_metrics):
    """clause: {"metric": "LLC_mpki", "op": ">=", "value": 20}."""
    value = baseline_metrics[clause["metric"]]
    op = clause["op"]
    threshold = float(clause["value"])
    # The analyst is asked for ">=" or "<" but sometimes writes the others.
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    raise ValueError("unsupported condition op: " + op)


def rule_clauses(rule):
    """A rule may carry one clause ("condition") or several ("conditions", ANDed)."""
    if "conditions" in rule and isinstance(rule["conditions"], list) and len(rule["conditions"]) > 0:
        return rule["conditions"]
    return [rule["condition"]]


def rule_applies(rule, baseline_metrics):
    """Every clause must hold on the untouched baseline of this chip+workload."""
    for clause in rule_clauses(rule):
        if not clause_holds(clause, baseline_metrics):
            return False
    return True


def rule_confidence(rule):
    """Prior confidence, pulled toward the rule's actual win rate over ALL its bets."""
    total = PRIOR_WEIGHT + rule["wins"] + rule["losses"]
    return (PRIOR_CONFIDENCE * PRIOR_WEIGHT + rule["wins"]) / total


def rule_is_credible(rule):
    if rule["status"] != "active":
        return False
    return rule_confidence(rule) >= CREDIBLE_CONFIDENCE


def speaking_rules(rules, baseline_metrics):
    """The rules that may forecast on this problem: active, credible, condition holds."""
    speaking = []
    for rule in rules:
        if not rule_is_credible(rule):
            continue
        if not rule_applies(rule, baseline_metrics):
            continue
        speaking.append(rule)
    return speaking


def sibling_knobs(knobs, knob, baseline_value):
    """The candidate with one knob put back to its baseline value."""
    sibling = dict(knobs)
    sibling[knob] = baseline_value
    return sibling


def find_measured(history, wanted_knobs):
    """Metrics of an already-run config with exactly these knobs, else None."""
    for entry in history:
        if same_knobs(entry["knobs"], wanted_knobs):
            return entry["metrics"]
    return None


def same_knobs(knobs_a, knobs_b):
    """Compare as strings: the analyst may return 1024 as "1024"."""
    if len(knobs_a) != len(knobs_b):
        return False
    for knob in knobs_a:
        if knob not in knobs_b or str(knobs_a[knob]) != str(knobs_b[knob]):
            return False
    return True


# ---- claims: a fixed value ("spp_dev") or a direction on an ordinal knob ("up" / "down") ----
# A direction claim says "moving this knob ONE step up (down) from where it is
# changes the objective by gain_pct". Directions transfer across chips whose
# budgets allow different absolute sizes; fixed sizes do not.


def is_direction(claim):
    return str(claim["value"]) in ["up", "down"]


def ordered_values(search_space, knob):
    values = list(search_space[knob])
    numeric = True
    for value in values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            numeric = False
    if numeric:
        values.sort()
    return values


def stepped_value(search_space, knob, value, direction):
    """The next allowed value one step up/down from `value`, or None at the edge."""
    values = ordered_values(search_space, knob)
    position = None
    for index, allowed in enumerate(values):
        if str(allowed) == str(value):
            position = index
    if position is None:
        return None
    if direction == "up":
        if position + 1 < len(values):
            return values[position + 1]
        return None
    if position - 1 >= 0:
        return values[position - 1]
    return None


def claim_target(claim, base_knobs, search_space):
    """The design that tests the claim from `base_knobs`: knob set to the value,
    or stepped once in the direction. None if the step is impossible."""
    knob = claim["knob"]
    target = dict(base_knobs)
    if is_direction(claim):
        new_value = stepped_value(search_space, knob, base_knobs[knob], str(claim["value"]))
        if new_value is None:
            return None
        target[knob] = new_value
        return target
    for allowed in search_space[knob]:
        if str(allowed) == str(claim["value"]):
            target[knob] = allowed
    return target


def differs_only_in(knobs_a, knobs_b, knob):
    """True when the two designs differ in `knob` and in nothing else."""
    if str(knobs_a.get(knob)) == str(knobs_b.get(knob)):
        return False
    for other in knobs_a:
        if other != knob and str(knobs_a[other]) != str(knobs_b.get(other)):
            return False
    return True


def claim_sibling(claim, knobs, baseline_knobs, search_space, history=None):
    """The design one claim-step BEHIND `knobs`. Value claim: the same design with
    the knob at another value: a MEASURED one from `history` when there is one
    (a controlled pair exists whatever the search started from), else the
    untouched chip's value, else the first other allowed value. Direction claim:
    one step against the direction (any adjacent step). None if `knobs` does not
    carry the claim at all."""
    knob = claim["knob"]
    if is_direction(claim):
        direction = str(claim["value"])
        opposite = "down" if direction == "up" else "up"
        # Any adjacent step counts, on either side of the baseline: "bigger LLC
        # helps" is the same claim at 1MB->2MB as at 2MB->4MB. Counting only
        # steps beyond the baseline would make capacity claims untestable
        # on chips whose budget cannot afford the step above the baseline.
        values = ordered_values(search_space, knob)
        position = None
        for index, allowed in enumerate(values):
            if str(allowed) == str(knobs[knob]):
                position = index
        if position is None:
            return None
        sibling_value = stepped_value(search_space, knob, knobs[knob], opposite)
        if sibling_value is None:
            return None
        sibling = dict(knobs)
        sibling[knob] = sibling_value
        return sibling
    if str(knobs[knob]) != str(claim["value"]):
        return None
    if history is not None:
        for entry in history:
            if differs_only_in(entry["knobs"], knobs, knob):
                return dict(entry["knobs"])
    sibling = dict(knobs)
    if str(baseline_knobs[knob]) != str(claim["value"]):
        sibling[knob] = baseline_knobs[knob]
        return sibling
    for allowed in search_space[knob]:
        if str(allowed) != str(claim["value"]):
            sibling[knob] = allowed
            return sibling
    return None


def paired_gains(history, claim, baseline_knobs, search_space, objective):
    """Every controlled comparison in `history` for one claim: pairs of runs
    that differ ONLY by one claim-step in `knob`. Gains in percent."""
    gains = []
    for entry in history:
        sibling = claim_sibling(claim, entry["knobs"], baseline_knobs, search_space, history)
        if sibling is None:
            continue
        sibling_metrics = find_measured(history, sibling)
        if sibling_metrics is None:
            continue
        gains.append(100.0 * (entry["metrics"][objective] / sibling_metrics[objective] - 1.0))
    return gains


def effective_gain(rule, history, baseline_knobs, search_space, objective):
    """The gain a rule uses on this chip: the mean of its controlled pairs here if
    any exist (chip-adapted), else the gain measured where the rule was learned."""
    gains = paired_gains(history, rule["claim"], baseline_knobs, search_space, objective)
    if len(gains) > 0:
        return sum(gains) / len(gains)
    return rule["claim"]["gain_pct"]


def measured_here(rule, history, baseline_knobs, search_space, objective):
    """True once at least one controlled pair for the claim exists on this problem."""
    gains = paired_gains(history, rule["claim"], baseline_knobs, search_space, objective)
    return len(gains) > 0


# ---- rule shifts: how a speaking rule takes part in selection ----
# The stacked shift on one candidate is capped at this many times the largest
# single effect among the speaking rules, so several rules pointing at one
# design cannot build a "super candidate" the search then exploits.
SHIFT_CAP_FACTOR = 1.5
# Until a controlled pair on THIS chip has measured a rule's size, the rule
# tilts by a fixed modest step in its claimed direction only: sizes learned on
# another chip and class over-promised (the first cell-2 run), directions held.
DIRECTION_STEP_PCT = 5.0


def carries_claim(knobs, claim, baseline_knobs, search_space):
    """Does this design sit inside the region a claim talks about? A value claim:
    the knob has that value. A direction claim: the knob is anywhere above (up)
    or below (down) the baseline value. Flat over the whole region on purpose:
    the rule says "bigger helps here", not how much bigger."""
    knob = claim["knob"]
    if is_direction(claim):
        values = ordered_values(search_space, knob)
        position = None
        base_position = None
        for index, allowed in enumerate(values):
            if str(allowed) == str(knobs[knob]):
                position = index
            if str(allowed) == str(baseline_knobs[knob]):
                base_position = index
        if position is None or base_position is None:
            return False
        if str(claim["value"]) == "up":
            return position > base_position
        return position < base_position
    return str(knobs[knob]) == str(claim["value"])


def rule_shifts(rules, history, baseline_knobs, search_space, objective):
    """One mean shift per speaking rule, in log-speedup units: the measured effect
    times the rule's credibility, divided by 1 + the number of real observations
    already inside the rule's region (once the simulator has spoken there, the
    rule goes quiet). Returns {"shifts": [...], "cap": max total shift per design}."""
    shifts = []
    largest_effect = 0.0
    for rule in rules:
        if measured_here(rule, history, baseline_knobs, search_space, objective):
            gain = effective_gain(rule, history, baseline_knobs, search_space, objective)
        else:
            gain = DIRECTION_STEP_PCT
            if rule["claim"]["gain_pct"] < 0:
                gain = -DIRECTION_STEP_PCT
        if gain < -90.0:
            gain = -90.0
        log_effect = math.log(1.0 + gain / 100.0)
        if abs(log_effect) > largest_effect:
            largest_effect = abs(log_effect)
        observations = 0
        for entry in history:
            if carries_claim(entry["knobs"], rule["claim"], baseline_knobs, search_space):
                observations += 1
        shifts.append({"rule_id": rule["id"], "claim": rule["claim"],
                       "log_shift": log_effect * rule_confidence(rule) / (1.0 + observations)})
    return {"shifts": shifts, "cap": SHIFT_CAP_FACTOR * largest_effect}


def shift_for(knobs, shifts, baseline_knobs, search_space):
    """Total (capped) mean shift for one design, in log-speedup units."""
    total = 0.0
    for shift in shifts["shifts"]:
        if carries_claim(knobs, shift["claim"], baseline_knobs, search_space):
            total += shift["log_shift"]
    if total > shifts["cap"]:
        total = shifts["cap"]
    if total < -shifts["cap"]:
        total = -shifts["cap"]
    return total


def shifted_means(means, knobs_list, shifts, baseline_knobs, search_space):
    """Apply the rule shifts to a batch of predicted means (objective units)."""
    shifted = []
    for mean, knobs in zip(means, knobs_list):
        shifted.append(float(mean) * math.exp(shift_for(knobs, shifts, baseline_knobs, search_space)))
    return shifted


def rule_forecast(sibling_value, gain_pct):
    return sibling_value * (1.0 + gain_pct / 100.0)


def gather_forecasts(candidates, surrogate, model, rules, replies, baseline_knobs,
                     baseline_metrics, objective, history, search_space):
    """For each candidate name: {"opinions": [(forecaster_id, predicted)], "surrogate_std",
    "siblings": {rule id: {"knobs", "value", "measured"}} for every rule whose claim the
    candidate carries (the sibling is the design one claim-step behind it).

    `rules` must already be the speaking rules for this problem. A rule speaks
    about every candidate that carries its claim (the same region its mean shift
    covers): its forecast is the candidate's sibling times the claimed effect.
    """
    names = list(candidates.keys())
    knobs_list = []
    for name in names:
        knobs_list.append(candidates[name])
    means, stds = surrogate.predict_many(model, knobs_list)

    gains = []
    for rule in rules:
        gains.append(effective_gain(rule, history, baseline_knobs, search_space, objective))

    # Siblings the rules need: predict them in one batch too.
    sibling_requests = []          # (candidate index, rule index, sibling knobs)
    for candidate_index, name in enumerate(names):
        knobs = candidates[name]
        for rule_index, rule in enumerate(rules):
            claim = rule["claim"]
            sibling = claim_sibling(claim, knobs, baseline_knobs, search_space, history)
            if sibling is None:
                continue
            sibling_requests.append((candidate_index, rule_index, sibling))
    sibling_knobs_list = []
    for _, _, sibling in sibling_requests:
        sibling_knobs_list.append(sibling)
    sibling_means = []
    if len(sibling_knobs_list) > 0:
        sibling_means, _ = surrogate.predict_many(model, sibling_knobs_list)

    forecasts = {}
    for candidate_index, name in enumerate(names):
        forecasts[name] = {"opinions": [("SURROGATE", float(means[candidate_index]))],
                           "surrogate_std": float(stds[candidate_index]),
                           "siblings": {}}
    for request_index, (candidate_index, rule_index, sibling) in enumerate(sibling_requests):
        name = names[candidate_index]
        rule = rules[rule_index]
        measured = find_measured(history, sibling)
        if measured is not None:
            sibling_value = measured[objective]
        else:
            sibling_value = float(sibling_means[request_index])
        forecasts[name]["siblings"][rule["id"]] = {"knobs": sibling, "value": sibling_value,
                                                   "measured": measured is not None}
        forecasts[name]["opinions"].append((rule["id"], rule_forecast(sibling_value, gains[rule_index])))
    for reply in replies:
        if reply["name"] in forecasts:
            forecasts[reply["name"]]["opinions"].append((reply["id"], reply["predicted"]))
    return forecasts


def expected_improvement(mean, std, best_so_far):
    """Standard EI for maximisation."""
    if std <= 0.0:
        return max(0.0, mean - best_so_far)
    z = (mean - best_so_far) / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mean - best_so_far) * cdf + std * pdf


def pick_expected_improvement(candidates, surrogate, model, history, search_space, objective,
                              how_many, best_so_far, seed=0, shifts=None, baseline_knobs=None,
                              already_chosen=None, honest_std=False, explore=0.0):
    """Textbook batch Bayesian optimisation: GP + Expected Improvement, batch of
    `how_many` via the kriging believer (each pick is added to the training set
    at its predicted mean and the GP is refit before the next pick).
    shifts: the speaking rules' mean shifts (rule_shifts); None for the pure BO arms.
    already_chosen: names picked this round by another route (claim test, reply).
    explore: a mild bonus of `explore` x predicted std added to EI (the
    `bo_pooled_x` arm); 0 is textbook EI."""
    names = list(candidates.keys())
    random.Random(seed).shuffle(names)
    believed_history = list(history)
    chosen = []
    if already_chosen is not None:
        chosen = list(already_chosen)
    reference = model["reference"]
    knobs_list = []
    for name in names:
        knobs_list.append(candidates[name])
    for pick in range(how_many):
        if pick > 0:
            model = surrogate.fit(believed_history, search_space, objective,
                                  honest_std=honest_std, reference=reference)
        means, stds = surrogate.predict_many(model, knobs_list)
        if shifts is not None:
            means = shifted_means(means, knobs_list, shifts, baseline_knobs, search_space)
        best_name = None
        best_score = None
        for index, name in enumerate(names):
            if name in chosen:
                continue
            score = expected_improvement(float(means[index]), float(stds[index]), best_so_far)
            score = score + explore * float(stds[index])
            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        chosen.append(best_name)
        believer_index = names.index(best_name)
        believed_history.append({"name": best_name, "knobs": candidates[best_name],
                                 "metrics": {objective: float(means[believer_index])},
                                 "reference": reference})
    if already_chosen is not None:
        return chosen[len(already_chosen):]
    return chosen


ANALYST_PRIOR_CREDIBILITY = 0.5   # the analyst starts at coin-flip credibility on every problem


def analyst_credibility(record):
    """record = {"wins": n, "losses": m} of the analyst's hypothesis bets on THIS
    problem. Like a rule, the analyst earns influence from its own bets: it
    starts at a coin flip and moves toward its win rate."""
    total = PRIOR_WEIGHT + record["wins"] + record["losses"]
    return (ANALYST_PRIOR_CREDIBILITY * PRIOR_WEIGHT + record["wins"]) / total


def one_knob_effects(designs, objective):
    """Every controlled pair among `designs` (two designs that differ in exactly
    one knob), grouped by (knob, from, to): mean gain in percent and the number
    of pairs, largest mean first. Chip- and space-agnostic: it only reads knobs."""
    groups = {}
    for index_a in range(len(designs)):
        for index_b in range(index_a + 1, len(designs)):
            knobs_a = designs[index_a]["knobs"]
            knobs_b = designs[index_b]["knobs"]
            differing = []
            for knob in knobs_a:
                if str(knobs_a[knob]) != str(knobs_b.get(knob)):
                    differing.append(knob)
            if len(differing) != 1:
                continue
            knob = differing[0]
            value_a = designs[index_a]["metrics"][objective]
            value_b = designs[index_b]["metrics"][objective]
            if value_a <= 0 or value_b <= 0:
                continue
            # Report the step in the order the values appear in the design list,
            # low to high when they compare, so "from -> to" reads naturally.
            if str(knobs_a[knob]) < str(knobs_b[knob]):
                key = (knob, str(knobs_a[knob]), str(knobs_b[knob]))
                gain = 100.0 * (value_b / value_a - 1.0)
            else:
                key = (knob, str(knobs_b[knob]), str(knobs_a[knob]))
                gain = 100.0 * (value_a / value_b - 1.0)
            groups.setdefault(key, []).append(gain)
    effects = []
    for (knob, low, high), gains in groups.items():
        mean = sum(gains) / len(gains)
        effects.append({"knob": knob, "from": low, "to": high, "gain_pct": mean, "pairs": len(gains)})
    effects.sort(key=lambda effect: -abs(effect["gain_pct"]))
    return effects


def format_effects(effects, limit=25):
    lines = []
    for effect in effects[:limit]:
        lines.append("{} {} -> {}: {:+.2f}% over {} pair(s)".format(
            effect["knob"], effect["from"], effect["to"], effect["gain_pct"], effect["pairs"]))
    if len(lines) == 0:
        return "(none measured yet)"
    return "\n".join(lines)


def stall_scan_candidate(history, is_candidate, name_of, search_space, surrogate, model, objective):
    """The architect's move when the search stalls: take the best design so far
    and change ONE knob: a categorical knob to a value nobody has tried yet on
    this problem (another prefetcher, another replacement policy), or an ordinal
    knob one step up or down. Among such unmeasured, in-budget designs, the one
    the surrogate is least sure about. None if there is none.

    Without it, two rule priors were enough for EI to exploit one basin for 24
    designs and never try a knob value worth +2.7%."""
    incumbent = None
    for entry in history:
        if incumbent is None or entry["metrics"][objective] > incumbent["metrics"][objective]:
            incumbent = entry
    if incumbent is None:
        return None
    tried = {}
    for entry in history:
        for knob, value in entry["knobs"].items():
            tried.setdefault(knob, set()).add(str(value))
    measured_names = set()
    for entry in history:
        measured_names.add(entry["name"])
    scan_names = []
    scan_knobs = []
    for knob, values in search_space.items():
        is_categorical = isinstance(values[0], str)
        neighbour_values = []
        if is_categorical:
            # Categorical: every value nobody has tried on this problem.
            for value in values:
                if str(value) not in tried.get(knob, set()):
                    neighbour_values.append(value)
        else:
            # Ordinal (sizes, ways, MSHRs): one step up and one step down from the
            # incumbent: a categorical-only scan can never propose "llc_sets one
            # step up", and that has been the missing move.
            for direction in ["up", "down"]:
                stepped = stepped_value(search_space, knob, incumbent["knobs"][knob], direction)
                if stepped is not None:
                    neighbour_values.append(stepped)
        for value in neighbour_values:
            design = dict(incumbent["knobs"])
            design[knob] = value
            if not is_candidate(design):
                continue
            name = name_of(design)
            if name in measured_names or name in scan_names:
                continue
            scan_names.append(name)
            scan_knobs.append(design)
    if len(scan_names) == 0:
        return None
    means, stds = surrogate.predict_many(model, scan_knobs)
    best_index = 0
    for index in range(len(scan_names)):
        if stds[index] > stds[best_index]:
            best_index = index
    return scan_names[best_index], scan_knobs[best_index]


def untested_claim_tests(rules, history, baseline_knobs, is_candidate, name_of, search_space, objective):
    """Claim tests still owed on this chip: for each speaking rule with no controlled
    pair here, the INCUMBENT (best design so far) moved one claim-step, if that
    design is a candidate: a pair next to the best design, not next to the untouched
    chip. Returned best first: credibility x |transferred gain|."""
    incumbent = history[0]["knobs"]
    best_value = None
    for entry in history:
        if best_value is None or entry["metrics"][objective] > best_value:
            best_value = entry["metrics"][objective]
            incumbent = entry["knobs"]
    scored = []
    seen = set()
    for rule in rules:
        claim = rule["claim"]
        key = (claim["knob"], str(claim["value"]))
        if key in seen:
            continue
        seen.add(key)
        gains = paired_gains(history, claim, baseline_knobs, search_space, objective)
        if len(gains) > 0:
            continue
        test_knobs = claim_target(claim, incumbent, search_space)
        if test_knobs is None or not is_candidate(test_knobs):
            continue
        test_name = name_of(test_knobs)
        if find_measured(history, test_knobs) is not None:
            continue
        scored.append((rule_confidence(rule) * abs(claim["gain_pct"]), test_name, rule["id"], test_knobs))
    scored.sort(key=lambda item: -item[0])
    tests = []
    for score, name, rule_id, knobs in scored:
        tests.append((name, rule_id, knobs))
    return tests
