"""Forecasters and experiment selection.

Three kinds of forecaster give a predicted objective for a candidate config:
  - the surrogate (always has an opinion),
  - playbook rules (only when their condition holds on this baseline, the
    candidate uses the knob value they talk about, and their record is
    still credible),
  - the analyst's hypotheses (only for the config they proposed).
We run the experiments where the forecasters DISAGREE most; a config
everyone agrees on teaches us nothing.

A rule is a DELTA claim: "switching knob K to value V changes the objective
by gain_pct %". Its forecast for a candidate is therefore relative to the
candidate's SIBLING (same config with K at the baseline value): the measured
sibling when we have it, the surrogate's estimate of it otherwise. Forecasting
baseline * (1 + gain) for every candidate, as v2 did, gave the same number to
hundreds of different configs and made rule bets a coin flip.
"""

import math
import random

PRIOR_CONFIDENCE = 0.7   # a brand-new rule's confidence, before any bets
PRIOR_WEIGHT = 2         # how many imaginary bets that prior is worth
CREDIBLE_CONFIDENCE = 0.5   # below this a rule stays silent on the current problem
CLAIM_TEST_BONUS_FRACTION = 0.05   # selection bonus for a run that settles a claim as a controlled comparison


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


def condition_holds(condition, baseline_metrics):
    """Backward-compatible single-clause check (audit, plots)."""
    return clause_holds(condition, baseline_metrics)


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


TRANSFER_SHRINK = 0.5   # a gain measured on another chip is carried over at half strength
RULE_SCOPE_OTHER_KNOBS = 1   # a rule speaks only about designs within this many OTHER knob changes of the baseline

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


def claim_sibling(claim, knobs, baseline_knobs, search_space):
    """The design one claim-step BEHIND `knobs`: knob back at baseline (value
    claim) or one step against the direction (direction claim, any adjacent
    step). None if `knobs` does not carry the claim at all."""
    knob = claim["knob"]
    if is_direction(claim):
        direction = str(claim["value"])
        opposite = "down" if direction == "up" else "up"
        # Any adjacent step counts, on either side of the baseline: "bigger LLC
        # helps" is the same claim at 1MB->2MB as at 2MB->4MB. Until Sep 7 only
        # steps beyond the baseline counted, which made capacity claims untestable
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
    if str(baseline_knobs[knob]) == str(claim["value"]):
        return None
    sibling = dict(knobs)
    sibling[knob] = baseline_knobs[knob]
    return sibling


def within_rule_scope(knobs, baseline_knobs, claim_knob):
    """A rule's measured gain is a MARGINAL effect near the baseline; with many
    other knobs changed the interactions are unknown, so the rule stays silent."""
    changed = 0
    for knob in knobs:
        if knob == claim_knob or knob == "soc":
            continue
        if str(knobs[knob]) != str(baseline_knobs.get(knob, knobs[knob])):
            changed += 1
    return changed <= RULE_SCOPE_OTHER_KNOBS


def paired_gains(history, claim, baseline_knobs, search_space, objective):
    """Every controlled comparison in `history` for one claim: pairs of runs
    that differ ONLY by one claim-step in `knob`. Gains in percent."""
    gains = []
    for entry in history:
        sibling = claim_sibling(claim, entry["knobs"], baseline_knobs, search_space)
        if sibling is None:
            continue
        sibling_metrics = find_measured(history, sibling)
        if sibling_metrics is None:
            continue
        gains.append(100.0 * (entry["metrics"][objective] / sibling_metrics[objective] - 1.0))
    return gains


def effective_gain(rule, history, baseline_knobs, search_space, objective):
    """The gain a rule uses on this chip: the mean of its controlled pairs here if
    any exist (chip-adapted), else the transferred gain shrunk toward zero. The
    magnitude of a one-knob effect rarely transfers exactly; its direction does."""
    gains = paired_gains(history, rule["claim"], baseline_knobs, search_space, objective)
    if len(gains) > 0:
        return sum(gains) / len(gains)
    return rule["claim"]["gain_pct"] * TRANSFER_SHRINK


def rule_forecast(sibling_value, gain_pct):
    return sibling_value * (1.0 + gain_pct / 100.0)


def gather_forecasts(candidates, surrogate, model, rules, hypotheses, baseline_knobs,
                     baseline_metrics, objective, history, search_space):
    """For each candidate name: {"opinions": [(forecaster_id, predicted)], "surrogate_std",
    "claim_tests": [rule ids whose claim this run would settle as a controlled comparison]}.

    `rules` must already be the speaking rules for this problem. A rule speaks
    about a candidate only when the candidate carries its claim and lies within
    the rule's scope (few other knobs changed from the baseline).
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
            if not within_rule_scope(knobs, baseline_knobs, claim["knob"]):
                continue
            sibling = claim_sibling(claim, knobs, baseline_knobs, search_space)
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
                           "claim_tests": []}
    for request_index, (candidate_index, rule_index, sibling) in enumerate(sibling_requests):
        name = names[candidate_index]
        rule = rules[rule_index]
        measured = find_measured(history, sibling)
        if measured is not None:
            sibling_value = measured[objective]
            forecasts[name]["claim_tests"].append(rule["id"])
        else:
            sibling_value = float(sibling_means[request_index])
        forecasts[name]["opinions"].append((rule["id"], rule_forecast(sibling_value, gains[rule_index])))
    for hypothesis in hypotheses:
        if hypothesis["name"] in forecasts:
            forecasts[hypothesis["name"]]["opinions"].append((hypothesis["id"], hypothesis["predicted"]))
    return forecasts


def disagreement(entry, best_so_far):
    """Spread between the most optimistic and most pessimistic forecaster, plus
    surrogate doubt, plus how much the optimist expects to beat the best so far,
    plus a bonus when the run settles a rule's claim as a controlled comparison.

    The promise term keeps the loop from spending runs on "is this config terrible
    or merely bad?" - a dispute nobody needs settled. The claim-test bonus makes
    the loop prefer runs whose outcome is attributable to one knob: those are the
    bets that can re-scope or confirm a rule.
    """
    values = []
    for _, predicted in entry["opinions"]:
        values.append(predicted)
    spread = max(values) - min(values)
    promise = max(0.0, max(values) - best_so_far)
    claim_bonus = 0.0
    if len(entry["claim_tests"]) > 0:
        claim_bonus = CLAIM_TEST_BONUS_FRACTION * abs(best_so_far)
    return spread + entry["surrogate_std"] + promise + claim_bonus


def pick_most_disagreed(forecasts, how_many, best_so_far, seed=0):
    """Ties (e.g. every candidate equal after only the baseline run) are broken
    at random with the given seed, never by config name."""
    names = list(forecasts.keys())
    random.Random(seed).shuffle(names)
    scored = []
    for position, name in enumerate(names):
        scored.append((disagreement(forecasts[name], best_so_far), -position, name))
    scored.sort(reverse=True)
    chosen = []
    for score, position, name in scored[:how_many]:
        chosen.append(name)
    return chosen


def expected_improvement(mean, std, best_so_far):
    """Standard EI for maximisation."""
    if std <= 0.0:
        return max(0.0, mean - best_so_far)
    z = (mean - best_so_far) / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mean - best_so_far) * cdf + std * pdf


def pick_expected_improvement(candidates, surrogate, model, history, search_space, objective,
                              how_many, best_so_far, seed=0, priors=None, already_chosen=None,
                              honest_std=False):
    """Textbook batch Bayesian optimisation: GP + Expected Improvement, batch of
    `how_many` via the kriging believer (each pick is added to the training set
    at its predicted mean and the GP is refit before the next pick).
    priors: virtual points kept in every refit (rule / hypothesis warm-start).
    already_chosen: names picked this round by another route (claim tests)."""
    if priors is None:
        priors = []
    names = list(candidates.keys())
    random.Random(seed).shuffle(names)
    believed_history = list(history)
    chosen = []
    if already_chosen is not None:
        chosen = list(already_chosen)
    reference = model["reference"]
    for pick in range(how_many):
        if pick > 0:
            model = surrogate.fit(believed_history, search_space, objective, priors,
                                  honest_std=honest_std, reference=reference)
        knobs_list = []
        for name in names:
            knobs_list.append(candidates[name])
        means, stds = surrogate.predict_many(model, knobs_list)
        best_name = None
        best_score = None
        for index, name in enumerate(names):
            if name in chosen:
                continue
            score = expected_improvement(float(means[index]), float(stds[index]), best_so_far)
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


def hypothesis_priors(hypotheses, candidates, record):
    """The analyst's forecasts as virtual experiments: its belief warm-starts the
    surrogate at (analyst credibility x stated confidence), and nothing at all
    once its record on this problem falls below the credibility floor. v4 trusted
    the LLM's stated 0.7-0.9 confidence; its bets won 32%, and the priors
    steered the search into bad designs."""
    credibility = analyst_credibility(record)
    if credibility < CREDIBLE_CONFIDENCE:
        return []
    priors = []
    for hypothesis in hypotheses:
        if hypothesis["name"] not in candidates:
            continue
        priors.append({"knobs": candidates[hypothesis["name"]], "value": hypothesis["predicted"],
                       "confidence": hypothesis["confidence"] * credibility})
    return priors


def stall_scan_candidate(history, candidates, search_space, surrogate, model, objective):
    """The architect's move when the search stalls: take the best design so far
    and change ONE categorical knob to a value nobody has tried yet on this
    problem (another prefetcher, another replacement policy). Among such
    designs, the one the surrogate is least sure about. None if every value of
    every categorical knob has been tried or no such design is in budget.

    Sep 7 autopsy: with two rule priors, EI exploited the SPP + capacity basin
    for 24 designs and never tried the LLC prefetcher, which was worth +2.7%."""
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
    scan_names = []
    scan_knobs = []
    for knob, values in search_space.items():
        is_categorical = isinstance(values[0], str)
        if not is_categorical:
            continue
        for value in values:
            if str(value) in tried.get(knob, set()):
                continue
            design = dict(incumbent["knobs"])
            design[knob] = value
            for name, knobs in candidates.items():
                if same_knobs(knobs, design):
                    scan_names.append(name)
                    scan_knobs.append(knobs)
                    break
    if len(scan_names) == 0:
        return None
    means, stds = surrogate.predict_many(model, scan_knobs)
    best_index = 0
    for index in range(len(scan_names)):
        if stds[index] > stds[best_index]:
            best_index = index
    return scan_names[best_index]


def untested_claim_tests(rules, history, baseline_knobs, candidates, search_space, objective):
    """Claim tests still owed on this chip: for each speaking rule with no controlled
    pair here, the baseline moved one claim-step (if that design is a candidate).
    Returned best first: credibility x |transferred gain|."""
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
        test_knobs = claim_target(claim, baseline_knobs, search_space)
        if test_knobs is None:
            continue
        test_name = None
        for name in candidates:
            if same_knobs(candidates[name], test_knobs):
                test_name = name
        if test_name is None:
            continue
        scored.append((rule_confidence(rule) * abs(claim["gain_pct"]), test_name, rule["id"]))
    scored.sort(reverse=True)
    tests = []
    for score, name, rule_id in scored:
        tests.append((name, rule_id))
    return tests


def rule_priors(rules, baseline_knobs, baseline_metrics, objective, history, search_space):
    """One virtual experiment per DISTINCT claim among the speaking rules: the
    baseline moved one claim-step, at the confidence-weighted mean of the
    effective gains. `rules` must already be the speaking rules."""
    grouped = {}
    for rule in rules:
        confidence = rule_confidence(rule)
        key = (rule["claim"]["knob"], str(rule["claim"]["value"]))
        gain = effective_gain(rule, history, baseline_knobs, search_space, objective)
        grouped.setdefault(key, []).append((gain, confidence, rule["claim"]))

    priors = []
    for key, claims in grouped.items():
        total_weight = 0.0
        weighted_gain = 0.0
        best_confidence = 0.0
        for gain, confidence, _ in claims:
            total_weight += confidence
            weighted_gain += gain * confidence
            if confidence > best_confidence:
                best_confidence = confidence
        mean_gain = weighted_gain / total_weight
        knobs = claim_target(claims[0][2], baseline_knobs, search_space)
        if knobs is None:
            continue
        priors.append({"knobs": knobs, "value": baseline_metrics[objective] * (1.0 + mean_gain / 100.0),
                       "confidence": best_confidence})
    return priors
