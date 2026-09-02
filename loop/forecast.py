"""Forecasters and experiment selection.

Three kinds of forecaster give a predicted IPC for a candidate config:
  - the surrogate (always has an opinion),
  - playbook rules (only when their condition holds and the candidate
    uses the knob value they talk about),
  - the analyst's hypotheses (only for the config they proposed).
We run the experiments where the forecasters DISAGREE most; a config
everyone agrees on teaches us nothing.
"""

PRIOR_CONFIDENCE = 0.7   # a brand-new rule's confidence, before any bets
PRIOR_WEIGHT = 2         # how many imaginary bets that prior is worth


def condition_holds(condition, baseline_metrics):
    """condition: {"metric": "LLC_mpki", "op": ">=", "value": 20}."""
    value = baseline_metrics[condition["metric"]]
    if condition["op"] == ">=":
        return value >= condition["value"]
    if condition["op"] == "<":
        return value < condition["value"]
    raise ValueError("unsupported condition op: " + condition["op"])


def rule_forecast(rule, knobs, baseline_metrics, objective):
    """Predicted IPC if the rule speaks about this candidate, else None."""
    if rule["status"] != "active":
        return None
    if not condition_holds(rule["condition"], baseline_metrics):
        return None
    claim = rule["claim"]
    if str(knobs[claim["knob"]]) != str(claim["value"]):
        return None
    return baseline_metrics[objective] * (1.0 + claim["gain_pct"] / 100.0)


def rule_confidence(rule):
    """Prior confidence, pulled toward the rule's actual win rate."""
    total = PRIOR_WEIGHT + rule["wins"] + rule["losses"]
    return (PRIOR_CONFIDENCE * PRIOR_WEIGHT + rule["wins"]) / total


def gather_forecasts(candidates, surrogate, model, rules, hypotheses, baseline_metrics, objective):
    """For each candidate name: list of (forecaster_id, predicted_ipc), plus surrogate std."""
    forecasts = {}
    for name in candidates:
        knobs = candidates[name]
        prediction, std = surrogate.predict(model, knobs)
        opinions = [("SURROGATE", prediction)]
        for rule in rules:
            predicted = rule_forecast(rule, knobs, baseline_metrics, objective)
            if predicted is not None:
                opinions.append((rule["id"], predicted))
        for hypothesis in hypotheses:
            if hypothesis["name"] == name:
                opinions.append((hypothesis["id"], hypothesis["predicted"]))
        forecasts[name] = {"opinions": opinions, "surrogate_std": std}
    return forecasts


def disagreement(entry, best_so_far):
    """Spread between the most optimistic and most pessimistic forecaster, plus
    surrogate doubt, plus how much the optimist expects to beat the best so far.

    The last term keeps the loop from spending runs on "is this config terrible
    or merely bad?" - a dispute nobody needs settled.
    """
    values = []
    for _, predicted in entry["opinions"]:
        values.append(predicted)
    spread = max(values) - min(values)
    promise = max(0.0, max(values) - best_so_far)
    return spread + entry["surrogate_std"] + promise


def pick_most_disagreed(forecasts, how_many, best_so_far):
    scored = []
    for name in forecasts:
        scored.append((disagreement(forecasts[name], best_so_far), name))
    scored.sort(reverse=True)
    chosen = []
    for score, name in scored[:how_many]:
        chosen.append(name)
    return chosen


def rule_priors(rules, baseline_knobs, baseline_metrics, objective):
    """One virtual experiment per applicable rule: the baseline with the rule's
    knob switched to its value, at the objective the rule promises."""
    priors = []
    for rule in rules:
        if rule["status"] != "active":
            continue
        if not condition_holds(rule["condition"], baseline_metrics):
            continue
        knobs = dict(baseline_knobs)
        knobs[rule["claim"]["knob"]] = rule["claim"]["value"]
        predicted = baseline_metrics[objective] * (1.0 + rule["claim"]["gain_pct"] / 100.0)
        priors.append({"knobs": knobs, "value": predicted, "confidence": rule_confidence(rule)})
    return priors
