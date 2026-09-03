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
    op = condition["op"]
    threshold = float(condition["value"])
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


def pick_most_disagreed(forecasts, how_many, best_so_far, seed=0):
    """Ties (e.g. every candidate equal after only the baseline run) are broken
    at random with the given seed, never by config name."""
    import random
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


def rule_priors(rules, baseline_knobs, baseline_metrics, objective):
    """One virtual experiment per DISTINCT claim (knob, value) among the rules that
    apply: the baseline with that knob switched, at the confidence-weighted mean of
    the promised gains. Rules with a losing record (confidence < 0.5) do not steer."""
    grouped = {}
    for rule in rules:
        if rule["status"] != "active":
            continue
        if not condition_holds(rule["condition"], baseline_metrics):
            continue
        confidence = rule_confidence(rule)
        if confidence < 0.5:
            continue
        key = (rule["claim"]["knob"], str(rule["claim"]["value"]))
        grouped.setdefault(key, []).append((rule["claim"]["gain_pct"], confidence))

    priors = []
    for (knob, value), claims in grouped.items():
        total_weight = sum(confidence for _, confidence in claims)
        mean_gain = sum(gain * confidence for gain, confidence in claims) / total_weight
        best_confidence = max(confidence for _, confidence in claims)
        knobs = dict(baseline_knobs)
        knobs[knob] = value
        priors.append({"knobs": knobs, "value": baseline_metrics[objective] * (1.0 + mean_gain / 100.0),
                       "confidence": best_confidence})
    return priors
