"""The loop: seed -> forecasters bet -> run the most disputed configs -> settle -> repeat.

Simulator-agnostic. Everything domain-specific arrives in the `problem` dict:
  name          - label, e.g. "B_midrange/mcf"
  search_space  - {knob: [allowed values]}
  candidates    - {config name: knobs} for every config allowed to be run
  baseline      - the knobs of the untouched design (must be in candidates)
  evaluate      - function(knobs) -> metrics dict (the ONLY call to a simulator)
  evaluate_many - function([knobs]) -> [metrics], may run them in parallel
  objective     - metric name to maximize, e.g. "ipc"
  table_metrics - metric names shown to the analyst

One round: fit surrogate -> analyst hypotheses -> every forecaster predicts
every candidate -> run the most disputed ones -> settle bets -> re-scope
losing rules. Every forecast (chosen or not) is logged for calibration plots.
"""

import json

from loop import analyst, forecast, playbook


def run_loop(problem, rounds, per_round, store, surrogate, use_rules, use_analyst, tag,
             prior_history=None):
    """Returns {"history": [...], "rounds": [...]}; bets land in `store`.

    surrogate:   module with fit / predict / probability_at_least
    use_rules:   let playbook rules forecast and bet (transfer arm)
    use_analyst: let the LLM propose hypotheses (else statistical search only)
    tag:         prefix for hypothesis ids so arms can be told apart in the ledger
    prior_history: runs from OTHER designs the surrogate may learn from
                 (pooled-surrogate transfer); they never count as simulations
    """
    if prior_history is None:
        prior_history = []
    objective = problem["objective"]
    candidates = dict(problem["candidates"])
    baseline_name = name_of(problem, problem["baseline"])
    baseline_metrics = problem["evaluate"](problem["baseline"])
    history = [{"name": baseline_name, "knobs": problem["baseline"], "metrics": baseline_metrics}]
    del candidates[baseline_name]

    rules = []
    if use_rules:
        rules = store["rules"]

    round_logs = []
    for round_number in range(1, rounds + 1):
        model = surrogate.fit(prior_history + history, problem["search_space"], objective)
        hypotheses = []
        if use_analyst:
            hypotheses = ask_hypotheses(problem, history, rules, candidates, per_round,
                                        "{}-HYP-r{}".format(tag, round_number))

        forecasts = forecast.gather_forecasts(candidates, surrogate, model, rules,
                                              hypotheses, baseline_metrics, objective)
        chosen = forecast.pick_most_disagreed(forecasts, per_round)
        round_logs.append({"round": round_number, "forecasts": forecasts,
                           "hypotheses": hypotheses, "chosen": chosen})

        # 1) every forecaster bets on every chosen config, BEFORE anything runs
        events = {}
        bets_by_name = {}
        for name in chosen:
            opinions = forecasts[name]["opinions"]
            bets_by_name[name] = []
            if len(opinions) > 1:
                # Settle at the midpoint of the dispute: exactly where they disagree.
                values = [predicted for _, predicted in opinions]
                threshold = (max(values) + min(values)) / 2.0
                events[name] = "{} >= {:.4f}".format(objective, threshold)
                for forecaster_id, predicted in opinions:
                    probability = to_probability(forecaster_id, predicted, threshold, surrogate,
                                                 model, candidates[name], rules, hypotheses)
                    bets_by_name[name].append(
                        playbook.place_bet(store, forecaster_id, name, events[name], probability))

        # 2) the chosen configs run in parallel (one CHIA task each)
        knobs_list = [candidates[name] for name in chosen]
        metrics_list = problem["evaluate_many"](knobs_list)

        # 3) settle; a rule that lost gets re-scoped
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            history.append({"name": name, "knobs": knobs, "metrics": metrics})
            del candidates[name]
            for bet_id in bets_by_name[name]:
                happened = check_event(events[name], metrics)
                playbook.settle_bet(store, bet_id, happened)
                lost_rule = losing_rule(store, bet_id, rules, happened)
                if lost_rule is not None:
                    rescope(lost_rule, problem, baseline_metrics, name, metrics)
            print("[{}] round {} | {} | {}={:.4f} | {} bets".format(
                tag, round_number, name, objective, metrics[objective],
                len(bets_by_name[name])), flush=True)
    return {"history": history, "rounds": round_logs}


def name_of(problem, knobs):
    for name in problem["candidates"]:
        if problem["candidates"][name] == knobs:
            return name
    raise KeyError("knobs not in candidates: " + json.dumps(knobs))


def ask_hypotheses(problem, history, rules, candidates, how_many, id_prefix):
    """Analyst proposals, filtered to valid, untested, in-budget configs."""
    proposals = analyst.propose_hypotheses(problem["search_space"], problem["objective"],
                                           format_table(history, problem["table_metrics"]),
                                           format_rules(rules), how_many)
    hypotheses = []
    for index, proposal in enumerate(proposals):
        name = None
        for candidate_name in candidates:
            if same_knobs(candidates[candidate_name], proposal["knobs"]):
                name = candidate_name
        if name is None:
            continue
        hypotheses.append({"id": "{}-{}".format(id_prefix, index + 1), "name": name,
                           "text": proposal["hypothesis"],
                           "predicted": float(proposal["predicted"]),
                           "confidence": float(proposal["confidence"])})
    return hypotheses


def same_knobs(knobs_a, knobs_b):
    """Compare as strings: the analyst may return 1024 as "1024"."""
    for knob in knobs_a:
        if knob not in knobs_b or str(knobs_a[knob]) != str(knobs_b[knob]):
            return False
    return True


def format_table(history, table_metrics):
    lines = []
    for entry in history:
        cells = []
        for metric in table_metrics:
            cells.append("{}={:.4f}".format(metric, entry["metrics"][metric]))
        lines.append(entry["name"] + " | " + " | ".join(cells))
    return "\n".join(lines)


def format_rules(rules):
    lines = []
    for rule in rules:
        if rule["status"] == "active":
            lines.append("{} [{}]: {} | condition {} | claim {}".format(
                rule["id"], playbook.rule_record(rule), rule["text"],
                json.dumps(rule["condition"]), json.dumps(rule["claim"])))
    if len(lines) == 0:
        return "(none yet)"
    return "\n".join(lines)


def to_probability(forecaster_id, predicted, threshold, surrogate, model, knobs, rules, hypotheses):
    """Turn a point forecast into P(objective >= threshold) for the bet."""
    if forecaster_id == "SURROGATE":
        return surrogate.probability_at_least(model, knobs, threshold)
    confidence = None
    for rule in rules:
        if rule["id"] == forecaster_id:
            confidence = forecast.rule_confidence(rule)
    for hypothesis in hypotheses:
        if hypothesis["id"] == forecaster_id:
            confidence = hypothesis["confidence"]
    if predicted >= threshold:
        return confidence
    return 1.0 - confidence


def check_event(event, metrics):
    """Settle objectively. Events look like 'ipc >= 0.35'."""
    metric_name, operator, threshold = event.split()
    if operator == ">=":
        return metrics[metric_name] >= float(threshold)
    raise ValueError("unsupported event: " + event)


def losing_rule(store, bet_id, rules, happened):
    for bet in store["bets"]:
        if bet["id"] == bet_id:
            leaned_yes = bet["probability"] >= 0.5
            if leaned_yes != happened:
                for rule in rules:
                    if rule["id"] == bet["forecaster"]:
                        return rule
    return None


def rescope(rule, problem, baseline_metrics, name, metrics):
    """Ask the analyst to sharpen the condition; keep the old one in the rule's trail."""
    context = "problem: {}\nbaseline metrics: {}\nexperiment {}: {}".format(
        problem["name"], json.dumps(baseline_metrics), name, json.dumps(metrics))
    updated = analyst.rescope_rule(rule, context)
    rule["origin"].append({"rescoped_from": rule["condition"], "because": context})
    rule["condition"] = updated["condition"]
    rule["text"] = updated["text"]
