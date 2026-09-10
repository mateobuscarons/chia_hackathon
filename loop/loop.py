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
  condition_metrics - metric names a rule condition may use

One round: fit surrogate -> pick designs (expected improvement, tilted by the
speaking rules; on a stall one claim test or scan, and the analyst's reply) ->
every forecaster answers three fixed questions per claim a chosen design carries
(sign, half the gain, the full gain) -> run -> settle, on the suite and on each
workload -> re-scope rules wrong on direction, re-bucket rules wrong on size.

How rules earn their say:
  - a rule's claimed gain is MEASURED from a controlled comparison before the
    rule enters the playbook (verify_claims); the LLM writes the condition and
    the words, never the number;
  - a rule forecasts relative to the candidate's sibling (same config, knob at
    baseline), so different candidates get different forecasts;
  - every bet a rule places counts toward its record; a rule that keeps losing
    on a problem is re-scoped so it stops firing there, and a rule below the
    credibility floor stays silent;
  - a bet's probability reflects how far the forecast sits from the dispute
    threshold, weighted by the forecaster's credibility.
"""

import json
import random

from loop import analyst, forecast, playbook
from loop.surrogate_gp import normal_tail

# A rule that has lost this many bets on the current problem, with a losing
# record there, is re-scoped so it no longer applies to this baseline.
RESCOPE_AFTER_LOSSES = 3
# A measured claim must move the objective by at least this much to be a rule.
MIN_CLAIM_GAIN_PCT = 1.0


def run_loop(problem, rounds, per_round, store, surrogate, use_rules, use_analyst, tag,
             prior_history=None, seed=0, explore=0.0, resume_history=None):
    """Returns {"history": [...], "rounds": [...]}; bets land in `store`.

    surrogate:   module with fit / predict_many / probability_at_least
    use_rules:   playbook rules shift the surrogate's mean and bet (the rules arm)
    use_analyst: the analyst gets a right of reply when the search stalls (the full arm)
    tag:         prefix for reply ids so arms can be told apart in the ledger
    prior_history: runs from OTHER chips the surrogate may learn from (pooled
                 transfer); they never count as simulations
    explore:     std bonus added to EI (bo_pooled_x); 0 is textbook EI
    resume_history: designs already measured on this problem by another selector
                 (the handoff arm's agent rounds); the loop continues from them

    Selection: expected improvement over a GP whose mean the speaking rules shift
    (bounded, decaying as real observations arrive). When the last round brought
    no improvement, one slot goes to the most valuable owed claim test (baseline +
    one knob) or, with none owed, to a stall scan of the incumbent; in the full arm
    the analyst then gets one design of its own. Textbook BO is the special case
    with no rules and no analyst: pure EI every round.
    """
    if prior_history is None:
        prior_history = []
    objective = problem["objective"]
    candidates = dict(problem["candidates"])
    baseline_name = name_of(problem, problem["baseline"])
    baseline_metrics = problem["evaluate"](problem["baseline"])
    reference = baseline_metrics[objective]
    # Conditions are checked on the workload descriptors (chip-independent); the
    # chip's own baseline run stays the reference for values and forecasts.
    descriptors = problem["descriptors"]
    history = [{"name": baseline_name, "knobs": problem["baseline"], "metrics": baseline_metrics,
                "reference": reference}]
    del candidates[baseline_name]
    if resume_history is not None:
        history = list(resume_history)
        for entry in history:
            if entry["name"] in candidates:
                del candidates[entry["name"]]

    all_rules = []
    if use_rules:
        all_rules = store["rules"]
    record_here = {}          # rule id -> {"sign": [won, lost], "size": [won, lost]} on THIS problem
    rescoped_here = set()
    analyst_record = {"wins": 0, "losses": 0}   # the analyst's reply bets on THIS problem
    honest_std = use_rules or use_analyst   # bets need honest doubt
    our_mechanism = use_rules or use_analyst   # baselines (bo, bo_pooled) stay textbook
    last_round_improved = True     # nothing can have stalled before round 1
    round_logs = []
    for round_number in range(1, rounds + 1):
        rules = forecast.speaking_rules(all_rules, descriptors)
        model = surrogate.fit(prior_history + history, problem["search_space"], objective,
                              honest_std=honest_std, reference=reference)
        shifts = None
        if use_rules:
            shifts = forecast.rule_shifts(rules, history, problem["baseline"], problem["search_space"], objective)
        best_so_far = max(entry["metrics"][objective] for entry in history)

        chosen = []
        slot_of = {}          # design name -> why it was picked outside EI
        replies = []
        stalled = not last_round_improved
        if stalled and our_mechanism:
            # One slot for the architect's move: settle an owed claim as a controlled
            # comparison, or, with nothing owed, try an untried neighbour of the incumbent.
            owed = forecast.untested_claim_tests(rules, history, problem["baseline"], candidates,
                                                problem["search_space"], objective)
            if len(owed) > 0:
                test_name, rule_id = owed[0]
                chosen.append(test_name)
                slot_of[test_name] = "claim test for " + rule_id
            else:
                scan_name = forecast.stall_scan_candidate(history, candidates, problem["search_space"],
                                                          surrogate, model, objective)
                if scan_name is not None:
                    chosen.append(scan_name)
                    slot_of[scan_name] = "stall scan"
            # Right of reply: the analyst names one design and bets on it.
            if use_analyst and len(chosen) < per_round:
                reply = ask_reply(problem, history, rules, candidates, "{}-REPLY-r{}".format(tag, round_number))
                if reply is not None and reply["name"] not in chosen:
                    replies.append(reply)
                    chosen.append(reply["name"])
                    slot_of[reply["name"]] = "reply " + reply["id"]
        more = forecast.pick_expected_improvement(
            candidates, surrogate, model, prior_history + history, problem["search_space"],
            objective, per_round - len(chosen), best_so_far, seed=seed * 1000 + round_number,
            shifts=shifts, baseline_knobs=problem["baseline"], already_chosen=chosen, honest_std=honest_std,
            explore=explore)
        chosen = chosen + more
        forecasts = None
        if len(rules) > 0 or len(replies) > 0:
            # Forecasts only for the chosen designs: that is all the bets need.
            chosen_candidates = {}
            for name in chosen:
                chosen_candidates[name] = candidates[name]
            forecasts = forecast.gather_forecasts(chosen_candidates, surrogate, model, rules, replies,
                                                  problem["baseline"], baseline_metrics, objective, history,
                                                  problem["search_space"])
        logged = None
        if forecasts is not None:
            logged = compact_forecasts(forecasts, chosen)
        round_logs.append({"round": round_number, "stalled": stalled, "forecasts": logged,
                           "replies": replies, "chosen": chosen, "slots": slot_of})

        # 1) every forecaster bets on every chosen design, BEFORE anything runs.
        # For each rule whose claim the design carries, three fixed questions
        # relative to the design's sibling: does the objective move the claimed way
        # (sign), by at least half the claimed gain (half), by the full gain (full)?
        # The rule, the surrogate and the analyst's reply each answer with their own
        # forecast; the simulator settles. With a measured sibling the same three
        # questions are also settled on every workload of the suite.
        bets_by_name = {}
        for name in chosen:
            bets_by_name[name] = []
            if forecasts is None:
                continue
            surrogate_mean = forecasts[name]["opinions"][0][1]
            surrogate_std = forecasts[name]["surrogate_std"]
            reply_for_design = None
            for reply in replies:
                if reply["name"] == name:
                    reply_for_design = reply
            for rule in rules:
                if rule["id"] not in forecasts[name]["siblings"]:
                    continue
                sibling = forecasts[name]["siblings"][rule["id"]]
                gain = forecast.effective_gain(rule, history, problem["baseline"], problem["search_space"], objective)
                rule_prediction = forecast.rule_forecast(sibling["value"], gain)
                for kind, threshold in claim_events(sibling["value"], gain):
                    event = event_text(objective, gain, threshold)
                    rule_probability = to_probability(rule["id"], rule_prediction, threshold, surrogate_std,
                                                      rules, replies, forecast.analyst_credibility(analyst_record))
                    surrogate_probability = to_probability("SURROGATE", surrogate_mean, threshold, surrogate_std,
                                                           rules, replies)
                    if gain < 0:
                        rule_probability = 1.0 - rule_probability
                        surrogate_probability = 1.0 - surrogate_probability
                    bets_by_name[name].append(playbook.place_bet(store, rule["id"], name, event, rule_probability, kind=kind))
                    bets_by_name[name].append(playbook.place_bet(store, "SURROGATE", name, event, surrogate_probability, kind=kind))
                    if reply_for_design is not None:
                        reply_probability = to_probability(reply_for_design["id"], reply_for_design["predicted"], threshold,
                                                           surrogate_std, rules, replies,
                                                           forecast.analyst_credibility(analyst_record))
                        if gain < 0:
                            reply_probability = 1.0 - reply_probability
                        bets_by_name[name].append(playbook.place_bet(store, reply_for_design["id"], name, event,
                                                                     reply_probability, kind=kind))
                # Per workload, only with a measured sibling: the rule states the same
                # probability for each workload (its claim is about the suite).
                if sibling["measured"]:
                    sibling_metrics = forecast.find_measured(history, sibling["knobs"])
                    for key in sibling_metrics:
                        if not key.endswith(":" + objective):
                            continue
                        workload_sibling = sibling_metrics[key]
                        workload_prediction = forecast.rule_forecast(workload_sibling, gain)
                        for kind, threshold in claim_events(workload_sibling, gain):
                            event = event_text(key, gain, threshold)
                            probability = to_probability(rule["id"], workload_prediction, threshold, surrogate_std,
                                                         rules, replies)
                            if gain < 0:
                                probability = 1.0 - probability
                            bets_by_name[name].append(playbook.place_bet(store, rule["id"], name, event, probability,
                                                                         kind=kind, level="workload"))
            # The reply also bets on its own forecast: within 5%, like the plain agent.
            if reply_for_design is not None:
                event = "{} >= {:.4f}".format(objective, 0.95 * reply_for_design["predicted"])
                probability = (1.0 + reply_for_design["confidence"]) / 2.0
                bets_by_name[name].append(playbook.place_bet(store, reply_for_design["id"], name, event, probability,
                                                             kind="forecast"))

        # 2) the chosen configs run in parallel (one CHIA task each)
        knobs_list = [candidates[name] for name in chosen]
        metrics_list = problem["evaluate_many"](knobs_list)

        # 3) settle; a rule that keeps losing here is re-scoped (sign) or re-bucketed (size)
        last_round_improved = False
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            if metrics[objective] > best_so_far:
                last_round_improved = True
            history.append({"name": name, "knobs": knobs, "metrics": metrics, "reference": reference})
            del candidates[name]
            for bet_id in bets_by_name[name]:
                bet = find_bet(store, bet_id)
                happened = check_event(bet["event"], metrics)
                playbook.settle_bet(store, bet_id, happened)
                leaned_yes = bet["probability"] >= 0.5
                won = leaned_yes == happened
                if "-REPLY-" in bet["forecaster"] and bet["kind"] == "forecast":
                    if won:
                        analyst_record["wins"] += 1
                    else:
                        analyst_record["losses"] += 1
                rule = rule_by_id(rules, bet["forecaster"])
                if rule is None:
                    continue
                record = record_here.setdefault(rule["id"], {"sign": [0, 0], "size": [0, 0]})
                slot = "size"
                if bet["kind"] == "sign":
                    slot = "sign"
                if won:
                    record[slot][0] += 1
                else:
                    record[slot][1] += 1
            label = "{} bets".format(len(bets_by_name[name]))
            if name in slot_of:
                label = label + " | " + slot_of[name]
            print("[{}] round {} | {} | {}={:.4f} | {}".format(
                tag, round_number, name, objective, metrics[objective], label), flush=True)

        for rule in rules:
            record = record_here.get(rule["id"])
            if record is None:
                continue
            sign_won, sign_lost = record["sign"]
            size_won, size_lost = record["size"]
            if sign_lost >= RESCOPE_AFTER_LOSSES and sign_lost > sign_won and rule["id"] not in rescoped_here:
                # Wrong direction here: sharpen the condition so the rule stops firing on cases like this.
                rescope(rule, problem, descriptors, history[-1]["name"], history[-1]["metrics"],
                        wins=sign_won, losses=sign_lost, ask_analyst=use_analyst)
                rescoped_here.add(rule["id"])
                print("[{}] re-scoped {} after sign {}-{} here: {}".format(
                    tag, rule["id"], sign_won, sign_lost, json.dumps(rule["conditions"])), flush=True)
            elif size_lost >= RESCOPE_AFTER_LOSSES and size_lost > size_won:
                # Right direction, wrong size: shrink the claim and start counting again.
                rebucket(rule, problem["name"], size_won, size_lost)
                record["size"] = [0, 0]
                print("[{}] re-bucketed {} to {:+.1f}% after size {}-{} here".format(
                    tag, rule["id"], rule["claim"]["gain_pct"], size_won, size_lost), flush=True)
    return {"history": history, "rounds": round_logs}


def run_llm_direct(problem, rounds, per_round, store, tag, seed=0):
    """The plain agent (AgentDSE's setting): each round the LLM picks the designs
    from the results table and the workload descriptors alone, and bets that each
    lands within 5% of its own forecast. A proposal that is malformed, out of
    budget or already measured is dropped and its slot filled with a random
    untested design, which the round log records. Same history shape as run_loop."""
    objective = problem["objective"]
    candidates = dict(problem["candidates"])
    baseline_name = name_of(problem, problem["baseline"])
    baseline_metrics = problem["evaluate"](problem["baseline"])
    history = [{"name": baseline_name, "knobs": problem["baseline"], "metrics": baseline_metrics,
                "reference": baseline_metrics[objective]}]
    del candidates[baseline_name]
    descriptors_text = json.dumps(problem["descriptors"], indent=1)
    filler = random.Random(seed)
    round_logs = []
    for round_number in range(1, rounds + 1):
        proposals = analyst.pick_designs(problem["search_space"], objective,
                                         format_table(history, problem["table_metrics"]),
                                         descriptors_text, per_round, problem["area_budget_kb"])
        chosen = []
        forecasts_of = {}
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
                    forecasts_of[candidate_name] = {"id": "{}-LLM-r{}-{}".format(tag, round_number, index + 1),
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
        round_logs.append({"round": round_number, "chosen": chosen, "proposals": proposals, "random_fill": filled})

        # The bet: the measured objective lands at or above 95% of the forecast. With
        # confidence c of landing within 5%, and the rest split evenly, P = (1 + c) / 2.
        bets_by_name = {}
        for name in chosen:
            bets_by_name[name] = []
            if name in forecasts_of:
                pick = forecasts_of[name]
                event = "{} >= {:.4f}".format(objective, 0.95 * pick["predicted"])
                probability = (1.0 + pick["confidence"]) / 2.0
                bets_by_name[name].append(playbook.place_bet(store, pick["id"], name, event, probability,
                                                             kind="forecast"))
        knobs_list = [candidates[name] for name in chosen]
        metrics_list = problem["evaluate_many"](knobs_list)
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            history.append({"name": name, "knobs": knobs, "metrics": metrics,
                            "reference": baseline_metrics[objective]})
            del candidates[name]
            for bet_id in bets_by_name[name]:
                bet = find_bet(store, bet_id)
                playbook.settle_bet(store, bet_id, check_event(bet["event"], metrics))
            label = "llm pick"
            if name not in forecasts_of:
                label = "random fill"
            print("[{}] round {} | {} | {}={:.4f} | {}".format(
                tag, round_number, name, objective, metrics[objective], label), flush=True)
    return {"history": history, "rounds": round_logs}


def valid_pick(proposal):
    """A pick must carry knobs, a numeric forecast and a confidence in [0, 1]."""
    try:
        float(proposal["predicted"])
        confidence = float(proposal["confidence"])
        return isinstance(proposal.get("knobs"), dict) and 0.0 <= confidence <= 1.0
    except (KeyError, TypeError, ValueError):
        return False


def compact_forecasts(forecasts, chosen):
    """Only the chosen configs' forecasts are kept in the log (the full table is
    tens of thousands of rows per round)."""
    kept = {}
    for name in chosen:
        kept[name] = forecasts[name]
    return kept


def rule_by_id(rules, rule_id):
    for rule in rules:
        if rule["id"] == rule_id:
            return rule
    return None


def name_of(problem, knobs):
    for name in problem["candidates"]:
        if problem["candidates"][name] == knobs:
            return name
    raise KeyError("knobs not in candidates: " + json.dumps(knobs))


def ask_reply(problem, history, rules, candidates, reply_id):
    """The analyst's one design on a stall, or None when its answer is malformed,
    already measured or out of budget (LLM output is untrusted)."""
    incumbent = history[0]
    for entry in history:
        if entry["metrics"][problem["objective"]] > incumbent["metrics"][problem["objective"]]:
            incumbent = entry
    proposals = analyst.propose_reply(problem["search_space"], problem["objective"],
                                      format_table(history, problem["table_metrics"]),
                                      format_rules(rules), incumbent["name"])
    for proposal in proposals:
        if not valid_hypothesis(proposal) or not isinstance(proposal.get("knobs"), dict):
            continue
        for candidate_name in candidates:
            if forecast.same_knobs(candidates[candidate_name], proposal["knobs"]):
                return {"id": reply_id, "name": candidate_name, "text": proposal["hypothesis"],
                        "predicted": float(proposal["predicted"]),
                        "confidence": float(proposal["confidence"])}
    return None


def rule_clauses_of(rule):
    """The analyst may answer with "condition" (one clause) or "conditions" (a list)."""
    if isinstance(rule.get("conditions"), list) and len(rule["conditions"]) > 0:
        return rule["conditions"]
    return [rule["condition"]]


def valid_rule(rule, problem):
    """LLM output is untrusted: a rule may only name real metrics, knobs and values."""
    try:
        clauses = rule_clauses_of(rule)
        if len(clauses) > 2:
            return False
        for clause in clauses:
            if clause["metric"] not in problem["condition_metrics"]:
                return False
            if clause["op"] not in [">=", ">", "<", "<=", "=="]:
                return False
            float(clause["value"])
        claim = rule["claim"]
        values = problem["search_space"][claim["knob"]]
        if forecast.is_direction(claim):
            # Directions are only meaningful on ordered (numeric) knobs.
            for value in values:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    return False
        else:
            allowed = [str(value) for value in values]
            if str(claim["value"]) not in allowed:
                return False
            # A claim about the baseline's own value says nothing and always "wins".
            if str(claim["value"]) == str(problem["baseline"][claim["knob"]]):
                return False
        float(claim["gain_pct"])
        if "direction" in claim and claim["direction"] not in ["helps", "hurts"]:
            return False
        return isinstance(rule["text"], str)
    except (KeyError, TypeError, ValueError):
        return False


def valid_hypothesis(proposal):
    """A proposal must carry a numeric forecast and a confidence in [0, 1]."""
    try:
        float(proposal["predicted"])
        confidence = float(proposal["confidence"])
        return 0.0 <= confidence <= 1.0 and isinstance(proposal["hypothesis"], str)
    except (KeyError, TypeError, ValueError):
        return False


def same_knobs(knobs_a, knobs_b):
    return forecast.same_knobs(knobs_a, knobs_b)


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
            lines.append("{} [{}]: {} | conditions {} | claim {}".format(
                rule["id"], playbook.rule_record(rule), rule["text"],
                json.dumps(forecast.rule_clauses(rule)), json.dumps(rule["claim"])))
    if len(lines) == 0:
        return "(none yet)"
    return "\n".join(lines)


def to_probability(forecaster_id, predicted, threshold, surrogate_std, rules, replies,
                   analyst_credibility=0.5):
    """P(objective >= threshold) for one forecaster's own forecast.

    The surrogate's probability is its own posterior. A rule or a reply gives a
    point forecast: a normal is placed around it with the surrogate's std (the
    forecaster is at least as unsure about the rest of the design as the
    surrogate is), then shrunk toward 0.5 by how little the forecaster is trusted.
    A rule at 0.7 credibility forecasting far above the threshold bets ~0.85;
    one forecasting just above it bets ~0.55.
    """
    if forecaster_id == "SURROGATE":
        return normal_tail(predicted, surrogate_std, threshold)
    credibility = None
    for rule in rules:
        if rule["id"] == forecaster_id:
            credibility = forecast.rule_confidence(rule)
    for reply in replies:
        if reply["id"] == forecaster_id:
            credibility = reply["confidence"] * analyst_credibility
    sharp = normal_tail(predicted, surrogate_std, threshold)
    return credibility * sharp + (1.0 - credibility) * 0.5


# A claimed gain of g% on a sibling worth s settles three questions.
SIGN_MARGIN = 0.001   # "moved the claimed way": beyond a 0.1% tie band


def claim_events(sibling_value, gain_pct):
    """[(kind, threshold)] for a claim: sign, half the gain, the full gain."""
    direction = 1.0
    if gain_pct < 0:
        direction = -1.0
    return [("sign", sibling_value * (1.0 + direction * SIGN_MARGIN)),
            ("half", sibling_value * (1.0 + gain_pct / 200.0)),
            ("full", sibling_value * (1.0 + gain_pct / 100.0))]


def event_text(metric_name, gain_pct, threshold):
    """A helping claim is settled with '>=', a hurting one with '<='."""
    if gain_pct < 0:
        return "{} <= {:.4f}".format(metric_name, threshold)
    return "{} >= {:.4f}".format(metric_name, threshold)


def check_event(event, metrics):
    """Settle objectively. Events look like 'ipc >= 0.35' or 'bfs.urand:ipc <= 0.41'."""
    metric_name, operator, threshold = event.split()
    if operator == ">=":
        return metrics[metric_name] >= float(threshold)
    if operator == "<=":
        return metrics[metric_name] <= float(threshold)
    raise ValueError("unsupported event: " + event)


def rebucket(rule, problem_name, size_won, size_lost):
    """The rule's direction held but its size did not: halve the claimed gain and
    keep the old one in the rule's trail."""
    old_gain = rule["claim"]["gain_pct"]
    new_gain = old_gain / 2.0
    # Never below the effect a rule needs to exist at all: a claim that small is
    # the sign alone, and the sign is what the direction bets already score.
    if abs(new_gain) < MIN_CLAIM_GAIN_PCT:
        if old_gain < 0:
            new_gain = -MIN_CLAIM_GAIN_PCT
        else:
            new_gain = MIN_CLAIM_GAIN_PCT
    if new_gain == old_gain:
        return
    rule["claim"]["gain_pct"] = new_gain
    rule["origin"].append({"rebucketed_from": old_gain, "to": new_gain,
                           "because": "size bets {}-{} on {}".format(size_won, size_lost, problem_name)})


def find_bet(store, bet_id):
    for bet in store["bets"]:
        if bet["id"] == bet_id:
            return bet
    raise KeyError(bet_id)


def rescope(rule, problem, descriptors, name, metrics, wins=0, losses=0, ask_analyst=False):
    """Sharpen a losing rule's condition so it stops firing on cases like this one;
    the old condition stays in the rule's trail. With ask_analyst the LLM proposes
    the new condition (the full arm); otherwise, and whenever the LLM's answer is
    unusable, the first clause is tightened mechanically just past the failing
    value (the rules arm runs with no LLM at test time).
    `descriptors`: the workload descriptors the conditions are checked against."""
    context = "problem: {}\nworkload descriptors (conditions are checked on these): {}\nrecord on this problem: {} wins, {} losses\nlast experiment {}: {}".format(
        problem["name"], json.dumps(descriptors), wins, losses, name, json.dumps(metrics))
    updated = {}
    if ask_analyst:
        updated = analyst.rescope_rule(rule, context, problem["condition_metrics"])
    candidate = dict(rule)
    if isinstance(updated.get("conditions"), list):
        candidate["conditions"] = updated["conditions"]
        candidate["condition"] = updated["conditions"][0] if len(updated["conditions"]) > 0 else None
    else:
        candidate["condition"] = updated.get("condition")
        candidate["conditions"] = [updated.get("condition")]
    candidate["text"] = updated.get("text")
    if (candidate["condition"] is not None and valid_rule(candidate, problem)
            and not forecast.rule_applies(candidate, descriptors)):
        new_clauses = []
        for clause in rule_clauses_of(candidate):
            clean = dict(clause)
            clean["value"] = float(clean["value"])
            new_clauses.append(clean)
        new_text = candidate["text"]
        how = "analyst"
    else:
        # The analyst's re-scope was invalid or still covers the failing case:
        # tighten the first clause mechanically so this baseline no longer qualifies.
        old_clauses = forecast.rule_clauses(rule)
        new_clauses = [tightened_condition(old_clauses[0], descriptors)] + old_clauses[1:]
        new_text = rule["text"] + " [narrowed: {} {} {:.3g}]".format(
            new_clauses[0]["metric"], new_clauses[0]["op"], new_clauses[0]["value"])
        how = "mechanical"
    rule["origin"].append({"rescoped_from": forecast.rule_clauses(rule), "because": context, "how": how})
    rule["conditions"] = new_clauses
    rule["condition"] = new_clauses[0]
    rule["text"] = new_text


def tightened_condition(condition, baseline_metrics):
    """Move the threshold just past the failing case's value, keeping the direction."""
    failing_value = baseline_metrics[condition["metric"]]
    tightened = dict(condition)
    if condition["op"] in [">=", ">"]:
        tightened["op"] = ">"
        tightened["value"] = failing_value
    elif condition["op"] in ["<", "<="]:
        tightened["op"] = "<"
        tightened["value"] = failing_value
    else:
        tightened["op"] = ">"
        tightened["value"] = failing_value
    return tightened


SIZE_KNOBS = ["l1d_sets", "l1d_ways", "l2_sets", "l2_ways", "llc_sets", "llc_ways"]


def verify_claims(store, proposals, problems, tag, max_sims_per_problem):
    """Admit LLM-proposed rules only with a MEASURED claim, pooled over every
    training problem (one per chip, each with its "history" of measured designs).

    For each proposal: collect every controlled pair for its claim on every
    problem; where a problem has none, run the one missing experiment there (the
    baseline moved one claim-step), at most `max_sims_per_problem` per problem.
    The rule enters with gain_pct = mean measured gain over all pairs of all
    chips; it is rejected when that gain is below MIN_CLAIM_GAIN_PCT in
    magnitude, points the other way from what the LLM claimed, duplicates a
    claim already in the playbook, or has a condition that contradicts its own
    mechanism (a size claim conditioned on the wrong side of movable MPKI). A
    fitted condition that leaves every training point on one side is kept but
    flagged "always". Returns {problem name: extra runs}.
    """
    objective = problems[0]["objective"]
    space = problems[0]["search_space"]
    extra_runs = {}
    sims_used = {}
    for problem in problems:
        extra_runs[problem["name"]] = []
        sims_used[problem["name"]] = 0
    for proposal in proposals:
        if not valid_rule(proposal, problems[0]):
            playbook.reject_rule(store, proposal, "malformed")
            print("[{}] rejected malformed rule: {}".format(tag, json.dumps(proposal)[:160]), flush=True)
            continue
        claim = dict(proposal["claim"])
        if not forecast.is_direction(claim):
            claim["value"] = matching_value(space[claim["knob"]], claim["value"])
        if claim_already_ruled(store, claim):
            playbook.reject_rule(store, proposal, "duplicate claim")
            print("[{}] rejected (duplicate claim {}={}): {}".format(
                tag, claim["knob"], claim["value"], proposal["text"][:120]), flush=True)
            continue
        contradiction = mechanism_contradiction(rule_clauses_of(proposal), claim)
        if contradiction is not None:
            playbook.reject_rule(store, proposal, contradiction)
            print("[{}] rejected ({}): {}".format(tag, contradiction, proposal["text"][:120]), flush=True)
            continue
        all_gains = []
        gains_by_point = {}
        descriptors_by_point = {}
        for problem in problems:
            evidence = problem["history"] + extra_runs[problem["name"]]
            gains = forecast.paired_gains(evidence, claim, problem["baseline"], space, objective)
            if len(gains) == 0 and sims_used[problem["name"]] < max_sims_per_problem:
                test_knobs = forecast.claim_target(claim, problem["baseline"], space)
                test_name = None
                if test_knobs is not None:
                    test_name = candidate_name(problem, test_knobs)
                if test_name is not None:
                    metrics = problem["evaluate"](test_knobs)
                    sims_used[problem["name"]] += 1
                    extra_runs[problem["name"]].append({"name": test_name, "knobs": test_knobs, "metrics": metrics})
                    evidence = problem["history"] + extra_runs[problem["name"]]
                    gains = forecast.paired_gains(evidence, claim, problem["baseline"], space, objective)
            all_gains = all_gains + gains
            per_workload = per_workload_gains(claim, evidence, problem["baseline"], space, objective)
            for workload in per_workload:
                point = problem["soc_name"] + "/" + workload
                gains_by_point[point] = per_workload[workload]
                descriptors_by_point[point] = problem["workload_descriptors"].get(workload)
        if len(all_gains) == 0:
            playbook.reject_rule(store, proposal, "no controlled comparison available")
            print("[{}] rejected (untestable) rule: {}".format(tag, proposal["text"][:120]), flush=True)
            continue
        measured = sum(all_gains) / len(all_gains)
        llm_gain = llm_signed_gain(claim)
        if abs(measured) < MIN_CLAIM_GAIN_PCT:
            playbook.reject_rule(store, proposal, "measured gain {:.2f}% below {}%".format(measured, MIN_CLAIM_GAIN_PCT))
            print("[{}] rejected (no effect, {:.2f}%): {}".format(tag, measured, proposal["text"][:120]), flush=True)
            continue
        if (measured > 0) != (llm_gain > 0):
            playbook.reject_rule(store, proposal, "measured {:.2f}% but LLM claimed {:.2f}%".format(measured, llm_gain))
            print("[{}] rejected (wrong sign, {:.2f}% vs {:.2f}%): {}".format(
                tag, measured, llm_gain, proposal["text"][:120]), flush=True)
            continue
        verified_claim = dict(claim)
        verified_claim["gain_pct"] = measured
        chips = []
        for problem in problems:
            chips.append(problem["soc_name"])
        example = "{} {} = {:+.2f}% over {} controlled pair(s) on {}".format(
            claim["knob"], claim["value"], measured, len(all_gains), ", ".join(chips))
        # The LLM chose WHICH descriptor matters; the data sets WHERE the threshold
        # lies: between the (chip, workload) points where the effect appeared and
        # where it did not.
        clauses, fit_note = fit_thresholds(rule_clauses_of(proposal), claim, gains_by_point, descriptors_by_point)
        always = fit_note.get("degenerate", False)
        rule_id = playbook.add_rule(store, clauses[0], verified_claim, example,
                                    proposal["text"], conditions=clauses,
                                    verification={"pairs": len(all_gains), "llm_gain_pct": llm_gain,
                                                  "threshold_fit": fit_note, "always": always})
        flag = ""
        if always:
            flag = " [ALWAYS: no training point on the other side of the threshold]"
        print("[{}] admitted {} ({:+.2f}% measured over {} pairs, LLM said {:+.1f}%): {} | conditions {}{}".format(
            tag, rule_id, measured, len(all_gains), llm_gain, proposal["text"][:100], json.dumps(clauses), flag), flush=True)
    return extra_runs


def mechanism_contradiction(clauses, claim):
    """A size claim's condition must read movable MPKI the way its mechanism does:
    "bigger helps" needs misses a bigger cache can remove (movable >= x), "smaller
    is fine" needs few of them (movable < x). The reverse is the inverted-sign
    condition a chip with almost no capacity headroom produced. None when fine."""
    if claim["knob"] not in SIZE_KNOBS or not forecast.is_direction(claim):
        return None
    for clause in clauses:
        if not str(clause["metric"]).startswith("movable"):
            continue
        greater = clause["op"] in [">=", ">"]
        if str(claim["value"]) == "up" and not greater:
            return "condition contradicts the claim's mechanism (bigger cache, fewer movable misses)"
        if str(claim["value"]) == "down" and greater:
            return "condition contradicts the claim's mechanism (smaller cache, more movable misses)"
    return None


def per_workload_gains(claim, evidence, baseline_knobs, search_space, objective):
    """Mean controlled-pair gain of a claim on EACH workload of a suite, from the
    per-workload objective kept in every suite result ("<workload>:<objective>").
    Empty for single-workload problems."""
    sums = {}
    counts = {}
    for entry in evidence:
        sibling = forecast.claim_sibling(claim, entry["knobs"], baseline_knobs, search_space)
        if sibling is None:
            continue
        sibling_metrics = forecast.find_measured(evidence, sibling)
        if sibling_metrics is None:
            continue
        for key in entry["metrics"]:
            if not key.endswith(":" + objective):
                continue
            workload = key.split(":")[0]
            if key not in sibling_metrics or sibling_metrics[key] <= 0:
                continue
            gain = 100.0 * (entry["metrics"][key] / sibling_metrics[key] - 1.0)
            sums[workload] = sums.get(workload, 0.0) + gain
            counts[workload] = counts.get(workload, 0) + 1
    means = {}
    for workload in sums:
        means[workload] = sums[workload] / counts[workload]
    return means


def fit_thresholds(clauses, claim, gains_by_point, descriptors_by_point):
    """Re-set each clause's threshold from evidence. A (chip, workload) point
    shows the effect when its controlled gain has the claim's sign and at least
    MIN_CLAIM_GAIN_PCT. For ">=" the threshold moves to the midpoint between the
    lowest descriptor value with the effect and the highest without it below
    that (geometric mean when both are positive); with no point below, it drops
    to the lowest value with the effect so the rule speaks wherever the effect
    was seen. "<" is the mirror. When no clause separates a present point from
    an absent one, the condition is degenerate: every training point sits on
    one side. Returns (clauses, note)."""
    expected_sign = 1.0
    if llm_signed_gain(claim) < 0:
        expected_sign = -1.0
    present = []
    absent = []
    for point, gain in gains_by_point.items():
        if descriptors_by_point.get(point) is None:
            continue
        if gain * expected_sign >= MIN_CLAIM_GAIN_PCT:
            present.append(point)
        else:
            absent.append(point)
    note = {"present": present, "absent": absent, "moved": [], "degenerate": False}
    if len(present) == 0:
        return clauses, note
    fitted = []
    separating_clauses = 0
    for clause in clauses:
        metric = clause["metric"]
        base_metric = metric
        if base_metric.startswith("max_"):
            base_metric = base_metric[len("max_"):]
        present_values = []
        for point in present:
            if base_metric in descriptors_by_point[point]:
                present_values.append(float(descriptors_by_point[point][base_metric]))
        absent_values = []
        for point in absent:
            if base_metric in descriptors_by_point[point]:
                absent_values.append(float(descriptors_by_point[point][base_metric]))
        new_clause = dict(clause)
        old_value = float(clause["value"])
        if len(present_values) == 0:
            fitted.append(new_clause)
            continue
        if clause["op"] in [">=", ">"]:
            edge = min(present_values)
            below = []
            for value in absent_values:
                if value < edge:
                    below.append(value)
            if len(below) > 0:
                new_clause["value"] = midpoint(max(below), edge)
                separating_clauses += 1
            else:
                new_clause["value"] = edge
            new_clause["op"] = ">="
        else:
            edge = max(present_values)
            above = []
            for value in absent_values:
                if value > edge:
                    above.append(value)
            if len(above) > 0:
                new_clause["value"] = midpoint(edge, min(above))
                separating_clauses += 1
            else:
                new_clause["value"] = edge * 1.001 + 1e-9
            new_clause["op"] = "<"
        if abs(new_clause["value"] - old_value) > 1e-9:
            note["moved"].append({"metric": metric, "from": old_value, "to": new_clause["value"]})
        fitted.append(new_clause)
    if separating_clauses == 0:
        note["degenerate"] = True
    return fitted, note


def midpoint(low, high):
    """Geometric midpoint when both ends are positive (ratios, fractions), else arithmetic."""
    if low > 0 and high > 0:
        return (low * high) ** 0.5
    return (low + high) / 2.0


def claim_already_ruled(store, claim):
    """True if an active rule in `store` already carries this knob and value."""
    for rule in store["rules"]:
        if rule.get("status", "active") != "active":
            continue
        same_knob = rule["claim"]["knob"] == claim["knob"]
        same_value = str(rule["claim"]["value"]) == str(claim["value"])
        if same_knob and same_value:
            return True
    return False


def llm_signed_gain(claim):
    """The LLM's claimed effect with its sign taken from "direction" (helps/hurts).
    Older claims without a direction carry the sign in gain_pct itself."""
    magnitude = abs(float(claim["gain_pct"]))
    if claim.get("direction") == "hurts":
        return -magnitude
    if claim.get("direction") == "helps":
        return magnitude
    return float(claim["gain_pct"])


def matching_value(allowed_values, value):
    """The search-space value equal to the LLM's (possibly string) value, with its real type."""
    for allowed in allowed_values:
        if str(allowed) == str(value):
            return allowed
    return value


def candidate_name(problem, knobs):
    for name in problem["candidates"]:
        if forecast.same_knobs(problem["candidates"][name], knobs):
            return name
    return None
