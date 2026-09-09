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
every forecaster bets on the chosen designs -> run -> settle -> re-scope losing
rules. Every forecast on a chosen design is logged for calibration plots.

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
             prior_history=None, seed=0):
    """Returns {"history": [...], "rounds": [...]}; bets land in `store`.

    surrogate:   module with fit / predict_many / probability_at_least
    use_rules:   playbook rules shift the surrogate's mean and bet (the rules arm)
    use_analyst: the analyst gets a right of reply when the search stalls (the full arm)
    tag:         prefix for reply ids so arms can be told apart in the ledger
    prior_history: runs from OTHER chips the surrogate may learn from (pooled
                 transfer); they never count as simulations

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

    all_rules = []
    if use_rules:
        all_rules = store["rules"]
    losses_here = {}          # rule id -> bets lost on THIS problem
    wins_here = {}
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
            shifts=shifts, baseline_knobs=problem["baseline"], already_chosen=chosen, honest_std=honest_std)
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

        # 1) every forecaster bets on every chosen config, BEFORE anything runs
        bets_by_name = {}
        for name in chosen:
            bets_by_name[name] = []
            if forecasts is None:
                continue
            opinions = forecasts[name]["opinions"]
            surrogate_std = forecasts[name]["surrogate_std"]
            if len(opinions) > 1:
                # Settle at the midpoint of the dispute: exactly where they disagree.
                values = [predicted for _, predicted in opinions]
                threshold = (max(values) + min(values)) / 2.0
                event = "{} >= {:.4f}".format(objective, threshold)
                for forecaster_id, predicted in opinions:
                    probability = to_probability(forecaster_id, predicted, threshold,
                                                 surrogate_std, rules, replies,
                                                 forecast.analyst_credibility(analyst_record))
                    bets_by_name[name].append(
                        playbook.place_bet(store, forecaster_id, name, event, probability))
            # A rule also bets on its OWN claim, but only as a controlled comparison:
            # the history must hold the sibling (same config, knob at baseline), so
            # the gain can be attributed to that knob alone.
            for rule in rules:
                if rule["id"] not in forecasts[name]["claim_tests"]:
                    continue
                claim = rule["claim"]
                sibling = forecast.claim_sibling(claim, candidates[name], problem["baseline"],
                                                 problem["search_space"])
                sibling_metrics = forecast.find_measured(history, sibling)
                gain = forecast.effective_gain(rule, history, problem["baseline"], problem["search_space"], objective)
                half_gain = sibling_metrics[objective] * (1.0 + gain / 200.0)
                claim_event = "{} >= {:.4f}".format(objective, half_gain)
                # A rule claiming a LOSS bets that the objective stays BELOW the
                # half-way mark: same event, probability on the other side.
                probability = forecast.rule_confidence(rule)
                if gain < 0:
                    probability = 1.0 - probability
                bets_by_name[name].append(playbook.place_bet(
                    store, rule["id"], name, claim_event, probability, kind="claim"))

        # 2) the chosen configs run in parallel (one CHIA task each)
        knobs_list = [candidates[name] for name in chosen]
        metrics_list = problem["evaluate_many"](knobs_list)

        # 3) settle; rules that keep losing here get re-scoped
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
                if "-REPLY-" in bet["forecaster"]:
                    if leaned_yes == happened:
                        analyst_record["wins"] += 1
                    else:
                        analyst_record["losses"] += 1
                rule = rule_by_id(rules, bet["forecaster"])
                if rule is None:
                    continue
                if leaned_yes == happened:
                    wins_here[rule["id"]] = wins_here.get(rule["id"], 0) + 1
                else:
                    losses_here[rule["id"]] = losses_here.get(rule["id"], 0) + 1
            label = "{} bets".format(len(bets_by_name[name]))
            if name in slot_of:
                label = label + " | " + slot_of[name]
            print("[{}] round {} | {} | {}={:.4f} | {}".format(
                tag, round_number, name, objective, metrics[objective], label), flush=True)

        for rule in rules:
            if rule["id"] in rescoped_here:
                continue
            lost = losses_here.get(rule["id"], 0)
            won = wins_here.get(rule["id"], 0)
            lost_claim = rule.get("claim_losses", 0) > 0 and lost > 0
            if (lost >= RESCOPE_AFTER_LOSSES and lost > won) or lost_claim:
                rescope(rule, problem, descriptors, history[-1]["name"], history[-1]["metrics"],
                        wins=won, losses=lost, ask_analyst=use_analyst)
                rescoped_here.add(rule["id"])
                print("[{}] re-scoped {} after {}-{} here: {}".format(
                    tag, rule["id"], won, lost, json.dumps(rule["conditions"])), flush=True)
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
    """P(objective >= threshold) for one forecaster's bet.

    The surrogate's probability is its own posterior. A rule or hypothesis gives
    a point forecast: we place a normal around it with the surrogate's std (the
    forecaster is at least as unsure about the rest of the config as the
    surrogate is), then shrink toward 0.5 by how little we trust the forecaster.
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


def check_event(event, metrics):
    """Settle objectively. Events look like 'ipc >= 0.35'."""
    metric_name, operator, threshold = event.split()
    if operator == ">=":
        return metrics[metric_name] >= float(threshold)
    raise ValueError("unsupported event: " + event)


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


def verify_claims(store, proposals, problem, history, tag, max_sims, evidence=None):
    """Admit LLM-proposed rules only with a MEASURED claim.

    For each proposal: collect every controlled pair for its claim in the
    evidence (all measured designs of this problem, defaults to `history`). If
    there is none, run the one missing experiment (the baseline moved one
    claim-step) - at most `max_sims` such runs. The rule enters with gain_pct =
    mean measured gain; it is rejected when the measured gain is below
    MIN_CLAIM_GAIN_PCT in magnitude or points the other way from what the LLM
    claimed. Returns the extra runs so the caller can count them.
    """
    objective = problem["objective"]
    baseline_knobs = problem["baseline"]
    space = problem["search_space"]
    if evidence is None:
        evidence = history
    extra_runs = []
    sims_used = 0
    for proposal in proposals:
        if not valid_rule(proposal, problem):
            playbook.reject_rule(store, proposal, "malformed")
            print("[{}] rejected malformed rule: {}".format(tag, json.dumps(proposal)[:160]), flush=True)
            continue
        claim = dict(proposal["claim"])
        if not forecast.is_direction(claim):
            claim["value"] = matching_value(space[claim["knob"]], claim["value"])
        # One claim, one rule: the same knob+value already in the playbook (from an
        # earlier batch or earlier in this one) is a duplicate, however it is worded.
        if claim_already_ruled(store, claim):
            playbook.reject_rule(store, proposal, "duplicate claim")
            print("[{}] rejected (duplicate claim {}={}): {}".format(
                tag, claim["knob"], claim["value"], proposal["text"][:120]), flush=True)
            continue
        gains = forecast.paired_gains(evidence + extra_runs, claim, baseline_knobs, space, objective)
        if len(gains) == 0 and sims_used < max_sims:
            test_knobs = forecast.claim_target(claim, baseline_knobs, space)
            test_name = None
            if test_knobs is not None:
                test_name = candidate_name(problem, test_knobs)
            if test_name is not None:
                metrics = problem["evaluate"](test_knobs)
                sims_used += 1
                extra_runs.append({"name": test_name, "knobs": test_knobs, "metrics": metrics})
                gains = forecast.paired_gains(evidence + extra_runs, claim, baseline_knobs, space, objective)
        if len(gains) == 0:
            playbook.reject_rule(store, proposal, "no controlled comparison available")
            print("[{}] rejected (untestable) rule: {}".format(tag, proposal["text"][:120]), flush=True)
            continue
        measured = sum(gains) / len(gains)
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
        example = "{}: {} {} = {:+.2f}% over {} controlled pair(s)".format(
            problem["name"], claim["knob"], claim["value"], measured, len(gains))
        # The LLM chose WHICH descriptor matters; the data sets WHERE the threshold
        # lies: between the workloads where the effect appeared and where it did not.
        clauses, fit_note = fit_thresholds(rule_clauses_of(proposal), claim, evidence + extra_runs,
                                           problem, baseline_knobs, space, objective)
        rule_id = playbook.add_rule(store, clauses[0], verified_claim, example,
                                    proposal["text"], conditions=clauses,
                                    verification={"pairs": len(gains), "llm_gain_pct": llm_gain,
                                                  "problem": problem["name"], "threshold_fit": fit_note})
        print("[{}] admitted {} ({:+.2f}% measured over {} pairs, LLM said {:+.1f}%): {} | conditions {}".format(
            tag, rule_id, measured, len(gains), llm_gain, proposal["text"][:100], json.dumps(clauses)), flush=True)
    return extra_runs


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


def fit_thresholds(clauses, claim, evidence, problem, baseline_knobs, search_space, objective):
    """Re-set each clause's threshold from evidence. A workload shows the effect
    when its controlled gain has the claim's sign and at least MIN_CLAIM_GAIN_PCT.
    For ">=" the threshold moves to the midpoint between the lowest descriptor
    value with the effect and the highest without it below that (geometric mean
    when both are positive); with no workload below, it drops to the lowest value
    with the effect so the rule speaks wherever the effect was seen. "<" is the
    mirror. Returns (clauses, note) where note records what moved and why."""
    workload_descriptors = problem.get("workload_descriptors") or {}
    gains = per_workload_gains(claim, evidence, baseline_knobs, search_space, objective)
    expected_sign = 1.0
    if llm_signed_gain(claim) < 0:
        expected_sign = -1.0
    present = []
    absent = []
    for workload, gain in gains.items():
        if workload not in workload_descriptors or workload_descriptors[workload] is None:
            continue
        if gain * expected_sign >= MIN_CLAIM_GAIN_PCT:
            present.append(workload)
        else:
            absent.append(workload)
    note = {"present": present, "absent": absent, "moved": []}
    if len(present) == 0:
        return clauses, note
    fitted = []
    for clause in clauses:
        metric = clause["metric"]
        base_metric = metric
        if base_metric.startswith("max_"):
            base_metric = base_metric[len("max_"):]
        present_values = []
        for workload in present:
            if base_metric in workload_descriptors[workload]:
                present_values.append(float(workload_descriptors[workload][base_metric]))
        absent_values = []
        for workload in absent:
            if base_metric in workload_descriptors[workload]:
                absent_values.append(float(workload_descriptors[workload][base_metric]))
        new_clause = dict(clause)
        old_value = float(clause["value"])
        if len(present_values) == 0:
            fitted.append(new_clause)
            continue
        if clause["op"] in [">=", ">"]:
            edge = min(present_values)
            below = [value for value in absent_values if value < edge]
            if len(below) > 0:
                new_clause["value"] = midpoint(max(below), edge)
            else:
                new_clause["value"] = edge
            new_clause["op"] = ">="
        else:
            edge = max(present_values)
            above = [value for value in absent_values if value > edge]
            if len(above) > 0:
                new_clause["value"] = midpoint(edge, min(above))
            else:
                new_clause["value"] = edge * 1.001 + 1e-9
            new_clause["op"] = "<"
        if abs(new_clause["value"] - old_value) > 1e-9:
            note["moved"].append({"metric": metric, "from": old_value, "to": new_clause["value"]})
        fitted.append(new_clause)
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
