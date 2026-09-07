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

One round: fit surrogate -> analyst hypotheses -> every forecaster predicts
every candidate -> run the most disputed ones -> settle bets -> re-scope
losing rules. Every forecast (chosen or not) is logged for calibration plots.

Mechanism v3 ("rules earn their bets"):
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

from loop import analyst, forecast, playbook
from loop.surrogate_gp import normal_tail

# A rule that has lost this many bets on the current problem, with a losing
# record there, is re-scoped so it no longer applies to this baseline.
RESCOPE_AFTER_LOSSES = 3
# A measured claim must move the objective by at least this much to be a rule.
MIN_CLAIM_GAIN_PCT = 1.0


# One slot per round for an untried categorical value when the last round did
# not improve the best design. Part of our mechanism (rules / full arms only).
EXPLORE_ON_STALL = True


def run_loop(problem, rounds, per_round, store, surrogate, use_rules, use_analyst, tag,
             prior_history=None, seed=0, selector="ei"):
    """Returns {"history": [...], "rounds": [...]}; bets land in `store`.

    surrogate:   module with fit / predict_many / probability_at_least
    use_rules:   let playbook rules forecast, bet and warm-start the surrogate (transfer arm)
    use_analyst: let the LLM propose hypotheses (they forecast, bet and warm-start too)
    tag:         prefix for hypothesis ids so arms can be told apart in the ledger
    prior_history: runs from OTHER designs the surrogate may learn from
                 (pooled transfer); they never count as simulations
    selector:    "ei" - expected improvement over a GP that carries every
                 forecaster's belief as a prior; one slot per round goes to an
                 owed claim test (baseline + one knob) while credible rules have
                 untested claims here. Textbook BO is the special case with no
                 rules and no analyst.
                 "disagreement" - v2/v3 selector (spread + doubt + promise), kept
                 as an ablation of the acquisition function.
    """
    if prior_history is None:
        prior_history = []
    objective = problem["objective"]
    candidates = dict(problem["candidates"])
    baseline_name = name_of(problem, problem["baseline"])
    baseline_metrics = problem["evaluate"](problem["baseline"])
    reference = baseline_metrics[objective]
    history = [{"name": baseline_name, "knobs": problem["baseline"], "metrics": baseline_metrics,
                "reference": reference}]
    del candidates[baseline_name]

    all_rules = []
    if use_rules:
        all_rules = store["rules"]
    losses_here = {}          # rule id -> bets lost on THIS problem
    wins_here = {}
    rescoped_here = set()
    analyst_record = {"wins": 0, "losses": 0}   # the analyst's hypothesis bets on THIS problem
    honest_std = use_rules or use_analyst or selector == "disagreement"   # bets need honest doubt
    # The stall scan is part of OUR mechanism: baselines (bo, bo_pooled) stay textbook.
    stall_scan_enabled = EXPLORE_ON_STALL and (use_rules or use_analyst)
    last_round_improved = True     # no stall scan in round 1
    silent_tested = set()          # silent rules that already had their one claim test here
    round_logs = []
    for round_number in range(1, rounds + 1):
        rules = forecast.speaking_rules(all_rules, baseline_metrics)
        hypotheses = []
        if use_analyst:
            hypotheses = ask_hypotheses(problem, history, rules, candidates, per_round,
                                        "{}-HYP-r{}".format(tag, round_number))
        # Every forecaster's belief warm-starts the surrogate as a virtual experiment
        # (re-built every round: a re-scoped or discredited rule stops or weakens its prior).
        priors = forecast.rule_priors(rules, problem["baseline"], baseline_metrics, objective, history,
                                      problem["search_space"])
        priors = priors + forecast.hypothesis_priors(hypotheses, candidates, analyst_record)
        model = surrogate.fit(prior_history + history, problem["search_space"], objective, priors,
                              honest_std=honest_std, reference=reference)
        best_so_far = max(entry["metrics"][objective] for entry in history)

        chosen = []
        claim_test_of = {}
        silent_test_rule = None      # set only by the EI selector below
        silent_test_name = None
        if selector == "ei":
            # One slot for the most valuable owed claim test, the rest by expected improvement.
            owed = forecast.untested_claim_tests(rules, history, problem["baseline"], candidates,
                                                problem["search_space"], objective)
            if len(owed) > 0 and per_round > 1:
                test_name, rule_id = owed[0]
                chosen.append(test_name)
                claim_test_of[test_name] = rule_id
            # Silent rules: a credible rule whose condition fails on this chip still
            # gets ONE claim test here (its threshold came from another chip's metric
            # regime; Sep 7 seed 1: the LLC-prefetcher rule was right on C but silent).
            # A won test widens the condition to include this chip; a lost one leaves
            # the rule silent, as its condition said.
            if len(chosen) == 0 and per_round > 1 and use_rules:
                untested_silent = []
                for rule in forecast.silent_rules(all_rules, baseline_metrics):
                    if rule["id"] not in silent_tested:
                        untested_silent.append(rule)
                owed_silent = forecast.untested_claim_tests(untested_silent, history, problem["baseline"],
                                                            candidates, problem["search_space"], objective)
                if len(owed_silent) > 0:
                    silent_test_name, rule_id = owed_silent[0]
                    chosen.append(silent_test_name)
                    claim_test_of[silent_test_name] = "silent-rule test " + rule_id
                    silent_tested.add(rule_id)
                    silent_test_rule = rule_by_id(all_rules, rule_id)
            # (d) Right of reply: a silenced analyst (credibility below the floor) keeps
            # one simulated design per round to bet on, so it can earn its voice back.
            # Without it one lost bet silenced it for good (Sep 7 autopsy).
            analyst_silenced = forecast.analyst_credibility(analyst_record) < forecast.CREDIBLE_CONFIDENCE
            if analyst_silenced and len(hypotheses) > 0 and len(chosen) < per_round:
                reply = None
                for hypothesis in hypotheses:
                    if hypothesis["name"] in chosen or hypothesis["name"] not in candidates:
                        continue
                    if reply is None or hypothesis["confidence"] > reply["confidence"]:
                        reply = hypothesis
                if reply is not None:
                    chosen.append(reply["name"])
                    claim_test_of[reply["name"]] = "reply " + reply["id"]
            # (a) Stall scan: when the last round brought no improvement, one slot goes
            # to the incumbent with a never-tried categorical value (see forecast).
            # It never takes the last slot, so EI always keeps at least one pick.
            if stall_scan_enabled and not last_round_improved and len(chosen) < per_round - 1:
                scan_name = forecast.stall_scan_candidate(history, candidates, problem["search_space"],
                                                          surrogate, model, objective)
                if scan_name is not None and scan_name not in chosen:
                    chosen.append(scan_name)
                    claim_test_of[scan_name] = "stall scan"
            more = forecast.pick_expected_improvement(
                candidates, surrogate, model, prior_history + history, problem["search_space"],
                objective, per_round - len(chosen), best_so_far, seed=seed * 1000 + round_number,
                priors=priors, already_chosen=chosen, honest_std=honest_std)
            chosen = chosen + more
            forecasts = None
            if len(rules) > 0 or len(hypotheses) > 0:
                # Forecasts only for the chosen designs: that is all the bets need.
                chosen_candidates = {}
                for name in chosen:
                    chosen_candidates[name] = candidates[name]
                forecasts = forecast.gather_forecasts(chosen_candidates, surrogate, model, rules, hypotheses,
                                                      problem["baseline"], baseline_metrics, objective, history,
                                                      problem["search_space"])
        else:
            forecasts = forecast.gather_forecasts(candidates, surrogate, model, rules, hypotheses,
                                                  problem["baseline"], baseline_metrics, objective, history,
                                                  problem["search_space"])
            chosen = forecast.pick_most_disagreed(forecasts, per_round, best_so_far,
                                                  seed=seed * 1000 + round_number)
        logged = None
        if forecasts is not None:
            logged = compact_forecasts(forecasts, chosen)
        round_logs.append({"round": round_number, "forecasts": logged, "hypotheses": hypotheses,
                           "chosen": chosen, "claim_tests": claim_test_of})

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
                                                 surrogate_std, rules, hypotheses,
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

        # A silent rule bets on its own claim test exactly like a speaking rule would.
        if silent_test_rule is not None:
            claim = silent_test_rule["claim"]
            sibling = forecast.claim_sibling(claim, candidates[silent_test_name], problem["baseline"],
                                             problem["search_space"])
            sibling_metrics = forecast.find_measured(history, sibling)
            if sibling_metrics is not None:
                gain = forecast.effective_gain(silent_test_rule, history, problem["baseline"],
                                               problem["search_space"], objective)
                half_gain = sibling_metrics[objective] * (1.0 + gain / 200.0)
                probability = forecast.rule_confidence(silent_test_rule)
                if gain < 0:
                    probability = 1.0 - probability
                bets_by_name[silent_test_name].append(playbook.place_bet(
                    store, silent_test_rule["id"], silent_test_name,
                    "{} >= {:.4f}".format(objective, half_gain), probability, kind="claim"))

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
                if "-HYP-" in bet["forecaster"]:
                    if leaned_yes == happened:
                        analyst_record["wins"] += 1
                    else:
                        analyst_record["losses"] += 1
                if silent_test_rule is not None and bet["forecaster"] == silent_test_rule["id"]:
                    if leaned_yes == happened:
                        widen_rule(silent_test_rule, baseline_metrics, problem["name"], name)
                        print("[{}] widened {} after its claim test won here: {}".format(
                            tag, silent_test_rule["id"], json.dumps(silent_test_rule["conditions"])), flush=True)
                    else:
                        print("[{}] {} stays silent here: claim test lost".format(
                            tag, silent_test_rule["id"]), flush=True)
                rule = rule_by_id(rules, bet["forecaster"])
                if rule is None:
                    continue
                if leaned_yes == happened:
                    wins_here[rule["id"]] = wins_here.get(rule["id"], 0) + 1
                else:
                    losses_here[rule["id"]] = losses_here.get(rule["id"], 0) + 1
            label = "{} bets".format(len(bets_by_name[name]))
            if name in claim_test_of:
                if claim_test_of[name].startswith("RULE"):
                    label = label + " | claim test for " + claim_test_of[name]
                else:
                    label = label + " | " + claim_test_of[name]     # reply / stall scan / silent-rule test
            print("[{}] round {} | {} | {}={:.4f} | {}".format(
                tag, round_number, name, objective, metrics[objective], label), flush=True)

        for rule in rules:
            if rule["id"] in rescoped_here:
                continue
            lost = losses_here.get(rule["id"], 0)
            won = wins_here.get(rule["id"], 0)
            lost_claim = rule.get("claim_losses", 0) > 0 and lost > 0
            if (lost >= RESCOPE_AFTER_LOSSES and lost > won) or lost_claim:
                rescope(rule, problem, baseline_metrics, history[-1]["name"], history[-1]["metrics"],
                        wins=won, losses=lost)
                rescoped_here.add(rule["id"])
                print("[{}] re-scoped {} after {}-{} here: {}".format(
                    tag, rule["id"], won, lost, json.dumps(rule["conditions"])), flush=True)
    return {"history": history, "rounds": round_logs}


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


def ask_hypotheses(problem, history, rules, candidates, how_many, id_prefix):
    """Analyst proposals, filtered to valid, untested, in-budget configs."""
    proposals = analyst.propose_hypotheses(problem["search_space"], problem["objective"],
                                           format_table(history, problem["table_metrics"]),
                                           format_rules(rules), how_many)
    hypotheses = []
    for index, proposal in enumerate(proposals):
        if not valid_hypothesis(proposal) or not isinstance(proposal.get("knobs"), dict):
            continue
        name = None
        for candidate_name in candidates:
            if forecast.same_knobs(candidates[candidate_name], proposal["knobs"]):
                name = candidate_name
        if name is None:
            continue
        hypotheses.append({"id": "{}-{}".format(id_prefix, index + 1), "name": name,
                           "text": proposal["hypothesis"],
                           "predicted": float(proposal["predicted"]),
                           "confidence": float(proposal["confidence"])})
    return hypotheses


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


def to_probability(forecaster_id, predicted, threshold, surrogate_std, rules, hypotheses,
                   analyst_credibility=0.5):
    """P(objective >= threshold) for one forecaster's bet.

    The surrogate's probability is its own posterior. A rule or hypothesis gives
    a point forecast: we place a normal around it with the surrogate's std (the
    forecaster is at least as unsure about the rest of the config as the
    surrogate is), then shrink toward 0.5 by how little we trust the forecaster.
    A rule at 0.7 credibility forecasting far above the threshold bets ~0.85;
    one forecasting just above it bets ~0.55. v2 bet a flat 0.7 either way.
    """
    if forecaster_id == "SURROGATE":
        return normal_tail(predicted, surrogate_std, threshold)
    credibility = None
    for rule in rules:
        if rule["id"] == forecaster_id:
            credibility = forecast.rule_confidence(rule)
    for hypothesis in hypotheses:
        if hypothesis["id"] == forecaster_id:
            credibility = hypothesis["confidence"] * analyst_credibility
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


def rescope(rule, problem, baseline_metrics, name, metrics, wins=0, losses=0):
    """Ask the analyst to sharpen the condition; keep the old one in the rule's trail."""
    context = "problem: {}\nbaseline metrics: {}\nrecord on this problem: {} wins, {} losses\nlast experiment {}: {}".format(
        problem["name"], json.dumps(baseline_metrics), wins, losses, name, json.dumps(metrics))
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
            and not forecast.rule_applies(candidate, baseline_metrics)):
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
        new_clauses = [tightened_condition(old_clauses[0], baseline_metrics)] + old_clauses[1:]
        new_text = rule["text"] + " [narrowed: {} {} {:.3g}]".format(
            new_clauses[0]["metric"], new_clauses[0]["op"], new_clauses[0]["value"])
        how = "mechanical"
    rule["origin"].append({"rescoped_from": forecast.rule_clauses(rule), "because": context, "how": how})
    rule["conditions"] = new_clauses
    rule["condition"] = new_clauses[0]
    rule["text"] = new_text


def widen_rule(rule, baseline_metrics, problem_name, test_name):
    """A silent rule's claim test won on this chip: widen its condition so it
    applies here too, keeping the old clauses in the rule's trail. The mirror of
    re-scoping: losers get narrower, verified winners get wider."""
    old_clauses = forecast.rule_clauses(rule)
    new_clauses = forecast.widened_clauses(old_clauses, baseline_metrics)
    rule["origin"].append({"widened_from": old_clauses,
                           "because": "claim test {} won on {}".format(test_name, problem_name),
                           "how": "mechanical"})
    rule["conditions"] = new_clauses
    rule["condition"] = new_clauses[0]
    changed = []
    for old_clause, new_clause in zip(old_clauses, new_clauses):
        if old_clause != new_clause:
            changed.append("{} {} {:.3g}".format(new_clause["metric"], new_clause["op"], new_clause["value"]))
    rule["text"] = rule["text"] + " [widened: {}]".format(", ".join(changed))


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
        rule_id = playbook.add_rule(store, rule_clauses_of(proposal)[0], verified_claim, example,
                                    proposal["text"], conditions=rule_clauses_of(proposal),
                                    verification={"pairs": len(gains), "llm_gain_pct": llm_gain,
                                                  "problem": problem["name"]})
        print("[{}] admitted {} ({:+.2f}% measured over {} pairs, LLM said {:+.1f}%): {}".format(
            tag, rule_id, measured, len(gains), llm_gain, proposal["text"][:120]), flush=True)
    return extra_runs


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
