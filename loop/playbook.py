"""The playbook: rules and their bet ledger, stored as one plain JSON file.

A rule is a dict; a bet is a dict. No classes, no database — a file a
human can open and audit. Brier score = (probability - outcome)^2 per
bet, averaged; 0 is perfect, 0.25 is what a coin-flipper scores.
"""

import json
import os


def load(path):
    """Read the playbook file, or start an empty one."""
    if not os.path.exists(path):
        return {"rules": [], "bets": []}
    with open(path) as playbook_file:
        return json.load(playbook_file)


def save(store, path):
    with open(path, "w") as playbook_file:
        json.dump(store, playbook_file, indent=2)


def add_rule(store, condition, claim, example, text):
    """Add a rule and return its id. New rules start with an empty record.

    condition: {"metric": "LLC_mpki", "op": ">=", "value": 20} - checked on
               the SoC/trace baseline run, decides whether the rule speaks.
    claim:     {"knob": "l2_prefetcher", "value": "spp_dev", "gain_pct": 15}
               - "this knob value changes the objective by about X% vs baseline"
               (a point estimate: it is scored against the real number).
    example:   the run that motivated the rule, e.g. "B_midrange/mcf: 0.31->0.41".
    text:      the rule in the architect's own words, for humans.
    """
    rule_id = "RULE-{:03d}".format(len(store["rules"]) + 1)
    # The analyst may write numbers as strings; normalise once, here.
    condition = dict(condition)
    condition["value"] = float(condition["value"])
    claim = dict(claim)
    claim["gain_pct"] = float(claim["gain_pct"])
    rule = {
        "id": rule_id,
        "condition": condition,
        "claim": claim,
        "example": example,
        "text": text,
        "origin": [],
        "wins": 0,
        "losses": 0,
        "brier_scores": [],
        "status": "active",
    }
    store["rules"].append(rule)
    return rule_id


def place_bet(store, forecaster_id, experiment, event, probability, kind="dispute"):
    """Log a prediction BEFORE its experiment runs. Returns the bet id.

    event is an objectively checkable claim like "ipc >= 0.35"; the loop
    settles it with a boolean. kind is "dispute" (the shared question all
    forecasters answer, used to compare them) or "claim" (a rule's own
    promise; only these decide whether the rule is re-scoped).
    """
    bet_id = "BET-{:04d}".format(len(store["bets"]) + 1)
    bet = {
        "id": bet_id,
        "forecaster": forecaster_id,
        "experiment": experiment,
        "event": event,
        "probability": probability,
        "kind": kind,
        "outcome": None,
    }
    store["bets"].append(bet)
    return bet_id


def settle_bet(store, bet_id, event_happened):
    """Record the outcome of a bet and update the forecaster's record."""
    bet = None
    for candidate in store["bets"]:
        if candidate["id"] == bet_id:
            bet = candidate
    bet["outcome"] = event_happened

    brier = (bet["probability"] - (1.0 if event_happened else 0.0)) ** 2

    # A bet is "won" when the forecaster leaned the right way.
    leaned_yes = bet["probability"] >= 0.5
    won = (leaned_yes == event_happened)

    # A rule's record tracks its own claims, not the shared dispute questions.
    for rule in store["rules"]:
        if rule["id"] == bet["forecaster"] and bet["kind"] == "claim":
            rule["brier_scores"].append(brier)
            if won:
                rule["wins"] += 1
            else:
                rule["losses"] += 1
    return brier


def rule_record(rule):
    """One-line human summary: '2-1, Brier 0.12'."""
    if len(rule["brier_scores"]) == 0:
        return "no bets yet"
    average_brier = sum(rule["brier_scores"]) / len(rule["brier_scores"])
    return "{}-{}, Brier {:.2f}".format(rule["wins"], rule["losses"], average_brier)
