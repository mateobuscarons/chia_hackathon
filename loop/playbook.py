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


def add_rule(store, condition, claim, example):
    """Add a rule and return its id. New rules start with an empty record."""
    rule_id = "RULE-{:03d}".format(len(store["rules"]) + 1)
    rule = {
        "id": rule_id,
        "condition": condition,
        "claim": claim,
        "example": example,
        "wins": 0,
        "losses": 0,
        "brier_scores": [],
        "status": "active",
    }
    store["rules"].append(rule)
    return rule_id


def place_bet(store, forecaster_id, experiment, event, probability):
    """Log a prediction BEFORE its experiment runs. Returns the bet id.

    event is a human-readable, objectively checkable claim, e.g.
    "ipc gain over baseline >= 5%". The loop settles it with a boolean.
    """
    bet_id = "BET-{:04d}".format(len(store["bets"]) + 1)
    bet = {
        "id": bet_id,
        "forecaster": forecaster_id,
        "experiment": experiment,
        "event": event,
        "probability": probability,
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

    for rule in store["rules"]:
        if rule["id"] == bet["forecaster"]:
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
