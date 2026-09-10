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


def add_rule(store, condition, claim, example, text, conditions=None, verification=None):
    """Add a rule and return its id. New rules start with an empty record.

    condition:  {"metric": "LLC_mpki", "op": ">=", "value": 20} - checked on
                the SoC/trace baseline run, decides whether the rule speaks.
    conditions: optional list of such clauses, ANDed (condition = the first one).
    claim:      {"knob": "l2_prefetcher", "value": "spp_dev", "gain_pct": 15}
                - "switching this knob to this value changes the objective by
                about X%". gain_pct is MEASURED from a controlled comparison
                (see loop.verify_claims), never copied from the LLM.
    example:    the controlled comparison that motivated the rule.
    text:       the rule in the architect's own words, for humans.
    verification: how the claim was measured ({"pairs": n, "llm_gain_pct": x}).
    """
    rule_id = "RULE-{:03d}".format(len(store["rules"]) + 1)
    # The analyst may write numbers as strings; normalise once, here.
    if conditions is None:
        conditions = [condition]
    clean_clauses = []
    for clause in conditions:
        clean = dict(clause)
        clean["value"] = float(clean["value"])
        clean_clauses.append(clean)
    claim = dict(claim)
    claim["gain_pct"] = float(claim["gain_pct"])
    rule = {
        "id": rule_id,
        "condition": clean_clauses[0],
        "conditions": clean_clauses,
        "claim": claim,
        "example": example,
        "text": text,
        "verification": verification,
        "origin": [],
        "wins": 0,
        "losses": 0,
        "size_wins": 0,
        "size_losses": 0,
        "brier_scores": [],
        "status": "active",
    }
    store["rules"].append(rule)
    return rule_id


def reject_rule(store, rule, reason):
    """A proposed rule that failed verification is kept for the audit trail,
    but never forecasts or bets."""
    rejected = dict(rule)
    rejected["reason"] = reason
    store.setdefault("rejected_rules", []).append(rejected)


def place_bet(store, forecaster_id, experiment, event, probability, kind="sign", level="suite"):
    """Log a prediction BEFORE its experiment runs. Returns the bet id.

    event is an objectively checkable claim like "ipc >= 0.35" (or
    "bfs.urand:ipc >= 0.41" for one workload of a suite); the loop settles it
    with a boolean. Every forecaster answers the same three fixed questions
    about a design that carries a claim, relative to the design's sibling:
      kind "sign"  - does the objective move the claimed way at all?
      kind "half"  - by at least half the claimed gain?
      kind "full"  - by at least the full claimed gain?
    A "forecast" bet is an agent's own within-5% forecast. level is "suite" or
    "workload".
    """
    bet_id = "BET-{:04d}".format(len(store["bets"]) + 1)
    bet = {
        "id": bet_id,
        "forecaster": forecaster_id,
        "experiment": experiment,
        "event": event,
        "probability": probability,
        "kind": kind,
        "level": level,
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

    # A rule's credibility comes from its SIGN bets only: a rule whose direction
    # is right is never silenced for getting the size wrong. Size bets (half,
    # full) are recorded apart and decide re-bucketing instead.
    for rule in store["rules"]:
        if rule["id"] == bet["forecaster"]:
            rule["brier_scores"].append(brier)
            if bet["kind"] == "sign":
                if won:
                    rule["wins"] += 1
                else:
                    rule["losses"] += 1
            else:
                if won:
                    rule["size_wins"] = rule.get("size_wins", 0) + 1
                else:
                    rule["size_losses"] = rule.get("size_losses", 0) + 1
    return brier


def rule_record(rule):
    """One-line human summary: 'sign 2-1, size 1-2, Brier 0.12'."""
    if len(rule["brier_scores"]) == 0:
        return "no bets yet"
    average_brier = sum(rule["brier_scores"]) / len(rule["brier_scores"])
    return "sign {}-{}, size {}-{}, Brier {:.2f}".format(
        rule["wins"], rule["losses"], rule.get("size_wins", 0), rule.get("size_losses", 0), average_brier)
