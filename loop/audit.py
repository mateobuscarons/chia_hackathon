"""Autopsy of a transfer experiment: WHY did the rules win or lose on the test SoC?

Usage: python -m loop.audit results/experiment_crosssoc_C.json
Reads the learned playbook, the test SoC's ground-truth tables and the claim bets.
"""

import json
import sys

from loop import forecast
from loop.champsim_problem import BASELINE_KNOBS, make_problem
from loop.configs import config_name

TRACES = {"mcf": "traces/605.mcf_s-665B.champsimtrace.xz", "lbm": "traces/619.lbm_s-2676B.champsimtrace.xz",
          "omnetpp": "traces/620.omnetpp_s-874B.champsimtrace.xz"}
SIZE_KNOBS = ["l2_sets", "llc_sets"]


def single_knob_truth(table, soc, knob, value):
    """True gain of changing ONE knob from the baseline, from the dense sweep."""
    knobs = dict(BASELINE_KNOBS)
    knobs[knob] = value
    baseline = table[config_name(BASELINE_KNOBS, soc)]["metrics"]["ipc"]
    changed = table[config_name(knobs, soc)]["metrics"]["ipc"]
    return 100.0 * (changed / baseline - 1.0)


def claim_bet_record(report, rule_id):
    wins = 0
    total = 0
    for problem_name in report["test"]:
        for arm in report["test"][problem_name]["arms"]:
            for run in report["test"][problem_name]["arms"][arm].values():
                for bet in run["bets"]:
                    if bet["forecaster"] == rule_id and bet.get("kind") == "claim" and bet["outcome"] is not None:
                        total += 1
                        if (bet["probability"] >= 0.5) == bet["outcome"]:
                            wins += 1
    return wins, total


def audit(report):
    soc = report["settings"]["test_soc"]
    rules = report["rules"]

    print("== 1. condition metric vs claim-bet record on the unseen SoC")
    by_metric = {}
    for rule in rules:
        wins, total = claim_bet_record(report, rule["id"])
        entry = by_metric.setdefault(rule["condition"]["metric"], [0, 0, 0])
        entry[0] += 1
        entry[1] += wins
        entry[2] += total
    for metric in by_metric:
        count, wins, total = by_metric[metric]
        rate = "n/a" if total == 0 else "{:.0%}".format(wins / total)
        print("   {:<10} {:>2} rules  claim bets {:>4}  win rate {}".format(metric, count, total, rate))

    print("== 2./3./4. claims of the rules that FIRE on each test workload")
    for trace_name in TRACES:
        problem = make_problem(soc, TRACES[trace_name], allow_simulation=False)
        table = problem["holder"].sweep_table
        baseline = problem["evaluate"](BASELINE_KNOBS)
        firing = [rule for rule in rules if forecast.condition_holds(rule["condition"], baseline)]
        distinct = set()
        vacuous = 0
        right_direction = 0
        magnitude_error = []
        by_type = {"policy": [0, 0], "size": [0, 0]}
        for rule in firing:
            claim = rule["claim"]
            distinct.add((claim["knob"], str(claim["value"])))
            if str(claim["value"]) == str(BASELINE_KNOBS[claim["knob"]]):
                vacuous += 1
                continue
            truth = single_knob_truth(table, soc, claim["knob"], type(BASELINE_KNOBS[claim["knob"]])(claim["value"]))
            promised = claim["gain_pct"]
            knob_type = "size" if claim["knob"] in SIZE_KNOBS else "policy"
            by_type[knob_type][1] += 1
            if (promised > 0) == (truth > 0.5):
                right_direction += 1
                by_type[knob_type][0] += 1
            magnitude_error.append(abs(promised - truth))
        informative = len(firing) - vacuous
        print("   {:<8} {:>2} rules fire | {:>2} distinct claims | {} vacuous (claim = baseline value) | direction right {}/{} | median |promised-true| {:.0f} pts | policy {}/{} right, size {}/{} right".format(
            trace_name, len(firing), len(distinct), vacuous, right_direction, informative,
            sorted(magnitude_error)[len(magnitude_error) // 2] if magnitude_error else 0,
            by_type["policy"][0], by_type["policy"][1], by_type["size"][0], by_type["size"][1]))


if __name__ == "__main__":
    with open(sys.argv[1]) as report_file:
        audit(json.load(report_file))
