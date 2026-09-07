"""Early signs from a RUNNING experiment. Usage: python -m loop.early results/experiment_v3_C.json results/v3_C.log

Reads the partial report (saved after every finished arm run) and the log, and prints:
  1. the playbook so far: admitted rules with measured vs LLM-claimed gain; rejected rules;
  2. per finished test run: arm, problem, seed, designs-to-target, best found;
  3. rule bets settled so far on the test SoC: win rate and Brier, per rule and overall.
Nothing here waits for the run to end.
"""

import json
import os
import re
import sys


def playbook_from_log(log_path):
    admitted = []
    rejected = []
    rescoped = []
    if not os.path.exists(log_path):
        return admitted, rejected, rescoped
    with open(log_path) as log_file:
        for line in log_file:
            if "] admitted " in line:
                admitted.append(line.strip())
            elif "] rejected " in line:
                rejected.append(line.strip())
            elif "] re-scoped " in line:
                rescoped.append(line.strip())
    return admitted, rejected, rescoped


def print_playbook(report, log_path):
    admitted, rejected, rescoped = playbook_from_log(log_path)
    print("== playbook so far: {} admitted, {} rejected, {} re-scope events".format(
        len(admitted), len(rejected), len(rescoped)))
    if report is not None:
        for rule in report.get("rules", []):
            verification = rule.get("verification") or {}
            print("   {} {} -> {}={} measured {:+.1f}% (LLM said {:+.1f}%, {} pair(s)) | {}".format(
                rule["id"], json.dumps(rule.get("conditions", [rule["condition"]])),
                rule["claim"]["knob"], rule["claim"]["value"], rule["claim"]["gain_pct"],
                float(verification.get("llm_gain_pct", 0.0)), verification.get("pairs", "?"),
                rule["text"][:90]))
        for rule in report.get("rejected_rules", []):
            claim = rule.get("claim", {})
            print("   REJECTED ({}) {}={} LLM said {}: {}".format(
                rule.get("reason"), claim.get("knob"), claim.get("value"), claim.get("gain_pct"),
                str(rule.get("text", ""))[:80]))
    else:
        for line in admitted + rejected:
            print("   " + line[:200])
    for line in rescoped:
        print("   " + line[:200])


def best_reference(trace_report):
    best = None
    for arm in trace_report["arms"]:
        for seed in trace_report["arms"][arm]:
            for row in trace_report["arms"][arm][seed]["history"]:
                if best is None or row["ipc"] > best:
                    best = row["ipc"]
    return best


def designs_to_target(history, reference):
    baseline = history[0]["ipc"]
    target = baseline + 0.9 * (reference - baseline)
    best = None
    for index, row in enumerate(history):
        if best is None or row["ipc"] > best:
            best = row["ipc"]
        if best >= target:
            return index
    return None


def print_runs(report):
    for problem_name in report["test"]:
        trace_report = report["test"][problem_name]
        reference = best_reference(trace_report)
        if reference is None:
            continue
        print("== {} | best found so far by anyone {:.4f}".format(problem_name, reference))
        for arm in trace_report["arms"]:
            for seed in sorted(trace_report["arms"][arm]):
                history = trace_report["arms"][arm][seed]["history"]
                count = designs_to_target(history, reference)
                shown = str(count) if count is not None else ">{}".format(len(history) - 1)
                print("   {:<10} seed {} | designs to 90% gain: {:>4} | best {:.4f} | baseline {:.4f}".format(
                    arm, seed, shown, max(row["ipc"] for row in history), history[0]["ipc"]))


def print_bets(report):
    per_rule = {}
    totals = {"rules": [0, 0, 0.0], "surrogate": [0, 0, 0.0], "hypotheses": [0, 0, 0.0]}
    for problem_name in report["test"]:
        for arm in report["test"][problem_name]["arms"]:
            for seed in report["test"][problem_name]["arms"][arm]:
                for bet in report["test"][problem_name]["arms"][arm][seed]["bets"]:
                    if bet["outcome"] is None:
                        continue
                    outcome = 1.0 if bet["outcome"] else 0.0
                    won = (bet["probability"] >= 0.5) == bet["outcome"]
                    brier = (bet["probability"] - outcome) ** 2
                    if bet["forecaster"] == "SURROGATE":
                        klass = "surrogate"
                    elif bet["forecaster"].startswith("RULE"):
                        klass = "rules"
                        record = per_rule.setdefault(bet["forecaster"], [0, 0, 0.0])
                        record[0] += 1
                        if won:
                            record[1] += 1
                        record[2] += brier
                    else:
                        klass = "hypotheses"
                    totals[klass][0] += 1
                    if won:
                        totals[klass][1] += 1
                    totals[klass][2] += brier
    print("== bets settled so far on the test SoC (coin flip = win 0.50, Brier 0.25)")
    for klass in totals:
        count, wins, brier_sum = totals[klass]
        if count == 0:
            continue
        print("   {:<11} bets={:>4} win-rate={:.2f} brier={:.3f}".format(
            klass, count, wins / count, brier_sum / count))
    for rule_id in sorted(per_rule):
        count, wins, brier_sum = per_rule[rule_id]
        print("   {:<11} bets={:>4} win-rate={:.2f} brier={:.3f}".format(
            rule_id, count, wins / count, brier_sum / count))


if __name__ == "__main__":
    report_path = sys.argv[1]
    log_path = sys.argv[2] if len(sys.argv) > 2 else report_path.replace("experiment_", "").replace(".json", ".log")
    report = None
    if os.path.exists(report_path):
        with open(report_path) as report_file:
            report = json.load(report_file)
    print_playbook(report, log_path)
    if report is None:
        print("(no report file yet: the learning phase is still running)")
    else:
        print_runs(report)
        print_bets(report)
