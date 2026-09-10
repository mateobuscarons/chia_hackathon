"""Tables from an experiment report, finished or still running.

  python -m loop.summarize results/experiment_X.json [more reports] [--reference 0.7491]

Per test problem: designs each arm needed to reach 90 / 95 / 99% of the
reference (per seed, censored median), how many seeds got there, final best,
area under the best-so-far curve; then the calibration scoreboard per forecaster
class on the shared dispute bets. The reference is each cell's fixed best known
design (pass it with --reference); without one the best design any arm found in
the report is used and the table says so. Arms and seeds still running are
simply the ones not yet in the report.
"""

import json
import sys

TARGET_FRACTIONS = [0.9, 0.95, 0.99]


def forecaster_class(forecaster_id):
    if forecaster_id == "SURROGATE":
        return "surrogate"
    if forecaster_id.startswith("RULE"):
        return "rules"
    if forecaster_id.startswith("CARD"):
        return "cards"
    if "-REPLY-" in forecaster_id:
        return "analyst"
    if "-LLM-" in forecaster_id:
        return "llm_direct"
    if "-MEM-" in forecaster_id:
        return "memory"
    if "-HANDOFF-" in forecaster_id:
        return "handoff"
    return "other"


def best_so_far(history, objective="ipc"):
    curve = []
    best = None
    for row in history:
        if best is None or row[objective] > best:
            best = row[objective]
        curve.append(best)
    return curve


def best_found(trace_report):
    """The best IPC any arm found in this report (fallback reference)."""
    best = None
    for arm in trace_report["arms"]:
        for run in trace_report["arms"][arm].values():
            for row in run["history"]:
                if best is None or row["ipc"] > best:
                    best = row["ipc"]
    return best


def designs_to_target(history, reference, fraction):
    """Designs until best-so-far captures `fraction` of the gap between the
    baseline (history[0]) and the reference; None if never. Index 0 is the
    free baseline run, so the count is the number of designs bought."""
    baseline = history[0]["ipc"]
    target = baseline + fraction * (reference - baseline)
    curve = best_so_far(history)
    for index, value in enumerate(curve):
        if value >= target:
            return index
    return None


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def area_under_curve(history, objective="ipc"):
    """Mean best-so-far gain over the baseline across the run (reference-free;
    rewards getting good early)."""
    baseline = history[0][objective]
    curve = best_so_far(history, objective)
    if len(curve) <= 1:
        return 0.0
    total = 0.0
    for value in curve[1:]:
        total += (value / baseline - 1.0)
    return 100.0 * total / (len(curve) - 1)


def all_bets(report):
    """Every settled bet in the report. Kinds: sign / half / full (the three fixed
    questions about a claim, answered by rules, surrogate and replies alike),
    forecast (an agent's own within-5% forecast). Levels: suite / workload."""
    bets = list(report.get("learn_bets", []))
    for problem_name in report["test"]:
        for arm in report["test"][problem_name]["arms"]:
            for seed in report["test"][problem_name]["arms"][arm]:
                bets += report["test"][problem_name]["arms"][arm][seed]["bets"]
    settled = []
    for bet in bets:
        if bet["outcome"] is not None:
            settled.append(bet)
    return settled


def brier_line(label, class_bets):
    brier = 0.0
    wins = 0
    for bet in class_bets:
        outcome = 1.0 if bet["outcome"] else 0.0
        brier += (bet["probability"] - outcome) ** 2
        if (bet["probability"] >= 0.5) == bet["outcome"]:
            wins += 1
    return "   {:<28} bets={:>4} brier={:.3f} win-rate={:.2f}".format(
        label, len(class_bets), brier / len(class_bets), wins / len(class_bets))


def summarize(report, reference=None):
    for problem_name in report["test"]:
        trace_report = report["test"][problem_name]
        cell_reference = reference
        reference_note = "fixed reference"
        if cell_reference is None:
            cell_reference = best_found(trace_report)
            reference_note = "best design any arm found in this report (pass --reference for the fixed one)"
        if cell_reference is None:
            print("== {} | no finished run yet".format(problem_name))
            continue
        print("== {} | reference {:.4f} ({})".format(problem_name, cell_reference, reference_note))
        print("   {:<11} {:>5} {:>22} {:>22} {:>22} {:>10} {:>7}".format(
            "arm", "seeds", "to 90% (median, hit)", "to 95% (median, hit)", "to 99% (median, hit)", "final best", "auc %"))
        for arm in trace_report["arms"]:
            runs = trace_report["arms"][arm]
            if len(runs) == 0:
                print("   {:<11} {:>5}  (not finished yet)".format(arm, 0))
                continue
            columns = []
            for fraction in TARGET_FRACTIONS:
                counts = []
                hits = 0
                for seed in runs:
                    history = runs[seed]["history"]
                    count = designs_to_target(history, cell_reference, fraction)
                    if count is None:
                        count = len(history)      # censored at the budget
                    else:
                        hits += 1
                    counts.append(count)
                shown = "{:g}".format(median(counts))
                if hits < len(counts):
                    shown = ">" + shown if hits == 0 else shown
                columns.append("{:>14} {:>3}/{}".format(shown, hits, len(counts)))
            bests = []
            aucs = []
            for seed in runs:
                history = runs[seed]["history"]
                bests.append(best_so_far(history)[-1])
                aucs.append(area_under_curve(history))
            print("   {:<11} {:>5} {} {} {} {:>10.4f} {:>7.2f}".format(
                arm, len(runs), columns[0], columns[1], columns[2], sum(bests) / len(bests), sum(aucs) / len(aucs)))

    print("== calibration: the same three questions per claim (sign / half / full), suite level, per forecaster")
    bets = all_bets(report)
    for klass in ["surrogate", "rules", "analyst"]:
        for kind in ["sign", "half", "full"]:
            class_bets = []
            for bet in bets:
                if forecaster_class(bet["forecaster"]) == klass and bet["kind"] == kind and bet.get("level", "suite") == "suite":
                    class_bets.append(bet)
            if len(class_bets) > 0:
                print(brier_line("{} / {}".format(klass, kind), class_bets))
    print("== calibration: rules per workload (same three questions)")
    for kind in ["sign", "half", "full"]:
        class_bets = []
        for bet in bets:
            if forecaster_class(bet["forecaster"]) == "rules" and bet["kind"] == kind and bet.get("level") == "workload":
                class_bets.append(bet)
        if len(class_bets) > 0:
            print(brier_line("rules / {} / workload".format(kind), class_bets))
    print("== calibration: agents on their own forecasts (measured within 5% of the prediction or above)")
    for klass in ["analyst", "llm_direct", "memory", "handoff"]:
        class_bets = []
        for bet in bets:
            if forecaster_class(bet["forecaster"]) == klass and bet["kind"] == "forecast":
                class_bets.append(bet)
        if len(class_bets) > 0:
            print(brier_line(klass, class_bets))

    print("== playbook: {} rules, {} rejected, {} failed jobs".format(
        len(report.get("rules", [])), len(report.get("rejected_rules", [])), len(report.get("failed_jobs", []))))
    for rule in report.get("rules", []):
        verification = rule.get("verification") or {}
        print("   {} {}={} measured {:+.1f}% (LLM said {:+.1f}%, {} pair(s)) | {}".format(
            rule["id"], rule["claim"]["knob"], rule["claim"]["value"], rule["claim"]["gain_pct"],
            verification.get("llm_gain_pct", 0.0), verification.get("pairs", 0), rule["text"][:90]))


def load_merged(paths):
    """Several reports of the same experiment (different seeds) read as one."""
    merged = None
    for path in paths:
        with open(path) as report_file:
            report = json.load(report_file)
        if merged is None:
            merged = report
            continue
        for problem_name in report["test"]:
            merged["test"].setdefault(problem_name, {"arms": {}})
            for arm in report["test"][problem_name]["arms"]:
                merged["test"][problem_name]["arms"].setdefault(arm, {}).update(
                    report["test"][problem_name]["arms"][arm])
    return merged


if __name__ == "__main__":
    arguments = sys.argv[1:]
    reference = None
    if "--reference" in arguments:
        position = arguments.index("--reference")
        reference = float(arguments[position + 1])
        arguments = arguments[:position] + arguments[position + 2:]
    summarize(load_merged(arguments), reference)
