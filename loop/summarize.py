"""One-screen summary of an experiment. Usage: python -m loop.summarize results/experiment_X.json [more files]

Prints, per test problem, the simulations each arm needed to capture 90% of
the achievable gain (per seed, then median) and the best IPC it found; then
the calibration scoreboard per forecaster class on the shared dispute bets.
"""

import json
import sys

from loop.plots import all_bets, forecaster_class, median, reference_optimum, sims_to_target


def area_under_curve(history, objective="ipc"):
    """Mean best-so-far gain over the baseline across the run (reference-free;
    rewards getting good early)."""
    baseline = history[0][objective]
    best = None
    total = 0.0
    for row in history[1:]:
        if best is None or row[objective] > best:
            best = row[objective]
        total += (best / baseline - 1.0)
    if len(history) <= 1:
        return 0.0
    return 100.0 * total / (len(history) - 1)


def summarize(report):
    for problem_name in report["test"]:
        trace_report = report["test"][problem_name]
        optimum = reference_optimum(trace_report)
        print("== {} | reference best {:.4f} (best design any arm found unless a dense sweep exists)".format(
            problem_name, optimum))
        print("   {:<17} {:>26} {:>7} {:>6}  {:>10} {:>8}  {}".format(
            "arm", "designs to 90% gain", "median", "hit", "final best", "auc %", "best per seed"))
        for arm in trace_report["arms"]:
            counts = []
            bests = []
            aucs = []
            reached = 0
            for seed in trace_report["arms"][arm]:
                history = trace_report["arms"][arm][seed]["history"]
                count = sims_to_target(history, optimum)
                if count is None:
                    # Never reached: censored at the budget (reported as > budget).
                    count = len(history)
                else:
                    reached += 1
                counts.append(count)
                bests.append(max(row["ipc"] for row in history))
                aucs.append(area_under_curve(history))
            seeds = len(counts)
            shown = []
            for count, best in zip(counts, bests):
                if best >= history[0]["ipc"] + 0.9 * (optimum - history[0]["ipc"]):
                    shown.append(str(count))
                else:
                    shown.append(">" + str(count - 1))
            print("   {:<17} {:>26} {:>7} {:>6}  {:>10.4f} {:>8.2f}  {}".format(
                arm, "[" + ", ".join(shown) + "]", median(counts), "{}/{}".format(reached, seeds),
                sum(bests) / len(bests), sum(aucs) / len(aucs),
                " ".join("{:.4f}".format(b) for b in bests)))

    print("== calibration on shared dispute bets")
    for klass in ["surrogate", "rules", "hypotheses"]:
        bets = [bet for bet in all_bets(report) if forecaster_class(bet["forecaster"]) == klass]
        if len(bets) == 0:
            continue
        brier = 0.0
        wins = 0
        for bet in bets:
            outcome = 1.0 if bet["outcome"] else 0.0
            brier += (bet["probability"] - outcome) ** 2
            if (bet["probability"] >= 0.5) == bet["outcome"]:
                wins += 1
        print("   {:<11} bets={:>4} brier={:.3f} win-rate={:.2f}".format(
            klass, len(bets), brier / len(bets), wins / len(bets)))


def load_merged(paths):
    """Several runs of the same experiment (different seeds) read as one report."""
    merged = None
    for path in paths:
        with open(path) as report_file:
            report = json.load(report_file)
        if merged is None:
            merged = report
            continue
        for problem_name in report["test"]:
            merged["test"].setdefault(problem_name, {"optimum": None, "arms": {}})
            for arm in report["test"][problem_name]["arms"]:
                merged["test"][problem_name]["arms"].setdefault(arm, {}).update(
                    report["test"][problem_name]["arms"][arm])
    return merged


if __name__ == "__main__":
    summarize(load_merged(sys.argv[1:]))
