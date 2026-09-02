"""One-screen summary of an experiment report. Usage: python -m loop.summarize results/experiment_X.json

Prints, per test problem, the simulations each arm needed to capture 90% of
the achievable gain (per seed, then median) and the best IPC it found; then
the calibration scoreboard per forecaster class on the shared dispute bets.
"""

import json
import sys

from loop.plots import all_bets, forecaster_class, median, sims_to_target


def summarize(report):
    for problem_name in report["test"]:
        trace_report = report["test"][problem_name]
        print("== {} | optimum {:.4f}".format(problem_name, trace_report["optimum"]))
        print("   {:<17} {:>22} {:>7}  {}".format("arm", "sims to 90% gain", "median", "best ipc per seed"))
        for arm in trace_report["arms"]:
            counts = []
            bests = []
            for seed in trace_report["arms"][arm]:
                history = trace_report["arms"][arm][seed]["history"]
                count = sims_to_target(history, trace_report["optimum"])
                if count is None:
                    count = len(history)
                counts.append(count)
                bests.append(max(row["ipc"] for row in history))
            print("   {:<17} {:>22} {:>7}  {}".format(
                arm, str(counts), median(counts), " ".join("{:.4f}".format(b) for b in bests)))

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


if __name__ == "__main__":
    with open(sys.argv[1]) as report_file:
        summarize(json.load(report_file))
