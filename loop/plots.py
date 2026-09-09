"""Paper figures from one experiment report. Usage: python -m loop.plots results/experiment_X.json [reference]

  fig_curves.png       best-so-far vs designs, per arm (median over seeds)
  fig_sims_to_target   designs to reach 90% of the reference, per arm
  fig_brier.png        running Brier score per forecaster class over settled bets
  fig_reliability.png  reliability diagram: stated probability vs observed frequency
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot

from loop.summarize import all_bets, best_found, best_so_far, designs_to_target, forecaster_class, median


def plot_curves(report, out_prefix, reference=None):
    problems = list(report["test"].keys())
    figure, axes = pyplot.subplots(1, len(problems), figsize=(5 * len(problems), 4), squeeze=False)
    for column, problem_name in enumerate(problems):
        axis = axes[0][column]
        trace_report = report["test"][problem_name]
        for arm in trace_report["arms"]:
            curves = []
            for seed in trace_report["arms"][arm]:
                curves.append(best_so_far(trace_report["arms"][arm][seed]["history"], "ipc"))
            length = min(len(curve) for curve in curves)
            medians = []
            for step in range(length):
                medians.append(median([curve[step] for curve in curves]))
            axis.plot(range(length), medians, label=arm)
        cell_reference = reference
        if cell_reference is None:
            cell_reference = best_found(trace_report)
        axis.axhline(cell_reference, color="black", linestyle=":", label="reference")
        axis.set_title(problem_name)
        axis.set_xlabel("designs")
        axis.set_ylabel("best IPC so far")
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(out_prefix + "_curves.png", dpi=150)


def plot_sims_to_target(report, out_prefix, reference=None):
    values_by_arm = {}
    for problem_name in report["test"]:
        trace_report = report["test"][problem_name]
        cell_reference = reference
        if cell_reference is None:
            cell_reference = best_found(trace_report)
        for arm in trace_report["arms"]:
            for seed in trace_report["arms"][arm]:
                history = trace_report["arms"][arm][seed]["history"]
                count = designs_to_target(history, cell_reference, 0.9)
                if count is None:
                    count = len(history)   # never reached: cap at the budget
                values_by_arm.setdefault(arm, []).append(count)
    arms = list(values_by_arm.keys())
    figure, axis = pyplot.subplots(figsize=(7, 4))
    axis.bar(arms, [median(values_by_arm[arm]) for arm in arms])
    axis.set_ylabel("designs to capture 90% of the reference gain (median)")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(out_prefix + "_sims_to_target.png", dpi=150)


def all_bets(report):
    bets = list(report["learn_bets"])
    for problem_name in report["test"]:
        for arm in report["test"][problem_name]["arms"]:
            for seed in report["test"][problem_name]["arms"][arm]:
                bets += report["test"][problem_name]["arms"][arm][seed]["bets"]
    # Classes are compared on the shared dispute questions; a rule's own claim
    # bets are a different (easier) question and would flatter the rules.
    return [bet for bet in bets if bet["outcome"] is not None and bet.get("kind", "dispute") == "dispute"]


def plot_brier(report, out_prefix):
    figure, axis = pyplot.subplots(figsize=(6, 4))
    bets = all_bets(report)
    for klass in ["surrogate", "rules", "analyst", "llm_direct"]:
        running = []
        total = 0.0
        count = 0
        for bet in bets:
            if forecaster_class(bet["forecaster"]) != klass:
                continue
            outcome = 1.0 if bet["outcome"] else 0.0
            total += (bet["probability"] - outcome) ** 2
            count += 1
            running.append(total / count)
        if len(running) > 0:
            axis.plot(range(1, len(running) + 1), running, label=klass)
    axis.axhline(0.25, color="gray", linestyle=":", label="coin flip")
    axis.set_xlabel("settled bets")
    axis.set_ylabel("running Brier score (lower is better)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_prefix + "_brier.png", dpi=150)


def plot_reliability(report, out_prefix):
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    figure, axis = pyplot.subplots(figsize=(4.5, 4.5))
    bets = all_bets(report)
    for klass in ["surrogate", "rules", "analyst", "llm_direct"]:
        xs = []
        ys = []
        for low, high in zip(bins[:-1], bins[1:]):
            in_bin = [bet for bet in bets
                      if forecaster_class(bet["forecaster"]) == klass and low <= bet["probability"] < high]
            if len(in_bin) == 0:
                continue
            xs.append(sum(bet["probability"] for bet in in_bin) / len(in_bin))
            ys.append(sum(1.0 for bet in in_bin if bet["outcome"]) / len(in_bin))
        axis.plot(xs, ys, marker="o", label=klass)
    axis.plot([0, 1], [0, 1], color="gray", linestyle=":")
    axis.set_xlabel("stated probability")
    axis.set_ylabel("observed frequency")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_prefix + "_reliability.png", dpi=150)


if __name__ == "__main__":
    report_path = sys.argv[1]
    with open(report_path) as report_file:
        report = json.load(report_file)
    prefix = report_path.replace(".json", "")
    reference = None
    if len(sys.argv) > 2:
        reference = float(sys.argv[2])
    plot_curves(report, prefix, reference)
    plot_sims_to_target(report, prefix, reference)
    plot_brier(report, prefix)
    plot_reliability(report, prefix)
    print("figures written with prefix", prefix)
