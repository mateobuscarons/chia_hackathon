"""Morning report: one markdown file with every number the paper needs so far.
Usage: python -m loop.report results/experiment_crosssoc_C.json results/REPORT.md
"""

import json
import sys

from loop.configs import config_name, within_budget
from loop.champsim_problem import BASELINE_KNOBS
from loop.plots import all_bets, forecaster_class, median, reference_optimum, sims_to_target

BASE = "champsim/champsim_config.json"
SOCS = ["A_mobile", "B_midrange", "C_server"]
TRACES = ["605.mcf_s-665B", "619.lbm_s-2676B", "620.omnetpp_s-874B"]


def ground_truth_table():
    lines = ["| SoC | trace | baseline IPC | in-budget optimum | headroom | best config |", "|---|---|---|---|---|---|"]
    for soc in SOCS:
        for trace in TRACES:
            table = json.load(open("results/sweep_{}_{}.json".format(soc, trace)))
            baseline = table[config_name(BASELINE_KNOBS, soc)]["metrics"]["ipc"]
            best = None
            for name in table:
                entry = table[name]
                if within_budget(entry["knobs"], soc, BASE):
                    if best is None or entry["metrics"]["ipc"] > best[0]:
                        best = (entry["metrics"]["ipc"], name.split("_", 2)[2])
            lines.append("| {} | {} | {:.4f} | {:.4f} | {:+.0f}% | `{}` |".format(
                soc, trace.split(".")[1].split("_")[0], baseline, best[0], 100 * (best[0] / baseline - 1), best[1]))
    return "\n".join(lines)


def arms_table(report):
    problems = list(report["test"].keys())
    arms = list(report["test"][problems[0]]["arms"].keys())
    header = "| arm | " + " | ".join(p.split("/")[1] + " (per seed, median)" for p in problems) + " |"
    lines = [header, "|---|" + "---|" * len(problems)]
    for arm in arms:
        cells = []
        for problem_name in problems:
            runs = report["test"][problem_name]["arms"].get(arm, {})
            counts = []
            for seed in runs:
                history = runs[seed]["history"]
                count = sims_to_target(history, reference_optimum(report["test"][problem_name]))
                counts.append(len(history) if count is None else count)
            if len(counts) == 0:
                cells.append("running")
            else:
                cells.append("{} → **{}**".format(counts, median(counts)))
        lines.append("| {} | {} |".format(arm, " | ".join(cells)))
    return "\n".join(lines)


def calibration_table(report):
    lines = ["| forecaster class | dispute bets | Brier (lower better; 0.25 = coin flip) | win rate |", "|---|---|---|---|"]
    for klass in ["surrogate", "rules", "hypotheses"]:
        bets = [bet for bet in all_bets(report) if forecaster_class(bet["forecaster"]) == klass]
        if len(bets) == 0:
            continue
        brier = sum((bet["probability"] - (1.0 if bet["outcome"] else 0.0)) ** 2 for bet in bets) / len(bets)
        wins = sum(1 for bet in bets if (bet["probability"] >= 0.5) == bet["outcome"]) / len(bets)
        lines.append("| {} | {} | {:.3f} | {:.0%} |".format(klass, len(bets), brier, wins))
    return "\n".join(lines)


def rules_table(report):
    lines = ["| rule | record | re-scoped | text |", "|---|---|---|---|"]
    for rule in report["rules"]:
        lines.append("| {} | {}-{} | {}x | {} |".format(rule["id"], rule["wins"], rule["losses"],
                                                       len(rule["origin"]), rule["text"].replace("|", "/")))
    return "\n".join(lines)


if __name__ == "__main__":
    report = json.load(open(sys.argv[1]))
    cost = json.load(open("loop/llm_usage.json"))["total_cost_usd"]
    settings = report["settings"]
    text = "# Results, Sep 3 2026\n\n"
    text += "## Ground truth (dense sweeps, 9 x 81 configs)\n\n" + ground_truth_table() + "\n\n"
    text += "## Headline: learn on {}, test on unseen {} ({} rounds x {} per round, {} seeds)\n\n".format(
        settings["train_socs"], settings["test_soc"], settings["rounds"], settings["per_round"], settings["seeds"])
    text += "Simulations needed to capture 90% of the achievable gain (lower is better).\n\n" + arms_table(report) + "\n\n"
    text += "## Calibration on shared dispute bets\n\n" + calibration_table(report) + "\n\n"
    text += "## Playbook after learning on A and B\n\n" + rules_table(report) + "\n\n"
    text += "LLM spend to date: ${:.2f}.\n".format(cost)
    open(sys.argv[2], "w").write(text)
    print(text)
