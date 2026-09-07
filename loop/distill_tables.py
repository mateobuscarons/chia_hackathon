"""Re-distill a playbook from EVERYTHING measured on the training problems.

The learning loop distills from its own 20 runs. The result tables hold every
design ever simulated on each training problem (hundreds), so a distillation
pass over the tables gives the analyst far more evidence and the verifier far
more controlled pairs, at the cost of a few LLM calls and at most
VERIFY_SIMS_PER_PROBLEM extra simulations per problem.

Usage: python -m loop.distill_tables results/experiment_v3_C_playbook.json results/experiment_v4_playbook.json
Starts from the first playbook (existing rules are shown to the analyst so it
sharpens or adds instead of repeating) and writes the merged result to the second.
"""

import json
import sys

from loop import analyst, experiment, loop, playbook
from loop.champsim_problem import make_problem

TRAIN_SOCS = ["A_mobile", "B_midrange"]
TRAIN_TRACES = ["traces/605.mcf_s-665B.champsimtrace.xz",
                "traces/619.lbm_s-2676B.champsimtrace.xz"]


def distill_from_table(store, soc_name, trace_path):
    problem = make_problem(soc_name, trace_path, space_name="B")
    tag = "tables-" + problem["name"]
    designs = experiment.measured_designs(problem)
    # Baseline first, then the rest sorted by objective so the analyst sees the best designs.
    baseline_name = loop.name_of(problem, problem["baseline"])
    ordered = []
    for design in designs:
        if design["name"] == baseline_name:
            ordered.append(design)
    others = []
    for design in designs:
        if design["name"] != baseline_name:
            others.append(design)
    others.sort(key=lambda design: design["metrics"]["ipc"], reverse=True)
    # The prompt cannot hold 500 rows: baseline + top 40 + 20 spread across the rest.
    step = max(1, len(others) // 20)
    shown = ordered + others[:40] + others[40::step]
    table = loop.format_table(shown, problem["table_metrics"])
    proposals = analyst.distill_rules(problem["search_space"], table, "(none)",
                                      loop.format_rules(store["rules"]), problem["condition_metrics"])
    print("[{}] {} designs measured, {} shown, {} proposals".format(
        tag, len(designs), len(shown), len(proposals)), flush=True)
    history = [ordered[0]] if len(ordered) > 0 else []
    loop.verify_claims(store, proposals, problem, history, tag, experiment.VERIFY_SIMS_PER_PROBLEM,
                       evidence=designs)


if __name__ == "__main__":
    source = sys.argv[1]
    destination = sys.argv[2]
    store = playbook.load(source)
    before = len(store["rules"])
    for soc_name in TRAIN_SOCS:
        for trace_path in TRAIN_TRACES:
            distill_from_table(store, soc_name, trace_path)
    playbook.save(store, destination)
    print("playbook: {} -> {} rules ({} rejected in total) -> {}".format(
        before, len(store["rules"]), len(store.get("rejected_rules", [])), destination), flush=True)
