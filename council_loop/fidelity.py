"""The fidelity check: do the designs rank the same at the development rung and at the headline rung?

    LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m council_loop.fidelity <cell> <high_warmup> <high_sim> [how_many]

Picks `how_many` designs (default 40) measured at the run's fidelity on the cell's trace: the
best feasible ones and the rest spread evenly over the IPC range, so the check covers where the
arms end up and where they start. Measures the same designs at the high rung (into that rung's
own table), then reports Spearman's rho, Kendall's tau and the top-five overlap between the two
rungs, with the ledger block's `rank_agreement`. Prints a Markdown table and writes
results/fidelity_<cell>.json.
"""

import importlib.util
import json
import os
import sys

from council_loop.search import CELLS, start_ray
from council_loop.space import violations
from council_loop.suite import make_suite_problem

LEDGER_MODULE = os.path.join("council_loop", "chia_blocks", "chia", "trace", "ledger.py")
specification = importlib.util.spec_from_file_location("ledger", LEDGER_MODULE)
ledger = importlib.util.module_from_spec(specification)
specification.loader.exec_module(ledger)


def pick(problem, how_many):
    """The designs to re-measure: the best feasible tenth of the low-rung table, then an even
    spread over the rest of the IPC range."""
    holder = problem["holders"][0]
    rows = []
    for name in holder.sweep_table:
        row = holder.sweep_table[name]
        if row.get("metrics") and row["metrics"].get("ipc") is not None:
            rows.append((row["metrics"]["ipc"], row["knobs"]))
    rows.sort(key=lambda item: -item[0])
    top = max(1, how_many // 4)
    chosen = rows[:top]
    rest = rows[top:]
    if rest:
        step = max(1, len(rest) // (how_many - top))
        for index in range(0, len(rest), step):
            if len(chosen) >= how_many:
                break
            chosen.append(rest[index])
    return chosen


def main(cell, high_warmup, high_sim, how_many=40):
    start_ray(1)                # the high rung simulates through CHIA's nodes
    low = make_suite_problem(CELLS[cell], allow_simulation=False)
    high = make_suite_problem(CELLS[cell], warmup=high_warmup, simulation=high_sim)
    chosen = pick(low, how_many)
    print("re-measuring {} designs of {} at {}M/{}M".format(len(chosen), cell, high_warmup // 1_000_000, high_sim // 1_000_000), flush=True)
    high_metrics = high["evaluate_many"]([knobs for _, knobs in chosen])
    low_values = []
    high_values = []
    rows = []
    for (low_ipc, knobs), metrics in zip(chosen, high_metrics):
        low_values.append(low_ipc)
        high_values.append(metrics["ipc"])
        rows.append({"knobs": knobs, "low_ipc": low_ipc, "high_ipc": metrics["ipc"],
                     "high_violations": violations(metrics, high["workloads"], high_metrics[0])})
    agreement = ledger.rank_agreement(low_values, high_values, top=5)
    result = {"cell": cell, "low_fidelity": list(low["fidelity"]), "high_fidelity": [high_warmup, high_sim],
              "count": len(rows), "agreement": agreement, "designs": rows}
    os.makedirs("results", exist_ok=True)
    with open("results/fidelity_{}.json".format(cell), "w") as out:
        json.dump(result, out, indent=1)
    print("| designs | Spearman | Kendall | top-5 overlap |")
    print("|---|---|---|---|")
    print("| {} | {:.3f} | {:.3f} | {:.0%} |".format(len(rows), agreement["spearman"], agreement["kendall"], agreement["top_overlap"]))
    return result


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 40)
