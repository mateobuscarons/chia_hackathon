"""Read a cell's report.

  python -m loop.summarize results/run_<cell>_<tag>.json [more reports of the same cell]
      Per arm, the share of the stock-to-best-known gap reached after 1..BUDGET
      designs (mean over seeds, with the spread) and the final best.
  python -m loop.summarize progress <run.log>
      Best-so-far per arm and seed while a cell is still running, from the one
      line every run prints per design.
The best known design is the best design measured on every workload of the
suite, in the cached tables (it grows as cells run).
"""

import json
import sys

from loop import run
from loop.champsim_problem import make_suite_problem, measured_designs


def load_reports(paths):
    """Several reports of the same cell (more seeds added later) read as one."""
    merged = None
    for path in paths:
        with open(path) as report_file:
            report = json.load(report_file)
        if merged is None:
            merged = report
            continue
        for arm in report["arms"]:
            if arm not in merged["arms"]:
                merged["arms"].append(arm)
        for arm in report["runs"]:
            merged["runs"].setdefault(arm, {})
            for seed in report["runs"][arm]:
                merged["runs"][arm][seed] = report["runs"][arm][seed]
    return merged


def best_so_far(designs):
    curve = []
    best = None
    for row in designs:
        if best is None or row["ipc"] > best:
            best = row["ipc"]
        curve.append(best)
    return curve


def summarize(report):
    problem = make_suite_problem(run.CELLS[report["cell"]]["test"], allow_simulation=False)
    known = measured_designs(problem)
    stock = None
    best_known = None
    for design in known:
        if design["name"] == problem["name_of"](problem["stock"]):
            stock = design["metrics"]["ipc"]
        if best_known is None or design["metrics"]["ipc"] > best_known:
            best_known = design["metrics"]["ipc"]
    budget = report["budget"]
    print("cell {} ({}): {} | stock {:.4f}, best known {:.4f} (+{:.1f}%) over {} designs measured on every workload".format(
        report["cell"], report["tag"], " + ".join(report["workloads"]), stock, best_known, 100.0 * (best_known / stock - 1.0), len(known)))
    print()
    header = "{:<14s} {:>5s}".format("arm", "seeds")
    for index in range(1, budget + 1):
        header += " {:>7s}".format("D{}".format(index))
    header += " {:>9s}".format("final")
    print("share of the stock-to-best-known gap reached after N designs (mean over seeds; min..max below)")
    print(header)
    for arm in report["arms"]:
        runs = report["runs"].get(arm, {})
        if len(runs) == 0:
            continue
        shares_by_index = []
        for index in range(budget + 1):
            shares_by_index.append([])
        finals = []
        for seed in sorted(runs):
            curve = best_so_far(runs[seed]["designs"])
            for index in range(1, budget + 1):
                value = curve[min(index, len(curve) - 1)]
                shares_by_index[index].append(100.0 * (value - stock) / (best_known - stock))
            finals.append(curve[-1])
        mean_line = "{:<14s} {:>5d}".format(arm, len(runs))
        spread_line = "{:<14s} {:>5s}".format("", "")
        for index in range(1, budget + 1):
            values = shares_by_index[index]
            mean_line += " {:>6.0f}%".format(sum(values) / len(values))
            spread_line += " {:>3.0f}..{:<3.0f}".format(min(values), max(values))
        mean_line += " {:>9.4f}".format(sum(finals) / len(finals))
        print(mean_line)
        if len(runs) > 1:
            print(spread_line)
    if len(report.get("failed", [])) > 0:
        print()
        print("failed runs:", json.dumps(report["failed"]))


def progress(log_path):
    """Best-so-far per arm and seed from the run log's per-design lines
    ("[arm-cell-sN] round R | Dk | ipc=V | source"), Ray worker prefixes included."""
    runs = {}
    with open(log_path) as log_file:
        for line in log_file:
            if "] round " not in line or "ipc=" not in line:
                continue
            head = line[:line.index("] round ")]
            if "[" not in head:
                continue
            tag = head[head.rindex("[") + 1:]
            try:
                value = float(line.split("ipc=")[1].split(" |")[0].strip())
            except (IndexError, ValueError):
                continue
            entry = runs.setdefault(tag, {"designs": 0, "best": None, "best_at": 0})
            entry["designs"] += 1
            if entry["best"] is None or value > entry["best"]:
                entry["best"] = value
                entry["best_at"] = entry["designs"]
    if len(runs) == 0:
        print("no design lines in " + log_path)
        return
    print("{:<12s} {:<6s} {:>8s} {:>9s} {:>9s}".format("arm", "seed", "designs", "best", "found at"))
    for tag in sorted(runs):
        entry = runs[tag]
        print("{:<12s} {:<6s} {:>8d} {:>9.4f} {:>9d}".format(tag.split("-")[0], "s" + tag.split("-s")[-1],
                                                              entry["designs"], entry["best"], entry["best_at"]))


if __name__ == "__main__":
    if sys.argv[1] == "progress":
        progress(sys.argv[2])
    else:
        summarize(load_reports(sys.argv[1:]))
