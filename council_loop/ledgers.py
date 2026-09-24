"""Ledgers from this loop's own records, so every run is scored the same way.

    python -m council_loop.ledgers <out_dir> <input> [<input> ...]

An input is a run report (`results/runs/*.json`, one ledger per arm and seed), `random:<cell>:
<seed>:<budget>` (the random arm rebuilt from its seed's draws and the table), a council
transcript (`results/transcripts/next-council-<tag>-s<seed>.log`, exact: every design with its
source and its refusal), or a run log of the forest or Gaussian-process arm
(`results/run_<tag>.log`; a design that beat the best and did not become the best was refused).
The ledger module is the CHIA block under council_loop/chia_blocks; it is loaded from its file so
nothing here depends on how the blocks are installed.
"""

import importlib.util
import json
import os
import re
import sys

LEDGER_MODULE = os.path.join("council_loop", "chia_blocks", "chia", "trace", "ledger.py")
specification = importlib.util.spec_from_file_location("ledger", LEDGER_MODULE)
ledger = importlib.util.module_from_spec(specification)
specification.loader.exec_module(ledger)


def fresh(out_dir, name):
    path = os.path.join(out_dir, name + ".jsonl")
    if os.path.exists(path):
        os.remove(path)
    return ledger.Ledger(path, "ipc")


def from_report(path, out_dir):
    """One ledger per arm and seed of a finished run report."""
    report = json.load(open(path))
    stem = os.path.basename(path)[:-5]
    made = {}
    for arm in report["runs"]:
        for seed in report["runs"][arm]:
            book = fresh(out_dir, "{}_{}_s{}".format(stem, arm, seed))
            for design in sorted(report["runs"][arm][seed]["designs"], key=lambda row: row["index"]):
                book.record(design["knobs"], {"ipc": design["ipc"], "mm2": design.get("mm2"), "watts": design.get("watts")},
                            design["source"], design["round"], design.get("hypothesis"), design.get("violations") or ())
            made[book.path] = book
    return made


def from_transcript(path, out_dir):
    """One ledger from a council transcript: `======== D<i> [<source>] ipc=<x> | ... ========`,
    with `| refused: <why>` when a cap refused the design."""
    stem = os.path.basename(path)[:-4]
    book = fresh(out_dir, stem)
    round_index = 0
    for line in open(path):
        head = re.match(r"#{16} round (\d+) \|", line)
        if head:
            round_index = int(head.group(1))
        row = re.match(r"======== D(\d+) \[([^\]]+)\] ipc=([\d.]+) \|(.*)========", line)
        if not row:
            continue
        violations = ()
        refused = re.search(r"\| refused: (.*?) $", row.group(4))
        if refused:
            violations = [refused.group(1).strip()]
        book.record({"design": int(row.group(1))}, {"ipc": float(row.group(3))}, row.group(2), round_index, None, violations)
    return {book.path: book}


def from_log(path, out_dir):
    """One ledger per run of the forest or Gaussian-process arm from its log lines:
    `[<run>] round <n> | D<i> | ipc=<x> | best <y> | <source> | <t> min`."""
    books = {}
    best = {}
    for line in open(path):
        row = re.match(r"\[([\w-]+)\] round (\d+) \| D(\d+) \| ipc=([\d.]+) \| best ([\d.]+) \| (\S+)", line)
        if not row:
            continue
        run = row.group(1)
        if run not in books:
            books[run] = fresh(out_dir, run)
            best[run] = None
        ipc = float(row.group(4))
        best_after = float(row.group(5))
        violations = ()
        if best[run] is not None and ipc > best[run] and best_after == best[run]:
            violations = ["refused (inferred from the log)"]
        best[run] = best_after
        books[run].record({"design": int(row.group(3))}, {"ipc": ipc}, row.group(6), int(row.group(2)), None, violations)
    return {book.path: book for book in books.values()}


def from_random_seed(cell, seed, budget, out_dir, opening=10, batch=5):
    """The random arm's ledger rebuilt from its own recipe: the seed fixes the draws, the table
    holds every measurement, so the ledger needs no report and no log. Rounds are read off the
    index (an opening wave, then `batch` a round), the same accounting every arm gets. A design
    the table does not hold yet ends the ledger there."""
    from council_loop.search import CELLS
    from council_loop.space import random_feasible_designs, violations
    from council_loop.suite import make_suite_problem
    problem = make_suite_problem(CELLS[cell], allow_simulation=False)
    holder = problem["holders"][0]
    stock_name = problem["name_of"](problem["stock"])
    book = fresh(out_dir, "{}_{}_random_seed{}".format(problem["holders"][0].trace_name, cell, seed))
    stock_row = holder.sweep_table.get(stock_name)
    stock_metrics = {problem["workloads"][0] + ":" + key: value for key, value in stock_row["metrics"].items()}
    stock_metrics["ipc"] = stock_row["metrics"]["ipc"]
    book.record(problem["stock"], {"ipc": stock_metrics["ipc"]}, "stock", 0)
    count = 0
    for knobs in random_feasible_designs(budget + 20, seed=seed):
        if count >= budget:
            break
        name = problem["name_of"](knobs)
        if name == stock_name:
            continue
        row = holder.sweep_table.get(name)
        if row is None or not row.get("metrics"):
            print("  table lacks design {} of seed {}; ledger ends here".format(count + 1, seed))
            break
        metrics = {problem["workloads"][0] + ":" + key: value for key, value in row["metrics"].items()}
        metrics["ipc"] = row["metrics"]["ipc"]
        metrics["watts"] = problem["evaluate"](knobs)["watts"]
        round_index = 1 if count < opening else 2 + (count - opening) // batch
        book.record(knobs, {"ipc": metrics["ipc"], "watts": metrics["watts"]}, "random", round_index, None,
                    violations(metrics, problem["workloads"], stock_metrics))
        count += 1
    return {book.path: book}


def main(out_dir, inputs):
    os.makedirs(out_dir, exist_ok=True)
    books = {}
    for path in inputs:
        if path.startswith("random:"):
            # random:<cell>:<seed>:<budget> rebuilds the random arm from its seed and the table
            _, cell, seed, budget = path.split(":")
            books.update(from_random_seed(cell, int(seed), int(budget), out_dir))
        elif path.endswith(".json"):
            books.update(from_report(path, out_dir))
        elif "/transcripts/" in path:
            books.update(from_transcript(path, out_dir))
        else:
            books.update(from_log(path, out_dir))
    named = {}
    for path in books:
        named[os.path.basename(path)[:-6]] = books[path]
    print(ledger.markdown(ledger.report(named, window=20)))
    print()
    print("best feasible after N designs (D0 is the stock):")
    for name in sorted(named):
        curve = named[name].best_so_far()
        cells = []
        for count in (10, 20, 30, 40, 60, 100, 200, 500, 1000):
            if count < len(curve) and curve[count] is not None:
                cells.append("{}: {:.4f}".format(count, curve[count]))
        print("  {:34s} {}".format(name, " | ".join(cells)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
