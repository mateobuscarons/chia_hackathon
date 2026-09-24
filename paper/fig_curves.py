"""Figure: best feasible IPC against designs measured (council, gated council, random), and the
cumulative model spend of the two council arms, caps on, means over seeds. Reads the shipped
ledgers and per-call rows; writes fig_curves.pdf beside this file.
    ../.venv/bin/python fig_curves.py
"""
import glob
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGERS = os.path.join(ROOT, "results", "ledgers", "lean")
CALLS = os.path.join(ROOT, "results", "vm2", "results", "llm_calls.jsonl")
specification = importlib.util.spec_from_file_location("ledger", os.path.join(ROOT, "council_loop", "chia_blocks", "chia", "trace", "ledger.py"))
ledger = importlib.util.module_from_spec(specification)
specification.loader.exec_module(ledger)

BUDGET = 60
ARMS = {"council": ("next_llama2_lean20_cap_council_s*.jsonl", "next-council-lean20_cap-s"),
        "council, gates applied": ("next_llama2_lean20_cap_apply_council_s*.jsonl", "next-council-lean20_cap_apply-s"),
        "random": ("llama2.c-llama2_7b.1_llama2_random_seed*.jsonl", None)}


def mean_curve(curves, length):
    """Mean over seeds at every index, each curve held at its last value past its end."""
    means = []
    for index in range(length + 1):
        values = []
        for curve in curves:
            values.append(curve[min(index, len(curve) - 1)])
        means.append(sum(values) / len(values))
    return means


calls_by_tag = {}
for line in open(CALLS):
    row = json.loads(line)
    calls_by_tag.setdefault(row["tag"], []).append(row)

best = {}
spend = {}
random_final = None
for arm in ARMS:
    pattern, tag_prefix = ARMS[arm]
    books = [ledger.Ledger(path, "ipc") for path in sorted(glob.glob(os.path.join(LEDGERS, pattern)))]
    curves = [book.best_so_far() for book in books]
    if arm == "random":
        random_final = sum(curve[1000] for curve in curves) / len(curves)
    best[arm] = mean_curve(curves, BUDGET)
    if tag_prefix is None:
        continue
    per_seed = []
    for seed, book in enumerate(books):
        usd_by_round = {}
        for row in calls_by_tag.get(tag_prefix + str(seed), []):
            usd_by_round[row["round"]] = usd_by_round.get(row["round"], 0.0) + row["cost_usd"]
        running = 0.0
        last_round = None
        cumulative = []
        for entry in book.rows:
            if entry["round"] != last_round:
                running += usd_by_round.get(entry["round"], 0.0)
                last_round = entry["round"]
            cumulative.append(running)
        per_seed.append(cumulative)
    spend[arm] = mean_curve(per_seed, BUDGET)

figure, (top, bottom) = pyplot.subplots(2, 1, figsize=(3.45, 2.75), sharex=True, gridspec_kw={"height_ratios": [1.25, 1]})
styles = {"council": ("#2F6FAF", 1.3), "council, gates applied": ("#6B4FA0", 1.8), "random": ("#9A9AA0", 1.3)}
for arm in ARMS:
    top.step(range(BUDGET + 1), best[arm], where="post", color=styles[arm][0], linewidth=styles[arm][1], label=arm)
top.axhline(random_final, color="#9A9AA0", linewidth=0.8)
top.text(1, random_final + 0.012, "random after 1000 designs", ha="left", va="bottom", fontsize=6.5, color="#555555")
top.set_ylabel("best feasible IPC", fontsize=7.5)
top.tick_params(labelsize=7)
top.legend(fontsize=6.5, frameon=False, loc="lower right")
for arm in spend:
    bottom.step(range(BUDGET + 1), spend[arm], where="post", color=styles[arm][0], linewidth=styles[arm][1])
bottom.set_ylabel("model USD, cumulative", fontsize=7.5)
bottom.set_xlabel("designs measured", fontsize=7.5)
bottom.tick_params(labelsize=7)
for axis in (top, bottom):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
figure.tight_layout(pad=0.4)
figure.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_curves.pdf"))
print("random after 1000: %.4f | council at 60: %.3f | applied at 60: %.3f | USD at 60: %.2f v %.2f" % (
    random_final, best["council"][BUDGET], best["council, gates applied"][BUDGET], spend["council"][BUDGET], spend["council, gates applied"][BUDGET]))
