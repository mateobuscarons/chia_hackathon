"""The report: every table and figure of the paper from the records in results/, no model call.

    python -m council_loop.report <ledger_dir> <vm_results_dir> <out_dir> <arm>=<tag>[,<tag>] ...

An arm is a name and the run tags behind it, e.g. off=new10 calls=gate10 all=full10. For each arm
and seed the report reads the ledger (results/ledgers), the per-call cost rows
(results/llm_calls.jsonl), the gate rows (skip_gate.jsonl, context_gate.jsonl) and the run
report (results/runs), and writes gating.md (the table) and money_<tag>.png (cumulative model
USD against designs measured, with the best IPC so far above it) per run whose cost rows carry
their round. Prices are the list prices in council_loop/analyst.py.
"""

import glob
import json
import os
import sys


BLOCKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "council_loop", "chia_blocks")
if BLOCKS not in sys.path:
    sys.path.append(BLOCKS)
from chia.trace import ledger  # noqa: E402
from chia.trace.spend import spend_summary  # noqa: E402


def rows_of(path):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def usd(row):
    """One call's USD, summed by the spend block from the row's own `cost_usd` (the loop wrote it at
    the list price), so the council's cost and the CIRCT stage's cost come from the same function."""
    return spend_summary([{"func": row["label"], "extra": row}])["total"]["usd"]


def run_facts(tag, seed, ledger_dir, vm_results, calls, skips, gates, target):
    """Everything the table says about one run (one tag, one seed)."""
    report_path = glob.glob(os.path.join(vm_results, "runs", "*_llama2_{}.json".format(tag)))[0]
    report = json.load(open(report_path))
    run_tag = "{}-council-{}-s{}".format(report["soc"], tag, seed)
    book_path = glob.glob(os.path.join(ledger_dir, "*_llama2_{}_council_s{}.jsonl".format(tag, seed)))[0]
    book = ledger.Ledger(book_path, "ipc")
    rounds = report["runs"]["council"][str(seed)]["rounds"]
    climb_rounds = 0
    for round_ in rounds:
        if round_["mode"] == "climb" and round_["round"] >= 2:
            climb_rounds += 1
    mine = []
    for row in calls:
        if row["tag"] == run_tag:
            mine.append(row)
    specialist_rows = []
    analyst_rows = []
    for row in mine:
        if row["label"] == "analyst":
            analyst_rows.append(row)
        else:
            specialist_rows.append(row)
    total_usd = 0.0
    for row in mine:
        total_usd += usd(row)
    skipped = 0
    for row in skips:
        if row["tag"] == run_tag and row.get("skipped"):
            skipped += 1
    dropped_share = None
    gated = []
    for row in gates:
        if row.get("tag") == run_tag and row["mode"] == "apply":
            gated.append(row)
    if gated:
        before = sum(row["tokens_before"] for row in gated)
        after = sum(row["tokens_after"] for row in gated)
        dropped_share = 1.0 - after / before if before else None
    specialist_input = None
    if specialist_rows:
        specialist_input = sum(row["input_tokens"] for row in specialist_rows) / len(specialist_rows)
    return {"tag": tag, "seed": seed, "final": book.final(), "designs": len(book.rows) - 1,
            "to_target": book.evaluations_to(target), "usd": total_usd, "calls": len(mine),
            "analyst_calls": len(analyst_rows), "specialist_calls": len(specialist_rows), "climb_rounds": climb_rounds,
            "specialist_calls_per_climb": len(specialist_rows) / climb_rounds if climb_rounds else None,
            "usd_per_climb": None, "skipped": skipped, "dropped_share": dropped_share,
            "specialist_input": specialist_input, "rows": mine, "book": book}


def usd_per_climb_round(facts):
    """The analyst's mean call plus the mean specialist call times the specialist calls a climb
    round: the price of a round that consults the team, independent of how many rounds stalled."""
    analyst_rows = [row for row in facts["rows"] if row["label"] == "analyst"]
    specialist_rows = [row for row in facts["rows"] if row["label"] != "analyst"]
    if not analyst_rows or not specialist_rows or facts["specialist_calls_per_climb"] is None:
        return None
    analyst_mean = sum(usd(row) for row in analyst_rows) / len(analyst_rows)
    specialist_mean = sum(usd(row) for row in specialist_rows) / len(specialist_rows)
    return analyst_mean + specialist_mean * facts["specialist_calls_per_climb"]


def fmt(value, form):
    if value is None:
        return "-"
    return form.format(value)


def gating_table(arms, facts_by_arm, target):
    lines = ["| arm | run | seed | best IPC at end | designs | designs to {:.4f} | model USD | calls (analyst + specialists) | specialist calls per climb round | USD per climb round | calls skipped | briefing tokens dropped | input tokens per specialist call |".format(target),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for facts in facts_by_arm[arm]:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                arm, facts["tag"], facts["seed"], fmt(facts["final"], "{:.4f}"), facts["designs"], fmt(facts["to_target"], "{}"),
                fmt(facts["usd"], "{:.2f}"), "{} ({} + {})".format(facts["calls"], facts["analyst_calls"], facts["specialist_calls"]),
                fmt(facts["specialist_calls_per_climb"], "{:.1f}"), fmt(usd_per_climb_round(facts), "{:.3f}"), facts["skipped"],
                fmt(facts["dropped_share"], "{:.0%}"), fmt(facts["specialist_input"], "{:.0f}")))
    return "\n".join(lines)


def money_figure(facts, out_path):
    """Cumulative model USD against designs measured, the best IPC so far above it. Needs the
    round on every cost row (runs launched after the round was recorded)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    rows = facts["rows"]
    if not rows or any("round" not in row for row in rows):
        return False
    book = facts["book"]
    usd_by_round = {}
    for row in rows:
        usd_by_round[row["round"]] = usd_by_round.get(row["round"], 0.0) + usd(row)
    designs = []
    spent = []
    best = []
    curve = book.best_so_far()
    running = 0.0
    last_round = None
    for index, entry in enumerate(book.rows):
        if entry["round"] != last_round:
            running += usd_by_round.get(entry["round"], 0.0)
            last_round = entry["round"]
        designs.append(index)
        spent.append(running)
        best.append(curve[index] if curve[index] is not None else 0.0)
    figure, (top, bottom) = pyplot.subplots(2, 1, figsize=(5.0, 4.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    top.step(designs, best, where="post", color="black")
    top.set_ylabel("best feasible IPC")
    top.set_title("{} seed {}: {:.2f} USD, best {:.4f}".format(facts["tag"], facts["seed"], facts["usd"], facts["final"] or 0.0), fontsize=9)
    bottom.step(designs, spent, where="post", color="tab:red")
    bottom.set_ylabel("model USD, cumulative")
    bottom.set_xlabel("designs measured")
    figure.tight_layout()
    figure.savefig(out_path, dpi=150)
    pyplot.close(figure)
    return True


def main(ledger_dir, vm_results, out_dir, arm_specs):
    os.makedirs(out_dir, exist_ok=True)
    calls = rows_of(os.path.join(vm_results, "llm_calls.jsonl"))
    skips = rows_of(os.path.join(vm_results, "skip_gate.jsonl"))
    gates = rows_of(os.path.join(vm_results, "context_gate.jsonl"))
    # the target: the strongest random seed's final value, from the random ledgers
    target = 0.0
    for path in glob.glob(os.path.join(ledger_dir, "*random_seed*.jsonl")):
        target = max(target, ledger.Ledger(path, "ipc").final() or 0.0)
    arms = []
    facts_by_arm = {}
    for spec in arm_specs:
        arm, tags = spec.split("=")
        arms.append(arm)
        facts_by_arm[arm] = []
        for tag in tags.split(","):
            for path in sorted(glob.glob(os.path.join(ledger_dir, "*_llama2_{}_council_s*.jsonl".format(tag)))):
                seed = int(path.rsplit("_s", 1)[1][:-6])
                facts = run_facts(tag, seed, ledger_dir, vm_results, calls, skips, gates, target)
                facts_by_arm[arm].append(facts)
                if money_figure(facts, os.path.join(out_dir, "money_{}_s{}.png".format(tag, seed))):
                    print("figure: money_{}_s{}.png".format(tag, seed))
    table = gating_table(arms, facts_by_arm, target)
    with open(os.path.join(out_dir, "gating.md"), "w") as out_file:
        out_file.write(table + "\n\nTarget: the strongest random seed's best after 1000 designs. USD at Gemini list prices; "
                       "USD per climb round = mean analyst call + mean specialist call x specialist calls per climb round.\n")
    print(table)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
