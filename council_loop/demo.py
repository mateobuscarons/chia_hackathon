"""The instruments' readout, from the records this repository ships: one command, no key, no
cluster, no simulator.

    python3 council_loop/demo.py

Four sections, each ending with the files its numbers come from: the call gate on CHIA's own
CIRCT loop; the gate's counterfactual on the council and what applying it saved; the ledger's
council-versus-random table; the loop's own CHIA profiler log summed by the spend view. The
spend and ledger blocks are loaded from this repository's copy, so the demo is the blocks at work.
"""

import glob
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = os.path.join(ROOT, "council_loop", "chia_blocks", "chia")


def block(name, path):
    """A block module loaded from its file, so nothing here depends on how CHIA is installed."""
    specification = importlib.util.spec_from_file_location(name, os.path.join(BLOCKS, path))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


spend = block("spend", os.path.join("trace", "spend.py"))
ledger = block("ledger", os.path.join("trace", "ledger.py"))


def rows_of(path):
    rows = []
    for line in open(path):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def circt_section():
    print("== 1. CHIA's own CIRCT assess stage, tracked and gated (one decider question per issue, threshold 0.65) ==")
    print("%-9s %6s %9s %13s %17s %8s %7s %s" % ("pass", "turns", "agree/16", "bugs skipped", "non-bugs skipped", "tokens", "USD", "decider USD"))
    folder = os.path.join(ROOT, "results", "audit", "circt_assess")
    for name in ["plain", "plain2", "plain3", "gated", "gated2"]:
        turns = 0
        agree = 0
        bugs_skipped = 0
        others_skipped = 0
        for row in rows_of(os.path.join(folder, "verdicts_{}.jsonl".format(name))):
            if row["assessed"]:
                turns += 1
            elif row["paper"] == "bug":
                bugs_skipped += 1
            else:
                others_skipped += 1
            if row["verdict"] and row["verdict"].split(" (")[0] == row["paper"]:
                agree += 1
        events = spend.read_profile(os.path.join(folder, name, "ChiaProfileCollector.log"))
        total = spend.spend_summary(events)["total"]
        decider = 0.0
        for event in events:
            if event["type"] == "call_gate":
                decider += event.get("decider_cost_usd") or 0.0
        label = "ungated" if name.startswith("plain") else "gated"
        print("%-9s %6d %9s %13d %17d %8s %7.3f %s" % (label, turns, "{}/16".format(agree), bugs_skipped, others_skipped,
                                                       "{:,}".format(total["input_tokens"] + total["output_tokens"]),
                                                       total["usd"], "%.4f" % decider if decider else "-"))
    print("from: results/audit/circt_assess/verdicts_*.jsonl and <pass>/ChiaProfileCollector.log (CHIA's Vertex node under its profiler);"
          " the paper's verdicts are its Table 5\n")


# The two regimes of the case study: the random arm's ledgers (the target is the mean of its
# 1000-design finals), the council's ledgers per gating arm, and the run tags of its per-call rows.
LEDGERS = os.path.join(ROOT, "results", "ledgers", "lean")
CALLS = os.path.join(ROOT, "results", "vm2", "results", "llm_calls.jsonl")
SKIPS = os.path.join(ROOT, "results", "vm2", "results", "skip_gate.jsonl")
REGIMES = {
    "caps on": {"random": "llama2.c-llama2_7b.1_llama2_random_seed*.jsonl",
                "shadow": ("next_llama2_lean20_cap_council_s*.jsonl", "next-council-lean20_cap-s"),
                "apply": ("next_llama2_lean20_cap_apply_council_s*.jsonl", "next-council-lean20_cap_apply-s")},
    "caps off": {"random": "nocap_llama2_nocap_random1000_random_s*.jsonl",
                 "shadow": ("nocap_llama2_lean20_nocap_council_s*.jsonl", "nocap-council-lean20_nocap-s"),
                 "apply": ("nocap_llama2_lean20_nocap_apply_council_s*.jsonl", "nocap-council-lean20_nocap_apply-s")},
}
CONCERNS = ["prefetch", "geometry", "replacement", "concurrency"]


def gate_section():
    print("== 2. The council's specialist calls: what the gate would skip (shadow verdicts, 8 runs), and what applying it saved ==")
    rows = []
    for row in rows_of(SKIPS):
        if "lean20" in row["tag"] and "apply" not in row["tag"] and row["probability"] is not None:
            rows.append(row)
    print("%-10s %8s %15s %10s %19s" % ("threshold", "skipped", "of which holds", "proposals", "gains > 0.005 lost"))
    for threshold in (0.3, 0.4, 0.5):
        skipped = [row for row in rows if row["probability"] < threshold]
        holds = sum(1 for row in skipped if not row["proposed"])
        lost = sum(1 for row in skipped if row["proposed"] and not row["refused"] and (row["gain"] or 0.0) > 0.005)
        print("%-10.1f %8s %15d %10d %19d" % (threshold, "{} ({:.0f}%)".format(len(skipped), 100.0 * len(skipped) / len(rows)),
                                             holds, len(skipped) - holds, lost))
    parts = []
    for concern in CONCERNS:
        mine = [row for row in rows if row["concern"] == concern]
        held = sum(1 for row in mine if not row["proposed"])
        skippable = sum(1 for row in mine if row["probability"] < 0.4)
        parts.append("{} held {:.0f}%, skippable at 0.4 {:.0f}%".format(concern, 100.0 * held / len(mine), 100.0 * skippable / len(mine)))
    print("per concern ({} calls each): {}".format(len(rows) // len(CONCERNS), "; ".join(parts)))
    print("from: results/vm2/results/skip_gate.jsonl ({} shadow calls of the frozen recipe)\n".format(len(rows)))


def arm_facts(regime, arm):
    """One gating arm of one regime: its ledgers, USD and calls per run from the per-call rows,
    finals, and designs to the regime's target (the mean of random's 1000-design finals)."""
    pattern, tag_prefix = REGIMES[regime][arm]
    randoms = []
    for path in sorted(glob.glob(os.path.join(LEDGERS, REGIMES[regime]["random"]))):
        randoms.append(ledger.Ledger(path, "ipc"))
    target = sum(book.final() for book in randoms) / len(randoms)
    books = []
    for path in sorted(glob.glob(os.path.join(LEDGERS, pattern))):
        books.append(ledger.Ledger(path, "ipc"))
    usd = []
    calls = []
    for seed in range(len(books)):
        tag = tag_prefix + str(seed)
        rows = [row for row in CALL_ROWS if row["tag"] == tag]
        usd.append(sum(spend.spend_summary([{"func": row["label"], "extra": row}])["total"]["usd"] for row in rows))
        calls.append(len(rows))
    to_target = [book.evaluations_to(target) for book in books]
    counted = [60 if n is None else n for n in to_target]
    return {"runs": len(books), "target": target, "randoms": randoms, "usd": sum(usd) / len(usd), "calls": sum(calls) / len(calls),
            "final": sum(book.final() for book in books) / len(books), "to_target": to_target,
            "mean_to_target": sum(counted) / len(counted), "reached": sum(1 for n in to_target if n is not None)}


CALL_ROWS = rows_of(CALLS)


def savings_lines():
    print("%-9s %-7s %5s %12s %10s %11s" % ("regime", "arm", "runs", "USD per run", "calls/run", "final IPC"))
    for regime in REGIMES:
        facts = {}
        for arm in ("shadow", "apply"):
            facts[arm] = arm_facts(regime, arm)
            print("%-9s %-7s %5d %12.2f %10.1f %11.3f" % (regime, arm, facts[arm]["runs"], facts[arm]["usd"], facts[arm]["calls"], facts[arm]["final"]))
        saving = 100.0 * (1.0 - facts["apply"]["usd"] / facts["shadow"]["usd"])
        print("          applied: the same final IPC for {:.0f}% less model spend".format(saving))
    print("from: results/vm2/results/llm_calls.jsonl (per-call rows, priced by the spend block) and results/ledgers/lean/*.jsonl\n")


def ledger_section():
    print("== 3. The ledger: the council as run against random search, scored from one row format ==")
    print("%-9s %8s %24s %8s" % ("regime", "target", "designs to target", "speedup"))
    for regime in REGIMES:
        facts = arm_facts(regime, "shadow")
        print("%-9s %8.4f %24s %8s" % (regime, facts["target"], "mean {:.0f} (60 = budget)".format(facts["mean_to_target"]),
                                       "{:.0f}x".format(1000.0 / facts["mean_to_target"])))
        curves = [book.best_so_far() for book in facts["randoms"]]
        points = []
        for n in (20, 40, 60, 100, 500, 1000):
            points.append("{:.3f} ({})".format(sum(curve[n] for curve in curves) / len(curves), n))
        print("          random's mean best-so-far after N designs: " + ", ".join(points))
    print("from: results/ledgers/lean/*.jsonl; speedup = random's 1000 designs / the council's mean designs to random's mean final\n")


def own_log_section():
    print("== 4. The loop's own CHIA profiler log, one per run: model calls, builds, runs, candidates, gate verdicts ==")
    for folder in sorted(glob.glob(os.path.join(ROOT, "results", "profiles", "*"))):
        events = spend.read_profile(os.path.join(folder, "ChiaProfileCollector.log"))
        counts = {}
        for event in events:
            if event["type"] == "complete":
                key = event.get("func")
            elif event["type"] in ("candidate", "call_gate", "call_gate_outcome", "context_gate"):
                key = event["type"]
            else:
                continue
            counts[key] = counts.get(key, 0) + 1
        total = spend.spend_summary(events)["total"]
        print("{}: {} | model spend {:.4f} USD over {} calls".format(
            os.path.basename(folder), ", ".join("{} {}".format(counts[key], key) for key in sorted(counts)), total["usd"], total["calls"]))
    print("from: results/profiles/<run>/ChiaProfileCollector.log, summed by the spend block (`chia viz-profile --format spend` gives the same table)")


if __name__ == "__main__":
    circt_section()
    gate_section()
    savings_lines()
    ledger_section()
    own_log_section()
