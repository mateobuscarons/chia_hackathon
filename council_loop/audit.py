"""The grounding audit: does the evidence an agent read support the move it proposed?

For every proposal a council member made in a finished run, the decider (TypeSafe's Jev, the
same one behind the gates) reads the sheet the member read, the move and the member's stated
reasoning, and answers one question: does the sheet's evidence support this move? Its
probability is written beside what the move then measured, so the audit reads whether grounded
proposals gain more often than ungrounded ones. Nothing is judged from numbers; the decider
reads text. Usage: python -m council_loop.audit <run report json>... ; rows go to results/audit/grounding.jsonl.
"""

import json
import os
import sys

from council_loop import analyst

MEMBERS = ("analyst", "prefetch", "geometry", "replacement", "concurrency", "restructure")
OUT = os.path.join("results", "audit", "grounding.jsonl")
QUESTION = {"grounded": {"type": "noul",
            "instructions": "Does the sheet's evidence (its bottleneck, level verdicts, findings and failing knobs) support this move, as the reasoning claims?",
            "criteria": {"true": "the move acts on a level or knob the sheet's evidence points at, and the reasoning cites that evidence",
                         "false": "the reasoning cites nothing the sheet shows, contradicts it, or moves knobs the sheet does not point at"}}}


def decider():
    """The loop's decider (TypeSafe's Jev), the same one behind its gates."""
    return analyst.decider()


def move_text(moves):
    parts = []
    for knob in moves:
        parts.append("{} {}->{}".format(knob, moves[knob][0], moves[knob][1]))
    return ", ".join(parts)


def audit_report(path, judge, out_file):
    report = json.load(open(path))
    rows = []
    for seed in report["runs"]["council"]:
        designs = report["runs"]["council"][seed]["designs"]
        for round_ in report["runs"]["council"][seed]["rounds"]:
            probes = round_.get("probes") or {}
            for name in round_.get("moves", {}):
                member = name.split(":")[-1] if name.startswith("repair:") else name
                if member not in MEMBERS:
                    continue
                reasoning = (round_.get("proposals", {}).get(name) or {}).get("reasoning")
                if not reasoning:
                    for design in designs:
                        if design["round"] == round_["round"] and design["source"] == "council:" + name and design.get("hypothesis"):
                            reasoning = design["hypothesis"]
                if not reasoning:
                    continue
                if isinstance(reasoning, list):
                    reasoning = " ".join(reasoning)
                state = {"sheet": json.dumps(round_.get("sheet") or {})[:2500],
                         "move": move_text(round_["moves"][name]), "reasoning": str(reasoning)[:1200]}
                answer = judge.decide(state, QUESTION)
                row = {"report": os.path.basename(path), "seed": seed, "round": round_["round"], "member": member, "name": name,
                       "probability": float(answer.get("grounded", {}).get("noul", -1)),
                       "gain": probes.get("parts", {}).get(name), "refused": name in (probes.get("refused") or {}),
                       "incumbent": probes.get("incumbent"), "move": state["move"], "reasoning": state["reasoning"][:200]}
                rows.append(row)
                out_file.write(json.dumps(row) + "\n")
                out_file.flush()
    return rows


def summary(rows):
    bins = [("< 0.3", 0.0, 0.3), ("0.3-0.5", 0.3, 0.5), ("0.5-0.7", 0.5, 0.7), (">= 0.7", 0.7, 1.01)]
    print("{} proposals audited".format(len(rows)))
    print("{:8s} {:>4s} {:>8s} {:>8s} {:>10s}".format("grounded", "n", "gained", "refused", "mean gain"))
    for label, low, high in bins:
        group = [row for row in rows if low <= row["probability"] < high]
        if not group:
            continue
        measured = [row for row in group if row["gain"] is not None]
        gained = sum(1 for row in measured if row["gain"] > 0 and not row["refused"])
        refused = sum(1 for row in group if row["refused"])
        mean_gain = sum(row["gain"] for row in measured) / len(measured) if measured else 0.0
        print("{:8s} {:4d} {:7.0%} {:7.0%} {:+10.4f}".format(label, len(group), gained / max(1, len(measured)), refused / len(group), mean_gain))
    print("\nper member: n, mean probability, gained")
    for member in MEMBERS:
        group = [row for row in rows if row["member"] == member]
        if not group:
            continue
        measured = [row for row in group if row["gain"] is not None]
        gained = sum(1 for row in measured if row["gain"] > 0 and not row["refused"])
        print("  {:12s} n={:3d}  p={:.2f}  gained={:.0%}".format(member, len(group), sum(r["probability"] for r in group) / len(group), gained / max(1, len(measured))))


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    judge = decider()
    all_rows = []
    with open(OUT, "a") as out_file:
        for path in sys.argv[1:]:
            all_rows.extend(audit_report(path, judge, out_file))
    summary(all_rows)
