"""CHIA's own CIRCT issue loop, assess stage, re-run under the blocks: tracked and gated.

The stage's prompt is CHIA's own (examples/circt_issue_solver/prompts: system.md and assess.md),
the issues are the sixteen of the paper's Table 5, the model node is CHIA's Vertex node under
CHIA's profiler. Two passes over the same issues: plain (every issue gets the assess turn) and
gated (the CHIA call gate asks the decider one question first, "is this a defect report?", and
the turn is skipped when it says no). Both passes leave a profiler log for
`chia viz-profile --format spend`; the verdicts are compared with the paper's.

Runs in an environment with `chialoops` installed and the blocks copied into it, one process per
pass so each pass has its own Ray and its own profiler collector:
    GOOGLE_CLOUD_PROJECT=... TYPESAFE_API_KEY=... python circt_assess.py <out_dir> plain|gated [threshold]
"""

import json
import os
import re
import sys
import urllib.request
from string import Template

import time

import ray
from chia.base.ChiaFunction import get
from chia.models.call_gate import CallGate, gate_and_log, log_outcome
from chia.models.decider import JevDecider
from chia.models.vertex import RateLimitError, VertexGeminiLLM
from chia.trace.profiler import get_collector, start_collector, stop_collector

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "chia", "examples", "circt_issue_solver", "prompts")
PAPER = {2266: "not a bug", 2669: "not a bug", 4354: "bug", 4396: "not a bug", 4649: "fix unclear",
         5626: "fix unclear", 5789: "bug", 6226: "bug", 6740: "bug", 7127: "not a bug", 7388: "bug",
         7531: "fix unclear", 7949: "bug", 8508: "fix unclear", 10104: "bug", 10571: "not a bug"}
CUTOFF = "2026-06-01"
MODEL = os.environ.get("ASSESS_MODEL", "gemini-3.1-pro-preview")
GATE_QUESTION = {"assess": {
    "instructions": "Does this issue report a defect: existing functionality behaving wrongly, rather than a feature request, a question, or a usage problem?",
    "criteria": {"true": "a concrete wrong behaviour of existing functionality is reported",
                 "false": "it asks for new behaviour, asks a question, or describes a usage or environment problem"}}}


def fetch(url):
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "chia-hackathon-assess"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def issue_markdown(number):
    """The issue as the loop's prompt receives it: title, labels, body, then the comments written
    before the study."""
    issue = fetch("https://api.github.com/repos/llvm/circt/issues/{}".format(number))
    comments = fetch("https://api.github.com/repos/llvm/circt/issues/{}/comments?per_page=100".format(number))
    lines = ["# Issue #{}: {}".format(number, issue["title"]), "Labels: " + ", ".join(label["name"] for label in issue["labels"]), "",
             (issue["body"] or "")[:6000], ""]
    for comment in comments:
        if comment["created_at"] < CUTOFF:
            lines.append("## Comment by {}".format(comment["user"]["login"]))
            lines.append((comment["body"] or "")[:2000])
            lines.append("")
    return "\n".join(lines)


def decision(text):
    """CHIA's own footer parser (examples/circt_issue_solver/issue_task.py, _assess_decision)."""
    verdicts = re.findall(r"(?im)^\s*DECISION:\s*(CLEAR|UNCLEAR|NOT[_ ]?A[_ ]?BUG)\b", text or "")
    verdict = verdicts[-1].upper() if verdicts else "CLEAR"
    if verdict == "CLEAR":
        return "bug"
    if verdict.startswith("NOT"):
        return "not a bug"
    return "fix unclear"


def run_pass(name, issues, llm, template, gate, out_dir, rows):
    log_dir = os.path.join(out_dir, name)
    ray.init(resources={"vertex_creds": 2}, num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    start_collector(log_dir=log_dir)
    answers = {}
    pending = {}
    for number in issues:
        state = {"issue": issues[number][:4000]}
        verdict = {"probability": None, "consult": True}
        if gate is not None:
            verdict = gate_and_log(gate, state, GATE_QUESTION, label="issue {}".format(number))["assess"]
        if verdict["consult"]:
            pending[number] = llm.prompt.chia_remote(llm, template.safe_substitute(issue=issues[number]))
        else:
            log_outcome("issue {}".format(number), "assess", verdict, made=False, outcome="skipped by the gate")
            rows.append({"pass": name, "issue": number, "gate_probability": verdict["probability"], "assessed": False,
                         "verdict": "not a bug (gate)", "paper": PAPER[number]})
        if verdict["consult"]:
            rows.append({"pass": name, "issue": number, "gate_probability": verdict["probability"], "assessed": True,
                         "verdict": None, "paper": PAPER[number]})
    for number in pending:
        # CHIA's Vertex node raises on a 429 at once; the replica waits and asks again.
        for attempt in range(6):
            try:
                answer = get(pending[number])
                break
            except RateLimitError:
                time.sleep(30 * (attempt + 1))
                pending[number] = llm.prompt.chia_remote(llm, template.safe_substitute(issue=issues[number]))
        verdict = decision(answer.result) if answer.success else "no answer"
        answers[number] = {"success": answer.success, "text": answer.result}
        for row in rows:
            if row["pass"] == name and row["issue"] == number:
                row["verdict"] = verdict
                row["success"] = answer.success
        if gate is not None:
            log_outcome("issue {}".format(number), "assess", {"probability": None, "consult": True}, made=True, outcome=verdict)
    # The profiler records fire-and-forget; a blocking call on the collector lands every event
    # already sent before the actor is stopped (otherwise the last ones are lost).
    ray.get(get_collector().get_events.remote())
    stop_collector()
    ray.shutdown()
    json.dump(answers, open(os.path.join(out_dir, "answers_{}.json".format(name)), "w"), indent=1)
    return log_dir


if __name__ == "__main__":
    out_dir = sys.argv[1]
    which = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.65
    os.makedirs(out_dir, exist_ok=True)
    # Their prompts offer MCP tools on a CIRCT checkout. This replica has none: told the tool
    # exists, Gemini attempts the call, Vertex answers MALFORMED_FUNCTION_CALL and CHIA's node
    # returns an empty success for 14 of 16 issues. So the replica's system prompt is one line and
    # the assess prompt's paragraph offering the tool is removed; everything else is theirs.
    system_text = ("You are a senior compiler engineer fluent in MLIR and the CIRCT project (github.com/llvm/circt). "
                   "No tools and no code checkout are available in this run: decide from the issue text alone.")
    assess_text = open(os.path.join(PROMPTS, "assess.md")).read()
    assess_text = re.sub(r"You have access to a circt_bash MCP tool.*?Do not edit anything — this is analysis only\.\n", "", assess_text, flags=re.S)
    template = Template(assess_text)
    issues_path = os.path.join(out_dir, "issues.json")
    if os.path.exists(issues_path):
        issues = {int(key): value for key, value in json.load(open(issues_path)).items()}
    else:
        issues = {}
        for number in PAPER:
            issues[number] = issue_markdown(number)
        json.dump(issues, open(issues_path, "w"), indent=1)
    llm = VertexGeminiLLM(model=MODEL, system_message=system_text, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"), logging_level=40)
    rows = []
    gate = None
    if which.startswith("gated"):     # a pass named gated, gated2, ... applies the gate
        gate = CallGate(JevDecider(), threshold=threshold, mode="apply")
    run_pass(which, issues, llm, template, gate, out_dir, rows)
    with open(os.path.join(out_dir, "verdicts_{}.jsonl".format(which)), "w") as out_file:
        for row in rows:
            out_file.write(json.dumps(row) + "\n")
    agree = sum(1 for row in rows if row["verdict"] and row["verdict"].split(" (")[0] == row["paper"])
    assessed = sum(1 for row in rows if row["assessed"])
    failed = sum(1 for row in rows if row["assessed"] and not row.get("success", True))
    print("{}: {} assess turns made ({} failed), verdict agrees with the paper on {} of {}".format(which, assessed, failed, agree, len(rows)))
    print("log: {}/{} ; run `chia viz-profile --format spend` on it".format(out_dir, which))
