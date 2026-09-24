"""The model: Gemini on Vertex. JSON in, JSON out. Everything the model can get wrong on the wire
is handled here - rate limits, malformed answers, the cost log - so the council never has to
know that it is talking to a language model.
"""

import fcntl
import json
import os
import sys
import time

from chia.base.ChiaFunction import get


GCP_PROJECT = os.environ.get("GCP_PROJECT", "project-c23a6080-f5d0-4871-9cb")   # the project billed for the calls
# ANALYST_MODEL picks the model, ANALYST_LOCATION the Vertex endpoint serving it. The
# 2.5 models serve from us-central1; every Gemini 3.x model 404s there and serves from
# "global" only, so a 3.x model needs ANALYST_LOCATION=global.
MODEL = os.environ.get("ANALYST_MODEL", "gemini-3.1-pro-preview")
LOCATION = os.environ.get("ANALYST_LOCATION", "global")
USAGE_LOG = "results/llm_usage.json"
CALLS_LOG = "results/llm_calls.jsonl"   # one row per call, with the run's tag: what each run cost
RUN_TAG = ""                            # set by the search at the start of a run (one process per seed)
ROUND = 0                               # set by the search at the start of every round

TEMPERATURE = 0.7
# THINKING_BUDGET: thinking tokens per call (they bill as output, at six times the input rate, and
# were two thirds of the spend at 4096). SPECIALIST_THINKING lets a specialist's call think less
# than the analyst's. In gate "apply" mode the gate's difficulty tier sets the budget instead.
THINKING_BUDGET = int(os.environ.get("THINKING_BUDGET", "4096"))
SPECIALIST_THINKING = int(os.environ.get("SPECIALIST_THINKING", str(THINKING_BUDGET)))
SPECIALISTS = ("prefetch", "geometry", "replacement", "concurrency")

# CONTEXT_GATE: "shadow" records what the context gate (a CHIA block under council_loop) would
# drop from every prompt and what the call then measured; "apply" sends the gated prompt and
# sets the thinking budget by the gate's difficulty; unset, no gate. A gate failure never stops
# a call: the prompt goes ungated and the failure is printed.
CONTEXT_GATE = os.environ.get("CONTEXT_GATE", "")
# The gates and the decider are CHIA blocks (council_loop/chia_blocks, CHIA's own layout). With
# CHIA installed they import from it; without, the blocks folder stands in for the package.
BLOCKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "council_loop", "chia_blocks")
if BLOCKS not in sys.path:
    sys.path.append(BLOCKS)
GATE_LOG = "results/context_gate.jsonl"
GATE_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.5"))                    # specialists
ANALYST_GATE_THRESHOLD = float(os.environ.get("ANALYST_GATE_THRESHOLD", "0.3"))    # the analyst reads more: stricter
# Instruction-bearing sections are never dropped; the rest is judged by its description below.
ALWAYS_KEPT = ("preamble", "Role", "Task", "The search is stalled", "Once more")
# In apply mode the gate's difficulty sets the thinking budget: a specialist call rated below "hard"
# thinks 1024 tokens (the median call uses about 1100); the analyst keeps its budget whatever the rating.
SPECIALIST_TIERS = {"easy": 1024, "medium": 1024, "hard": SPECIALIST_THINKING}
ANALYST_TIERS = {"easy": THINKING_BUDGET, "medium": THINKING_BUDGET, "hard": THINKING_BUDGET}
DESCRIBE = [   # (section name prefix, what the section holds): what the decider judges
    ("Principles", "general rules of the trade for this concern; the same text every round"),
    ("Chip", "the chip card: core width, clock, memory, process node; the same every round"),
    ("Your knobs and the values available", "the legal values of the knobs this member may move, each shape with its latency and silicon"),
    ("Knobs and the values available", "the legal values of every knob, each shape with its latency and silicon"),
    ("The current design", "the incumbent design's knobs and its measured per-level counters"),
    ("What each workload contributes", "the share of memory time each workload holds"),
    ("The analyst's sheet", "this round's diagnosis: per-level verdicts, the bottleneck with its evidence, findings, failing knobs, cap headroom"),
    ("The design measured last round", "the previous incumbent and how the current one differs from it"),
    ("What your knobs have scored", "the counters this member's knobs act on, for the current design"),
    ("What each value of your knobs has done", "the value ledger: mean IPC with and without each value of this member's knobs"),
    ("What each value of each knob has done", "the value ledger: mean IPC with and without each value of every knob"),
    ("Every move of your concern", "this member's own past moves and their measured gains"),
    ("Every move this run has made", "every move of the run with its measured gain"),
    ("What your concern's designs measured", "per round, the gain each of this member's designs measured against the round's incumbent"),
    ("What each round's wave measured", "per round, every design's gain against that round's incumbent, refusals with their reason"),
    ("The team's recent designs", "the last six designs anyone proposed, with knobs and gains"),
    ("The best designs measured so far", "the three best designs of the run, every knob given"),
    ("What the statistician expects", "the surrogate's five highest predicted designs and its pick"),
    ("Your previous hypothesis", "this member's last prediction and what was then measured"),
]
_decider = None
_gates = {}


def describe(name):
    for prefix, text in DESCRIBE:
        if name.startswith(prefix):
            return text
    return None


def key_from_env_file():
    """The decider's key from `.env` (`jev_api=` or `TYPESAFE_API_KEY=`) when not in the environment."""
    if not os.path.exists(".env"):
        return None
    for line in open(".env"):
        for name in ("TYPESAFE_API_KEY", "jev_api"):
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def decider():
    """The one decider (TypeSafe's Jev) behind every gate of this loop."""
    global _decider
    if _decider is None:
        from chia.models.decider import JevDecider
        key = os.environ.get("TYPESAFE_API_KEY") or key_from_env_file()
        _decider = JevDecider(api_key=key)
    return _decider


def gate(label=None):
    """The context gate for a caller: the analyst's (stricter threshold, its thinking kept) or
    the specialists' (threshold 0.5, thinking capped when the task is not rated hard)."""
    kind = "analyst" if label not in SPECIALISTS else "specialist"
    if kind not in _gates:
        from chia.models.context_gate import ContextGate
        if kind == "analyst":
            _gates[kind] = ContextGate(decider(), threshold=ANALYST_GATE_THRESHOLD, mode=CONTEXT_GATE,
                                       always_kept=ALWAYS_KEPT, describe=describe, tiers=ANALYST_TIERS)
        else:
            _gates[kind] = ContextGate(decider(), threshold=GATE_THRESHOLD, mode=CONTEXT_GATE,
                                       always_kept=ALWAYS_KEPT, describe=describe, tiers=SPECIALIST_TIERS)
    return _gates[kind]


def log_gate(label, gated, measured_tokens, thinking_sent, thoughts_used=0, answer_tokens=0):
    """One row per gated call: the gate's estimate and verdict beside what the model measured."""
    row = {"time": time.time(), "tag": RUN_TAG, "label": label, "mode": gated["mode"], "tokens_before": gated["tokens_before"],
           "tokens_after": gated["tokens_after"], "tokens_measured": measured_tokens,
           "dropped": gated["dropped"], "probabilities": gated["probabilities"], "difficulty": gated["difficulty"],
           "thinking_budget": gated["thinking_budget"], "thinking_sent": thinking_sent,
           "thoughts_used": thoughts_used, "answer_tokens": answer_tokens,
           "decider": getattr(decider(), "last_usage", {})}
    with open(GATE_LOG + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        with open(GATE_LOG, "a") as log_file:
            log_file.write(json.dumps(row) + "\n")
        fcntl.flock(lock_file, fcntl.LOCK_UN)

# System instruction: persona plus the one constraint that matters.
ROLE = ("You are a computer architect running simulation experiments. You propose "
        "and explain; a simulator measures everything, so state only what one "
        "experiment can test. Answer only with the JSON that is asked for.")

_node = None


# ---------------------------------------------------------------- the call ----

def node():
    """The loop's model node: the `VertexGeminiJSON` block, one per process. Every call is a
    CHIA task, so the run's profiler log holds each call's tokens, USD, caller and round."""
    global _node
    if _node is None:
        from chia.models.vertex_json import VertexGeminiJSON
        _node = VertexGeminiJSON(MODEL, system_message=ROLE, project=GCP_PROJECT, location=LOCATION,
                                 temperature=TEMPERATURE, timeout_seconds=120)
    return _node


def ask(prompt_text, label=None):
    """One call, parsed JSON back. Malformed or empty answers are retried a few times, each a
    billed call of its own. `label` names the caller (analyst, a specialist) for the logs."""
    gated = None
    sent = prompt_text
    thinking = SPECIALIST_THINKING if label in SPECIALISTS else THINKING_BUDGET
    if CONTEXT_GATE:
        try:
            from chia.models.context_gate import gate_and_log
            gated = gate_and_log(gate(label), prompt_text, label=label)
            if CONTEXT_GATE == "apply":
                sent = gated["text"]
                thinking = gated["thinking_budget"]
        except Exception as error:
            print("  [gate] failed, prompt sent ungated: " + repr(error)[-120:], flush=True)
    for attempt in range(5):
        started = time.time()
        info = {"label": label, "round": ROUND, "tag": RUN_TAG}
        text, usage = get(node().prompt.chia_remote(node(), sent, thinking, info))
        log_cost(usage, label)
        if gated is not None:
            log_gate(label, gated, usage["input_tokens"], thinking, usage["thinking_tokens"],
                     usage["output_tokens"] - usage["thinking_tokens"])
        print("  [gemini] {:.1f}s".format(time.time() - started), flush=True)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print("  [gemini] bad JSON, retrying. Tail:", repr(text[-200:]), flush=True)
    raise RuntimeError("Gemini returned malformed JSON five times")


def log_cost(usage, label=None):
    """Append one row for this call under the run's tag, so the cost of one run (an A/B arm, a
    seed) can be read back apart from the grand total, and add it to the running totals. The
    tokens and the USD are the node's (`usage`), priced from the spend block's table."""
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    cost = usage["cost_usd"] or 0.0
    row = {"time": time.time(), "tag": RUN_TAG, "round": ROUND, "label": label, "model": MODEL, "input_tokens": input_tokens,
           "output_tokens": output_tokens, "thinking_tokens": usage["thinking_tokens"], "cost_usd": round(cost, 6)}
    with open(CALLS_LOG, "a") as calls_file:
        calls_file.write(json.dumps(row) + "\n")
    # Parallel runs write this file at the same time: hold a lock while updating.
    with open(USAGE_LOG + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        log = {"total_cost_usd": 0.0}
        if os.path.exists(USAGE_LOG):
            with open(USAGE_LOG) as log_file:
                log = json.load(log_file)
        log.pop("calls", None)
        log.setdefault("by_model", {})
        totals = log["by_model"].setdefault(MODEL, {"calls": 0, "in": 0, "out": 0, "cost_usd": 0.0})
        totals["calls"] += 1
        totals["in"] += input_tokens
        totals["out"] += output_tokens
        totals["cost_usd"] = round(totals["cost_usd"] + cost, 6)
        log["total_cost_usd"] = round(log["total_cost_usd"] + cost, 6)
        with open(USAGE_LOG, "w") as log_file:
            json.dump(log, log_file, indent=2)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
