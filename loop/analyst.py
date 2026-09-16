"""The model: Gemini on Vertex, JSON in and JSON out, and the prompt it reads.

One call serves the memory build's searcher (`loop.search.llm_search`):
`designs_for_round` returns the designs to simulate next. Everything the model can get wrong is handled here and nowhere else - rate
limits, malformed answers, designs that are not real, designs already measured -
so a search reading this module never has to know that it is talking to a language model.

A proposal must be a real, in-budget, unmeasured design. If it is not, the model
gets one retry with the rejected designs listed; a slot still empty is filled by a
deterministic one-knob perturbation of the incumbent. Never a random design.
"""

import fcntl
import json
import os
import time

from google import genai

from loop.space import KNOB_PRIORITY, SEARCH_SPACE, knobs_changed, typed_knobs

GCP_PROJECT = "project-c23a6080-f5d0-4871-9cb"
# ANALYST_MODEL picks the model, ANALYST_LOCATION the Vertex endpoint serving it. The
# 2.5 models serve from us-central1; every Gemini 3.x model 404s there and serves from
# "global" only, so a 3.x model needs ANALYST_LOCATION=global.
MODEL = os.environ.get("ANALYST_MODEL", "gemini-3.1-pro-preview")
LOCATION = os.environ.get("ANALYST_LOCATION", "global")
USAGE_LOG = "results/llm_usage.json"

# USD per 1M tokens (input, output), prompts under 200k tokens. Thinking bills as output.
PRICES = {"gemini-2.5-flash": (0.30, 2.50),
          "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-2.5-pro": (1.25, 10.00),
          "gemini-3.1-pro-preview": (2.00, 12.00)}

TEMPERATURE = 0.7
THINKING_BUDGET = 4096

# System instruction: persona plus the one constraint that matters.
ROLE = ("You are a computer architect running simulation experiments. You propose "
        "and explain; a simulator measures everything, so state only what one "
        "experiment can test. Answer only with the JSON that is asked for.")

_client = None


# ---------------------------------------------------------------- the call ----

def client():
    global _client
    if _client is None:
        # 120 s timeout per call: a hung request must fail loudly, not stall the loop.
        _client = genai.Client(vertexai=True, project=GCP_PROJECT, location=LOCATION,
                               http_options={"timeout": 120_000})
    return _client


def ask(prompt_text):
    """One call, parsed JSON back. Malformed or empty answers are retried a few times."""
    for attempt in range(5):
        started = time.time()
        response = generate_with_backoff(prompt_text)
        # Thinking bills at the output rate but is reported apart from the answer, so
        # counting `candidates` alone understated every thinking model's cost.
        usage = response.usage_metadata
        log_cost(usage.prompt_token_count or 0,
                 (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
        print("  [gemini] {:.1f}s".format(time.time() - started), flush=True)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("  [gemini] bad JSON, retrying. Tail:", repr(response.text[-200:]), flush=True)
    raise RuntimeError("Gemini returned malformed JSON five times")


def generate_with_backoff(prompt_text):
    """Rate limits (429) and server hiccups (5xx) are transient: wait and retry."""
    from google.genai import errors
    delays = [15, 30, 60, 120, 180, 240]
    for attempt in range(len(delays) + 1):
        try:
            return client().models.generate_content(
                model=MODEL, contents=prompt_text,
                config={"system_instruction": ROLE,
                        "response_mime_type": "application/json", "max_output_tokens": 16384,
                        "temperature": TEMPERATURE,
                        "thinking_config": {"thinking_budget": THINKING_BUDGET}},
            )
        except errors.APIError as error:
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt == len(delays):
                raise
            print("  [gemini] {} - waiting {}s".format(error.code, delays[attempt]), flush=True)
            time.sleep(delays[attempt])


def log_cost(input_tokens, output_tokens):
    """Add this call to the running totals. Totals only, not one row per call: a
    cell run makes hundreds of calls and nothing ever reads them back."""
    price_in, price_out = PRICES[MODEL]
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
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


# ---------------------------------------------------------------- the prompt ----

PICK_SCHEMA = """## Output format
A JSON list of objects, one per design, with EXACTLY these keys:
- "hypothesis": one sentence: which bottleneck this design tests and why it should raise the objective.
- "knobs": the full design, one allowed value per knob (every knob listed above).

For the shape only (fictional knob names):
[{"hypothesis": "The mid-level cache is capacity-bound, so a larger widget_size should cut its misses",
  "knobs": {"widget_size": 2048, "widget_ways": 8, "gadget_policy": "plain"}}]
"""


def prompt(problem, history, how_many):
    """The sections in order, each generated from data: the chip and its knobs,
    every design measured so far, the task. Everything the model is told about the
    workloads it has measured itself - there is no prior."""
    sections = [section_problem(problem),
                "## Simulation results so far (one row per design; D0 is the stock chip)\n"
                + results_table(history, problem),
                section_task(how_many, problem["objective"])]
    return "\n\n".join(sections)


def section_problem(problem):
    lines = ["## Problem", problem["chip_text"],
             "Maximize {}: the geometric mean of IPC over {}. A design must fit the area budget: "
             "L2 capacity + LLC capacity (sets x ways x 64 bytes each) <= {} KB.".format(
                 problem["objective"], " + ".join(problem["workloads"]), problem["area_budget_kb"]),
             "Knobs and allowed values:"]
    for knob in SEARCH_SPACE:
        lines.append("- {}: {}".format(knob, json.dumps(SEARCH_SPACE[knob])))
    lines.append("The stock chip (design D0): " + json.dumps(problem["stock"]))
    return "\n".join(lines)


def results_table(history, problem):
    """One row per design: index, who chose it, the knobs it changes from stock,
    the suite objective and the per-workload IPC and LLC misses, then the
    hypothesis it tested."""
    lines = []
    for entry in history:
        changed = knobs_changed(entry["knobs"], problem["stock"])
        if len(changed) == 0:
            design_text = "stock"
        else:
            parts = []
            for knob in changed:
                parts.append("{}={}".format(knob, changed[knob][1]))
            design_text = ", ".join(parts)
        cells = ["{}={:.4f}".format(problem["objective"], entry["metrics"][problem["objective"]])]
        for workload in problem["workloads"]:
            cells.append("{}: ipc {:.4f}, LLC mpki {:.2f}, LLC hit {:.2f}".format(
                workload, entry["metrics"][workload + ":ipc"], entry["metrics"][workload + ":LLC_mpki"],
                entry["metrics"][workload + ":LLC_hit_ratio"]))
        lines.append("D{} [{}] {} | {}".format(entry["index"], entry["source"], design_text, " | ".join(cells)))
        if entry.get("hypothesis"):
            lines.append("    {}".format(entry["hypothesis"]))
    return "\n".join(lines)


def section_task(how_many, objective):
    return """## Task
You are tuning this cache hierarchy with a small simulation budget. Propose the {n} untested
in-budget design(s) most likely to raise {objective}, most promising first, each with the
hypothesis it tests.

{schema}""".format(n=how_many, objective=objective, schema=PICK_SCHEMA)


# ---------------------------------------------------------------- the designs ----

def designs_for_round(problem, history, how_many, taken):
    """`how_many` designs to simulate next, and the log of how they were chosen.

    The model's own picks come first, in its order, validated and unmeasured. A
    slot it could not fill goes to a perturbation, so a round always spends its
    full budget and the arms stay comparable."""
    prompt_text = prompt(problem, history, how_many)
    chosen, rejected = propose(problem, prompt_text, how_many, taken)
    while len(chosen) < how_many:
        fallback = perturbation(problem, history, taken)
        if fallback is None:
            break
        taken.add(fallback["name"])
        chosen.append(fallback)
    log = {"prompt": prompt_text, "rejected": rejected,
           "chosen": [design["name"] for design in chosen]}
    return chosen, log


def propose(problem, prompt_text, how_many, taken):
    """Ask once, retry once with the rejected designs listed. Returns the valid,
    unmeasured, in-budget proposals in the model's order, and what was rejected."""
    accepted = []
    rejected = []
    try:
        proposals = as_list(ask(prompt_text))
    except RuntimeError as error:
        # The model is down for this round: the fallback fills the slots, the run goes on.
        return accepted, ["LLM failed: " + repr(error)[:160]]
    for attempt in range(2):
        for proposal in proposals:
            if len(accepted) == how_many:
                break
            if not valid_proposal(proposal) or not problem["is_candidate"](proposal["knobs"]):
                rejected.append(json.dumps(proposal.get("knobs") if isinstance(proposal, dict) else proposal))
                continue
            knobs = typed_knobs(proposal["knobs"])
            name = problem["name_of"](knobs)
            if name in taken:
                rejected.append(name)
                continue
            taken.add(name)
            accepted.append({"name": name, "knobs": knobs, "source": "pick",
                             "hypothesis": proposal["hypothesis"]})
        if len(accepted) == how_many or attempt == 1 or len(rejected) == 0:
            break
        retry_prompt = prompt_text + "\n\n## Already tested or not allowed (do not propose these again)\n" + "\n".join(rejected)
        try:
            proposals = as_list(ask(retry_prompt))
        except RuntimeError as error:
            rejected.append("LLM failed on the retry: " + repr(error)[:160])
            break
    return accepted, rejected


def as_list(answer):
    """The schema asks for a list; a single object is accepted as a list of one."""
    if isinstance(answer, dict):
        return [answer]
    if isinstance(answer, list):
        return answer
    return []


def valid_proposal(proposal):
    return (isinstance(proposal, dict) and isinstance(proposal.get("knobs"), dict)
            and isinstance(proposal.get("hypothesis"), str))


def perturbation(problem, history, taken):
    """The fallback: the incumbent with one knob moved to a value this run has not
    tried on that knob, knobs in KNOB_PRIORITY. Deterministic."""
    incumbent = history[0]
    for entry in history:
        if entry["metrics"][problem["objective"]] > incumbent["metrics"][problem["objective"]]:
            incumbent = entry
    tried = {}
    for entry in history:
        for knob in SEARCH_SPACE:
            tried.setdefault(knob, set()).add(str(entry["knobs"][knob]))
    for knob in KNOB_PRIORITY:
        for value in SEARCH_SPACE[knob]:
            if str(value) in tried[knob]:
                continue
            design = dict(incumbent["knobs"])
            design[knob] = value
            if not problem["is_candidate"](design):
                continue
            design = typed_knobs(design)
            name = problem["name_of"](design)
            if name in taken:
                continue
            return {"name": name, "knobs": design, "source": "perturbation",
                    "hypothesis": "fallback: {} {} -> {} on D{}".format(
                        knob, incumbent["knobs"][knob], value, incumbent["index"])}
    return None
