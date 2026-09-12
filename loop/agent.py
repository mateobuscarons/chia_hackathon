"""The LLM agent, one code path for the four LLM arms, two switches:

  memory_slot the prompt carries the memory's digest of the nearest remembered
              cases (loop.memory.digest) with this design in its copyable slot:
              "nearest" (the closest workload's best) or "pooled" (the best across
              the memory); None, the agent sees only the problem, the workloads
              and its own results.
  use_gp      the LLM proposes CANDIDATES designs per round, most promising first;
              once three designs are measured, a Gaussian process fit on this run
              picks the ones to simulate by expected improvement (loop.bo); before
              that, the LLM's top picks run. Off, the LLM proposes exactly the
              designs that run.

  llm_direct = none; memory = digest (nearest) + GP; memory_pooled = digest (pooled) + GP.

A proposal must be a real, in-budget, unmeasured design; otherwise the LLM gets
one retry with the rejected designs listed, and an empty slot goes to a
deterministic one-knob perturbation of the incumbent. Never a random design.

Every LLM call goes through `ask`: Gemini on Vertex, JSON in, parsed JSON out.
Under CHIA (LOOP_DISPATCH=chia) the call goes through chia.models.vertex so the
profiler records it; that layer forwards only the prompt, so temperature and
thinking budget are the model defaults there.
"""

import fcntl
import json
import os
import time

from google import genai

from loop import bo, memory as memory_module
from loop.configs import KNOB_PRIORITY, SEARCH_SPACE, knobs_changed, typed_knobs

# ---------------------------------------------------------------- the LLM call ----

GCP_PROJECT = "project-c23a6080-f5d0-4871-9cb"
# gemini-2.5-flash is the strongest flash this project can call (Gemini 3 is not enabled on it);
# ANALYST_MODEL=gemini-2.5-pro is the only Pro.
MODEL = os.environ.get("ANALYST_MODEL", "gemini-2.5-flash")
USAGE_LOG = "loop/llm_usage.json"

# USD per 1M tokens (input, output), prompts under 200k tokens. Thinking bills as output.
PRICES = {"gemini-2.5-flash": (0.30, 2.50),
          "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-2.5-pro": (1.25, 10.00)}

TEMPERATURE = 0.7
THINKING_BUDGET = 4096

# System instruction: persona plus the one constraint that matters.
ROLE = ("You are a computer architect running simulation experiments. You propose "
        "and explain; a simulator measures everything, so state only what one "
        "experiment can test. Answer only with the JSON that is asked for.")

_client = None
_chia_node = None


def client():
    global _client
    if _client is None:
        # 120 s timeout per call: a hung request must fail loudly, not stall the loop.
        _client = genai.Client(vertexai=True, project=GCP_PROJECT, location="us-central1",
                               http_options={"timeout": 120_000})
    return _client


def chia_node():
    global _chia_node
    if _chia_node is None:
        from loop.chia_nodes import AnalystNode
        _chia_node = AnalystNode(model=MODEL, project=GCP_PROJECT)
    return _chia_node


def ask(prompt):
    """One call, parsed JSON back. Malformed or empty answers are retried a few times."""
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        # CHIA's layer raises on rate limits (429), truncation and malformed JSON
        # without waiting; with many runs asking at once, wait and try again.
        node = chia_node()
        delays = [15, 30, 60, 120, 240, 300]
        for attempt in range(len(delays) + 1):
            try:
                answer = node.ask(ROLE + "\n\n" + prompt)
            except Exception as error:
                if attempt == len(delays):
                    raise RuntimeError("Gemini failed {} times: {}".format(attempt + 1, repr(error)[:200]))
                print("  [gemini] {} - waiting {}s".format(repr(error)[:100], delays[attempt]), flush=True)
                time.sleep(delays[attempt])
                continue
            tokens = node.llm._last_metadata
            log_cost(tokens.get("input_tokens", 0), tokens.get("output_tokens", 0))
            return answer
    for attempt in range(5):
        started = time.time()
        response = generate_with_backoff(prompt)
        log_cost(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
        print("  [gemini] {:.1f}s".format(time.time() - started), flush=True)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("  [gemini] bad JSON, retrying. Tail:", repr(response.text[-200:]), flush=True)
    raise RuntimeError("Gemini returned malformed JSON five times")


def generate_with_backoff(prompt):
    """Rate limits (429) and server hiccups (5xx) are transient: wait and retry."""
    from google.genai import errors
    delays = [15, 30, 60, 120, 180, 240]
    for attempt in range(len(delays) + 1):
        try:
            return client().models.generate_content(
                model=MODEL, contents=prompt,
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
    """Append this call's tokens and cost to the running usage log."""
    price_in, price_out = PRICES[MODEL]
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    # Parallel runs write this file at the same time: hold a lock while updating.
    with open(USAGE_LOG + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if os.path.exists(USAGE_LOG):
            with open(USAGE_LOG) as log_file:
                log = json.load(log_file)
        else:
            log = {"total_cost_usd": 0.0, "calls": []}
        log["total_cost_usd"] += cost
        log["calls"].append({"model": MODEL, "in": input_tokens, "out": output_tokens, "cost_usd": round(cost, 6)})
        with open(USAGE_LOG, "w") as log_file:
            json.dump(log, log_file, indent=2)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def as_list(answer):
    """The schemas ask for a list; a single object is accepted as a list of one."""
    if isinstance(answer, dict):
        return [answer]
    if isinstance(answer, list):
        return answer
    return []


# ---------------------------------------------------------------- prompt ----

PICK_SCHEMA = """## Output format
A JSON list of objects, one per design, with EXACTLY these keys:
- "hypothesis": one sentence: which bottleneck this design tests and why it should raise the objective.
- "knobs": the full design, one allowed value per knob (every knob listed above).

For the shape only (fictional knob names):
[{"hypothesis": "The mid-level cache is capacity-bound, so a larger widget_size should cut its misses",
  "knobs": {"widget_size": 2048, "widget_ways": 8, "gadget_policy": "plain"}}]
"""


# ---------------------------------------------------------------- prompt ----

def section_problem(problem):
    lines = ["## Problem", problem["chip_text"],
             "Maximize {}: the geometric mean of IPC over the workloads below. A design must fit the area budget: "
             "L2 capacity + LLC capacity (sets x ways x 64 bytes each) <= {} KB.".format(problem["objective"], problem["area_budget_kb"]),
             "Knobs and allowed values:"]
    for knob in SEARCH_SPACE:
        lines.append("- {}: {}".format(knob, json.dumps(SEARCH_SPACE[knob])))
    lines.append("The stock chip (design D0): " + json.dumps(problem["stock"]))
    return "\n".join(lines)


def section_workloads(problem):
    lines = ["## Workloads (descriptors profiled from the traces, read against the stock chip)",
             problem_legend()]
    for workload in problem["workloads"]:
        rounded = {}
        for name, value in problem["descriptors"][workload].items():
            rounded[name] = round(value, 3)
        lines.append("- {}: {}".format(workload, json.dumps(rounded)))
    return "\n".join(lines)


def problem_legend():
    from loop.champsim_problem import DESCRIPTOR_LEGEND
    return DESCRIPTOR_LEGEND


def format_table(history, problem):
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


def assemble_prompt(problem, history, memory, retrieval, how_many, slot="pooled"):
    """The sections in order, each generated from data. An empty memory produces
    exactly the plain agent's prompt."""
    sections = [section_problem(problem), section_workloads(problem)]
    memory_text = memory_module.digest(memory, problem, retrieval, slot)
    if memory_text:
        sections.append(memory_text)
    sections.append("## Simulation results so far (one row per design; D0 is the stock chip)\n" + format_table(history, problem))
    sections.append(section_task(how_many, problem["objective"]))
    return "\n\n".join(sections)


# ---------------------------------------------------------------- picks ----

def valid_proposal(proposal):
    return isinstance(proposal, dict) and isinstance(proposal.get("knobs"), dict) and isinstance(proposal.get("hypothesis"), str)


def incumbent_of(history, objective):
    best = history[0]
    for entry in history:
        if entry["metrics"][objective] > best["metrics"][objective]:
            best = entry
    return best


def perturbation(problem, history, taken_names):
    """The fallback when the LLM cannot name a new design: the incumbent with one
    knob moved to a value this run has not tried on that knob, knobs in
    KNOB_PRIORITY. Deterministic; never a random design."""
    incumbent = incumbent_of(history, problem["objective"])
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
            if name in taken_names:
                continue
            return {"name": name, "knobs": design, "source": "perturbation",
                    "hypothesis": "fallback: {} {} -> {} on D{}".format(knob, incumbent["knobs"][knob], value, incumbent["index"])}
    return None


CANDIDATES = 8            # proposals per round when the GP selects
GP_FROM_DESIGNS = 3       # the GP selects once this many designs are measured


def propose(problem, prompt, how_many, taken):
    """Ask once, retry once with the rejected designs listed. Returns the valid,
    unmeasured, in-budget proposals in the LLM's order, and what was rejected."""
    accepted = []
    rejected = []
    try:
        proposals = as_list(ask(prompt))
    except RuntimeError as error:
        # The LLM is down for this round: the fallback fills the slots, the run goes on.
        return accepted, [], ["LLM failed: " + repr(error)[:160]]
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
            accepted.append({"name": name, "knobs": knobs, "source": "pick", "hypothesis": proposal["hypothesis"]})
        if len(accepted) == how_many or attempt == 1 or len(rejected) == 0:
            break
        retry_prompt = prompt + "\n\n## Already tested or not allowed (do not propose these again)\n" + "\n".join(rejected)
        try:
            proposals = as_list(ask(retry_prompt))
        except RuntimeError as error:
            rejected.append("LLM failed on the retry: " + repr(error)[:160])
            break
    return accepted, proposals, rejected


def select_by_gp(problem, history, accepted, per_round, seed, round_number):
    """The GP fit on this run's designs picks `per_round` of the LLM's proposals by
    expected improvement. Returns the chosen proposals (source "pick+gp") and the
    GP's predicted mean per proposal."""
    objective = problem["objective"]
    model = bo.fit(history, SEARCH_SPACE, objective, honest_std=False, reference=history[0]["metrics"][objective])
    candidates = {}
    for proposal in accepted:
        candidates[proposal["name"]] = proposal["knobs"]
    knobs_list = []
    for proposal in accepted:
        knobs_list.append(proposal["knobs"])
    means, stds = bo.predict_many(model, knobs_list)
    predicted = {}
    for index, proposal in enumerate(accepted):
        predicted[proposal["name"]] = round(float(means[index]), 4)
    chosen_names = bo.pick_by_expected_improvement(candidates, model, history, objective, per_round, seed * 1000 + round_number)
    chosen = []
    for proposal in accepted:
        if proposal["name"] in chosen_names:
            selected = dict(proposal)
            selected["source"] = "pick+gp"
            chosen.append(selected)
    return chosen, predicted


def run_agent(problem, rounds, per_round, tag, memory_path, memory_slot, use_gp, seed=0):
    """Returns {"designs": history, "rounds": round logs}."""
    objective = problem["objective"]
    memory = memory_module.empty()
    retrieval = None
    if memory_slot is not None:
        memory = memory_module.load(memory_path)
        retrieval = memory_module.retrieve(memory, problem)
    stock_metrics = problem["evaluate"](problem["stock"])
    history = [{"index": 0, "round": 0, "name": problem["name_of"](problem["stock"]), "knobs": problem["stock"],
                "metrics": stock_metrics, "source": "stock", "hypothesis": None}]
    taken = {history[0]["name"]}
    round_logs = []
    for round_number in range(1, rounds + 1):
        how_many = per_round
        gp_selects = use_gp and len(history) >= GP_FROM_DESIGNS
        if use_gp:
            how_many = CANDIDATES
        prompt = assemble_prompt(problem, history, memory, retrieval, how_many, memory_slot)
        accepted, proposals, rejected = propose(problem, prompt, how_many, taken)
        predicted = None
        if gp_selects and len(accepted) > per_round:
            chosen, predicted = select_by_gp(problem, history, accepted, per_round, seed, round_number)
        else:
            chosen = accepted[:per_round]
        while len(chosen) < per_round:
            fallback = perturbation(problem, history, taken)
            if fallback is None:
                break
            taken.add(fallback["name"])
            chosen.append(fallback)
        knobs_list = []
        for design in chosen:
            knobs_list.append(design["knobs"])
        metrics_list = problem["evaluate_many"](knobs_list)
        for design, metrics in zip(chosen, metrics_list):
            entry = dict(design)
            entry["index"] = len(history)
            entry["round"] = round_number
            entry["metrics"] = metrics
            history.append(entry)
            print("[{}] round {} | D{} | {}={:.4f} | {}".format(tag, round_number, entry["index"], objective,
                                                                metrics[objective], entry["source"]), flush=True)
        candidates_logged = []
        for proposal in accepted:
            candidates_logged.append({"name": proposal["name"], "knobs": proposal["knobs"], "hypothesis": proposal["hypothesis"]})
        round_logs.append({"round": round_number, "prompt": prompt, "candidates": candidates_logged, "gp_predicted": predicted,
                           "rejected": rejected, "chosen": [design["name"] for design in chosen]})
    return {"designs": history, "rounds": round_logs}
