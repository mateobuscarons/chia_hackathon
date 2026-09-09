"""The analyst: every LLM call in the loop lives in this file.

The LLM proposes, explains and distills; it never scores anything. It
sees results as compact text and must answer in strict JSON so the loop
can act on it mechanically.
"""

import fcntl
import json
import os
import time

from google import genai

GCP_PROJECT = "project-c23a6080-f5d0-4871-9cb"
# ANALYST_MODEL=gemini-2.5-pro runs the same loop with a stronger analyst (ablation).
MODEL = os.environ.get("ANALYST_MODEL", "gemini-2.5-flash")
USAGE_LOG = "loop/llm_usage.json"

# USD per 1M tokens (input, output), prompts under 200k tokens. Pro bills its
# thinking tokens as output.
PRICES = {"gemini-2.5-flash": (0.30, 2.50),
          "gemini-2.5-flash-lite": (0.10, 0.40),
          "gemini-2.5-pro": (1.25, 10.00)}

# 120 s timeout per call: a hung request must fail loudly, not stall the loop.
_client = genai.Client(vertexai=True, project=GCP_PROJECT, location="us-central1",
                      http_options={"timeout": 120_000})

# System instruction (persona + the one behavioural constraint that matters).
# Everything task-specific goes in the user prompt, context first, task last.
ROLE = ("You are a computer architect running simulation experiments. You propose "
        "and explain; a simulator measures everything, so state only what one "
        "experiment can test. Answer only with the JSON that is asked for.")

# Temperature and thinking budget per call type. Distillation and re-scoping
# are consistency tasks (numbers, signs, exact schema): low temperature, more
# thinking. The reply needs some diversity: default-ish temperature.
DISTILL_TEMPERATURE = 0.3
HYPOTHESIS_TEMPERATURE = 0.7
THINKING_BUDGET = 2048
DISTILL_THINKING_BUDGET = 8192

# One fictional, deliberately generic example per output type: the guide says
# always show the shape, and one example cannot be copied as content. The knob
# and metric names below do not exist in any search space, so a copied example
# is rejected by the validator.
RULE_SCHEMA = """## Output format
A JSON list of rules. A rule is a JSON object with EXACTLY these keys:
- "text": the rule in one sentence, in your own words.
- "conditions": a list of 1 or 2 clauses, all of which must hold for the rule to apply. Each clause is
  {"metric": one of {metrics}, "op": ">=" or "<", "value": number}. Clauses are checked on the
  workload descriptors (program profile and program-vs-chip-geometry) and say WHEN the rule
  applies. They never use the chip's measured speed, so IPC is not a condition metric.
- "claim": {"knob": knob name, "value": allowed value OR "up" / "down" for a numeric knob,
  "direction": "helps" or "hurts", "gain_pct": positive number}. It reads: switching this ONE knob
  to this value (or one step up/down from the baseline for a numeric knob) helps/hurts the objective
  by about gain_pct %. "direction" is the sign of the effect, "gain_pct" its size (always positive).
  Prefer "up"/"down" for sizes and ways: directions transfer to chips with other area budgets, where
  a fixed size may not be allowed. The loop measures every claim with a controlled comparison
  before the rule is admitted.
- "example": the run(s) that motivated it, e.g. "chip_x/workload_y: 0.31 -> 0.41".

A good rule reads like this (fictional knob and metric, for the shape only):
{"text": "When the workload misses the mid-level cache heavily, one more step of widget_ways pays off",
 "conditions": [{"metric": "metric_a", "op": ">=", "value": 12.0}],
 "claim": {"knob": "widget_ways", "value": "up", "direction": "helps", "gain_pct": 4.0},
 "example": "chip_x/workload_y: 0.310 -> 0.322"}
"""

REPLY_SCHEMA = """## Output format
A JSON list with ONE object with EXACTLY these keys:
- "hypothesis": one sentence naming the bottleneck and why this design tests it.
- "knobs": the design, one allowed value per knob (every knob listed above).
- "predicted": your point forecast of the objective for that design (number).
- "confidence": 0 to 1, how sure you are the measured value lands within 5% of "predicted".

The object looks like this (fictional knob names, for the shape only):
{"hypothesis": "The mid-level cache is capacity-bound, so a larger widget_size should cut its misses",
 "knobs": {"widget_size": 2048, "widget_ways": 8, "gadget_policy": "plain"},
 "predicted": 0.71, "confidence": 0.6}
"""


# LOOP_DISPATCH=chia (set by run_chia) routes every call through CHIA's Vertex
# node, so the profiler records it. Read from the environment, not module state,
# because Ray workers do not share the driver's module state.
_chia_node = None


def chia_node():
    global _chia_node
    if _chia_node is None:
        from loop.chia_nodes import AnalystNode
        _chia_node = AnalystNode(model=MODEL, project=GCP_PROJECT)
    return _chia_node


def ask_gemini(prompt, temperature=HYPOTHESIS_TEMPERATURE, thinking_budget=THINKING_BUDGET):
    """One LLM call, JSON-mode, parsed. All analyst functions go through here.
    ROLE goes in as the system instruction; `prompt` is the user turn."""
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        node = chia_node()
        answer = node.ask(ROLE + "\n\n" + prompt)
        tokens = node.llm._last_metadata
        _log_cost(tokens.get("input_tokens", 0), tokens.get("output_tokens", 0))
        return answer
    # The model occasionally returns truncated or malformed JSON; retry a few times.
    for attempt in range(5):
        started = time.time()
        response = _generate_with_backoff(prompt, temperature, thinking_budget)
        _log_usage(response)
        print("  [gemini] {:.1f}s".format(time.time() - started), flush=True)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("  [gemini] bad JSON, retrying. Tail:", repr(response.text[-200:]), flush=True)
    raise RuntimeError("Gemini returned malformed JSON five times")


def _generate_with_backoff(prompt, temperature, thinking_budget):
    """Rate limits (429) and server hiccups (5xx) are transient: wait and retry."""
    from google.genai import errors
    delays = [15, 30, 60, 120, 180, 240]
    for attempt in range(len(delays) + 1):
        try:
            return _client.models.generate_content(
                model=MODEL, contents=prompt,
                # The model "thinks" before answering and those tokens count against
                # the output cap; without a thinking budget it starves its own answer.
                config={"system_instruction": ROLE,
                        "response_mime_type": "application/json", "max_output_tokens": 16384,
                        "temperature": temperature,
                        "thinking_config": {"thinking_budget": thinking_budget}},
            )
        except errors.APIError as error:
            retryable = error.code == 429 or error.code >= 500
            if not retryable or attempt == len(delays):
                raise
            print("  [gemini] {} - waiting {}s".format(error.code, delays[attempt]), flush=True)
            time.sleep(delays[attempt])


def _log_usage(response):
    _log_cost(response.usage_metadata.prompt_token_count,
              response.usage_metadata.candidates_token_count)


def _log_cost(input_tokens, output_tokens):
    """Append this call's tokens and cost to the running usage log."""
    price_in, price_out = PRICES[MODEL]
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    # Parallel arm runs write this file at the same time: hold a lock while updating.
    with open(USAGE_LOG + ".lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if os.path.exists(USAGE_LOG):
            with open(USAGE_LOG) as log_file:
                log = json.load(log_file)
        else:
            log = {"total_cost_usd": 0.0, "calls": []}
        log["total_cost_usd"] += cost
        log["calls"].append({"model": MODEL, "in": input_tokens, "out": output_tokens,
                             "cost_usd": round(cost, 6)})
        with open(USAGE_LOG, "w") as log_file:
            json.dump(log, log_file, indent=2)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def as_list(answer):
    """The model may return {"rules": [...]} or a bare object instead of a list."""
    if isinstance(answer, list):
        return answer
    if isinstance(answer, dict):
        for value in answer.values():
            if isinstance(value, list):
                return value
        return [answer]
    return []


def knobs_text(search_space):
    return "## Knobs\nTunable knobs and their ONLY allowed values:\n{}\n".format(json.dumps(search_space, indent=2))


def rule_schema(condition_metrics):
    return RULE_SCHEMA.replace("{metrics}", " / ".join(condition_metrics))


def propose_reply(search_space, objective, results_table, rules_text, incumbent_name):
    """The right of reply: the search stalled, and the analyst names ONE untested
    design that should break out of it, with a point forecast the loop can score.

    Returns a list with one {hypothesis, knobs, predicted, confidence}.
    """
    prompt = knobs_text(search_space) + """
## Objective
Maximize {objective}.

## Playbook rules learned on earlier chips (may or may not apply here)
{rules}

## Simulation results so far on THIS chip and workload (one row per experiment)
{table}

## Task
The last round brought no improvement over the incumbent ({incumbent}). Name the
bottleneck the search is stuck on and propose the ONE untested design that best
tests it, with a pre-registered point forecast of its {objective}.

{schema}""".format(objective=objective, rules=rules_text, table=results_table, incumbent=incumbent_name,
                   schema=REPLY_SCHEMA)
    return as_list(ask_gemini(prompt, temperature=HYPOTHESIS_TEMPERATURE))


PICK_SCHEMA = """## Output format
A JSON list of objects, one per design, with EXACTLY these keys:
- "reasoning": one sentence on why this design should improve the objective.
- "knobs": the design, one allowed value per knob (every knob listed above).
- "predicted": your point forecast of the objective for that design (number).
- "confidence": 0 to 1, how sure you are the measured value lands within 5% of "predicted".
"""


def pick_designs(search_space, objective, results_table, descriptors_text, how_many, area_budget_kb):
    """The plain agent: from the results so far and the workload descriptors alone,
    the LLM names the next designs to simulate and forecasts each. No surrogate,
    no rules. Returns a list of {reasoning, knobs, predicted, confidence}."""
    prompt = knobs_text(search_space) + """
## Objective
Maximize {objective}. A design must fit the area budget: L2 capacity + LLC capacity
(sets x ways x 64 bytes each) <= {budget} KB.

## Workload descriptors (profiled from the traces, no simulator)
{descriptors}

## Simulation results so far on THIS chip and workload (one row per design)
{table}

## Task
You are tuning this cache hierarchy by yourself. Propose the {n} untested designs
most likely to raise {objective}, and pre-register a point forecast for each.

{schema}""".format(objective=objective, budget=area_budget_kb, descriptors=descriptors_text,
                   table=results_table, n=how_many, schema=PICK_SCHEMA)
    return as_list(ask_gemini(prompt, temperature=HYPOTHESIS_TEMPERATURE))


def distill_rules(search_space, results_table, ledger_text, existing_rules_text, condition_metrics,
                  effects_text="", descriptors_text="(not available)"):
    """Turn a finished chip's evidence into transferable rules. Returns a list of rules.
    effects_text: the measured one-knob effects (controlled pairs), largest first;
    the LLM writes the condition and the words over numbers that are already true."""
    prompt = knobs_text(search_space) + """
## All simulation results on this chip and workload
{table}

## Measured one-knob effects on this chip
Pairs of designs that differ in that knob only; mean change of the objective, largest first.
{effects}

## Workload descriptors (conditions use ONLY these)
Two kinds. Program-only, profiled from the trace with no simulator, identical on
every chip: memory accesses per 1000 instructions, write fraction, working set
(footprint_kb), fraction of stride-regular accesses (what a stride prefetcher
catches), fraction of reuses within 1024 accesses (temporal locality). Program
against THIS chip's cache sizes: working set over L2 size and over L2+LLC size,
the predicted LRU miss ratio at this chip's L1D, L2 and LLC capacity (footprint
theory, from the trace), and movable_l2_mpki / movable_llc_mpki: the misses per
1000 instructions that a bigger L2 / LLC would remove, from this chip's baseline
size up to the largest its area budget affords. Movable MPKI is THE capacity
descriptor: a program with a huge footprint but movable_llc_mpki near zero reads
its data once and no cache size helps it, while a large movable MPKI means
capacity pays. Footprint ratios cannot tell those two apart. A condition like
"movable_llc_mpki >= 5" means "a bigger LLC would remove at least 5 misses per
1000 instructions on this chip" and means the same on any chip. Every claim about a
SIZE knob (l2_sets, l2_ways, llc_sets, llc_ways) must therefore condition on the
matching movable MPKI (movable_l2_mpki for the L2, movable_llc_mpki for the LLC),
never on a footprint ratio alone.
{descriptors}

## Settled bets (what was predicted vs what happened)
{ledger}

## Rules already in the playbook (sharpen or add; one claim, one rule)
{existing}

## Task
Write AT MOST 5 rules an architect should carry to the NEXT chip, which may have a
different area budget and different feasible sizes. Work down the measured effects
from the largest: each rule takes one effect of a few percent or more that has no
rule yet, names the ONE knob it changes, and gives the condition (which workload
behaviour, in the metrics above, makes that effect appear). The effect's size is
measured, not yours to state; an effect near zero is not worth a rule. Claims about
the baseline's own value (no change) are not rules.

{schema}""".format(table=results_table, effects=effects_text or "(none measured yet)", ledger=ledger_text,
                   existing=existing_rules_text, descriptors=descriptors_text,
                   schema=rule_schema(condition_metrics))
    return as_list(ask_gemini(prompt, temperature=DISTILL_TEMPERATURE, thinking_budget=DISTILL_THINKING_BUDGET))


def rescope_rule(rule, losing_context, condition_metrics=None):
    """A rule lost a bet: sharpen its condition so it stops applying where it fails."""
    prompt = """## The rule that just lost a bet
{rule}

## Where it failed (baseline metrics of that chip+workload, and the outcome)
{context}

## Task
Keep the rule; re-scope it. Change ONLY "conditions" (and "text" to match) so the
rule no longer applies to cases like this one but still covers the example that
motivated it. A second clause on another metric is allowed when one threshold
cannot separate the two cases. Answer with the single updated rule as a JSON object.

{schema}""".format(rule=json.dumps(rule, indent=2), context=losing_context,
                   schema=rule_schema(condition_metrics or ["L1D_mpki", "L2C_mpki", "LLC_mpki"]))
    answer = ask_gemini(prompt, temperature=DISTILL_TEMPERATURE, thinking_budget=DISTILL_THINKING_BUDGET)
    if isinstance(answer, list) and len(answer) > 0:
        answer = answer[0]
    if not isinstance(answer, dict):
        return {}
    return answer


def textbook_rules(search_space, objective, how_many, condition_metrics):
    """Baseline arm: rules from prior knowledge only, before seeing any result.

    The delta between these and the loop's rules measures what the loop adds,
    and defends against "the LLM already knew this".
    """
    prompt = knobs_text(search_space) + """
## Task
You have NOT run any experiment. From textbook knowledge alone, write the {n}
rules you would carry into tuning these knobs to maximize {objective} on SPEC
CPU2017-like workloads. Conditions may only use the listed workload descriptors.

{schema}""".format(n=how_many, objective=objective, schema=rule_schema(condition_metrics))
    return as_list(ask_gemini(prompt, temperature=DISTILL_TEMPERATURE))
