"""The analyst: every LLM call in the loop lives in this file.

The LLM proposes, explains and distills; it never scores anything. It
sees results as compact text and must answer in strict JSON so the loop
can act on it mechanically.
"""

import json
import os
import time

from google import genai

GCP_PROJECT = "project-c23a6080-f5d0-4871-9cb"
MODEL = "gemini-2.5-flash"
USAGE_LOG = "loop/llm_usage.json"

# USD per 1M tokens (input, output). Update if we change model.
PRICES = {"gemini-2.5-flash": (0.30, 2.50)}

# 120 s timeout per call: a hung request must fail loudly, not stall the loop.
_client = genai.Client(vertexai=True, project=GCP_PROJECT, location="us-central1",
                      http_options={"timeout": 120_000})

ROLE = "You are a computer architect running simulation experiments.\n"
RULE_SCHEMA = """A rule is a JSON object with EXACTLY these keys:
"text": the rule in one sentence, in your own words,
"condition": {"metric": one of ipc / L2C_mpki / LLC_mpki, "op": ">=" or "<", "value": number}
             - checked on the untouched baseline run of a chip+workload; it says WHEN the rule applies,
"claim": {"knob": knob name, "value": allowed value, "gain_pct": number}
             - "using this knob value changes the objective by about gain_pct % vs the baseline"
             (a point estimate, negative if it hurts; it is scored against the real number),
"example": the run(s) that motivated it, e.g. "B_midrange/mcf: 0.31 -> 0.41".
"""


# When set (see use_chia_node), calls go through CHIA's Vertex backend instead.
_chia_node = None


def use_chia_node(node):
    global _chia_node
    _chia_node = node


def ask_gemini(prompt):
    """One LLM call, JSON-mode, parsed. All analyst functions go through here."""
    if _chia_node is not None:
        answer = _chia_node.ask(prompt)
        tokens = _chia_node.llm._last_metadata
        _log_cost(tokens.get("input_tokens", 0), tokens.get("output_tokens", 0))
        return answer
    # The model occasionally returns truncated or malformed JSON; retry a few times.
    for attempt in range(3):
        started = time.time()
        response = _client.models.generate_content(
            model=MODEL, contents=prompt,
            # Flash "thinks" before answering and those tokens count against the
            # output cap; without a thinking budget it starves its own answer.
            config={"response_mime_type": "application/json", "max_output_tokens": 16384,
                    "thinking_config": {"thinking_budget": 2048}},
        )
        _log_usage(response)
        print("  [gemini] {:.1f}s".format(time.time() - started), flush=True)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            print("  [gemini] bad JSON, retrying. Tail:", repr(response.text[-200:]), flush=True)
    raise RuntimeError("Gemini returned malformed JSON three times")


def _log_usage(response):
    _log_cost(response.usage_metadata.prompt_token_count,
              response.usage_metadata.candidates_token_count)


def _log_cost(input_tokens, output_tokens):
    """Append this call's tokens and cost to the running usage log."""
    price_in, price_out = PRICES[MODEL]
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000

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


def knobs_text(search_space):
    return "Tunable knobs and their ONLY allowed values:\n{}\n".format(json.dumps(search_space, indent=2))


def propose_hypotheses(search_space, objective, results_table, rules_text, how_many):
    """Competing bottleneck hypotheses, each with a point forecast for one experiment.

    Returns a list of {hypothesis, knobs, predicted, confidence}.
    """
    prompt = ROLE + knobs_text(search_space) + """
The objective to maximize is: {objective}.
Playbook rules learned on earlier chips (may or may not apply here):
{rules}

Simulation results so far on THIS chip and workload (one row per experiment):
{table}

Propose {n} COMPETING hypotheses about the current performance bottleneck.
They must disagree: if one is right, another should be wrong. For each,
pick the single untested experiment (one value per knob) that best tests it,
and pre-register a point forecast of its {objective}.

Answer with a JSON list of objects with EXACTLY these keys:
"hypothesis" (one sentence), "knobs" (one allowed value per knob),
"predicted" (number), "confidence" (0 to 1: how sure you are the real
value lands within 5% of your forecast).
""".format(objective=objective, rules=rules_text, table=results_table, n=how_many)
    return ask_gemini(prompt)


def distill_rules(search_space, results_table, ledger_text, existing_rules_text):
    """Turn a finished chip's evidence into transferable rules. Returns a list of rules."""
    prompt = ROLE + knobs_text(search_space) + """
All simulation results on this chip and workload:
{table}

Settled bets (what was predicted vs what happened):
{ledger}

Rules already in the playbook (do not repeat them; sharpen or add):
{existing}

Write the rules an architect should carry to the NEXT chip. Only claim
what the evidence supports; give each rule a condition that says when it
applies. {schema}
Answer with a JSON list of rules.
""".format(table=results_table, ledger=ledger_text, existing=existing_rules_text,
           schema=RULE_SCHEMA)
    return ask_gemini(prompt)


def rescope_rule(rule, losing_context):
    """A rule lost a bet: sharpen its condition so it stops applying where it fails."""
    prompt = ROLE + """
This playbook rule just lost a bet:
{rule}

Where it failed (baseline metrics of that chip+workload, and the outcome):
{context}

Do NOT delete the rule. Re-scope it: change ONLY "condition" (and "text"
to match) so the rule no longer applies to cases like this one but still
covers the example that motivated it. {schema}
Answer with the single updated rule as a JSON object.
""".format(rule=json.dumps(rule, indent=2), context=losing_context, schema=RULE_SCHEMA)
    return ask_gemini(prompt)


def textbook_rules(search_space, objective, how_many):
    """Baseline arm: rules from prior knowledge only, before seeing any result.

    The delta between these and the loop's rules measures what the loop adds,
    and defends against "the LLM already knew this".
    """
    prompt = ROLE + knobs_text(search_space) + """
You have NOT run any experiment. From textbook knowledge alone, write the
{n} rules you would carry into tuning these knobs to maximize {objective}
on SPEC CPU2017-like workloads. Conditions may only use the metrics
ipc, L2C_mpki, LLC_mpki of the untouched baseline design. {schema}
Answer with a JSON list of rules.
""".format(n=how_many, objective=objective, schema=RULE_SCHEMA)
    return ask_gemini(prompt)
