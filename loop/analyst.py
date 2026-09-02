"""The analyst: every LLM call in the loop lives in this file.

The LLM proposes and explains; it never scores anything. It sees results
as a compact text table and must answer in strict JSON so the loop can
act on it mechanically.
"""

import json
import os

from google import genai

from loop.configs import SEARCH_SPACE

GCP_PROJECT = "project-c23a6080-f5d0-4871-9cb"
MODEL = "gemini-2.5-flash"
USAGE_LOG = "loop/llm_usage.json"

# USD per 1M tokens (input, output). Update if we change model.
PRICES = {"gemini-2.5-flash": (0.30, 2.50)}

_client = genai.Client(vertexai=True, project=GCP_PROJECT, location="us-central1")


def ask_gemini(prompt):
    """One LLM call, JSON-mode, parsed. All analyst functions go through here."""
    response = _client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    _log_usage(response)
    return json.loads(response.text)


def _log_usage(response):
    """Append this call's tokens and cost to the running usage log."""
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
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


def propose_hypotheses(results_table, how_many):
    """Ask for competing bottleneck hypotheses, each with a pre-registered bet.

    Returns a list of dicts:
      {hypothesis, knobs, event, probability}
    where knobs picks one value per search-space dimension (the experiment
    that tests the hypothesis) and event is an objectively checkable claim.
    """
    prompt = """You are a CPU cache-hierarchy architect running experiments.

Tunable knobs and their ONLY allowed values:
{space}

Simulation results so far (one row per experiment):
{table}

Propose {n} COMPETING hypotheses about the current performance bottleneck.
They must disagree: if one is right, another should be wrong.
For each, pick the single experiment (one value per knob) that best tests
it, and pre-register a bet: an objectively checkable event about that
experiment's outcome (e.g. "ipc >= 0.35") and your probability (0 to 1).

Answer with a JSON list of objects with EXACTLY these keys:
"hypothesis" (one sentence), "knobs" (object with one allowed value per
knob), "event" (checkable claim about ipc using >= or <), "probability".
""".format(space=json.dumps(SEARCH_SPACE, indent=2), table=results_table, n=how_many)

    return ask_gemini(prompt)
