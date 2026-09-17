"""The model: Gemini on Vertex. JSON in, JSON out. Everything the model can get wrong on the wire
is handled here - rate limits, malformed answers, the cost log - so the council never has to
know that it is talking to a language model.
"""

import fcntl
import json
import os
import time

from google import genai


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
