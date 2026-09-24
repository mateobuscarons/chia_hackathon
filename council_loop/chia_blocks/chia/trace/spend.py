"""What a loop's model calls cost, added up and capped.

CHIA's model nodes already write what each call used into the profiler (input_tokens,
output_tokens, cache tokens, cost_usd, model). Nothing adds them up, and nothing stops a loop
that is spending more than it should. `spend_summary` reads the profiler's events and returns
USD and tokens per node and per model; `LLMSpend` (later in this module) is a loop-wide cap.
A model without a price here is summed in tokens and reported as unpriced, never guessed.
"""

import fcntl
import json
import os

# USD per million tokens, input then output. Thinking tokens bill as output.
PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-opus-4-1": (15.00, 75.00),
    "jev-latest": (0.042, 0.0),          # TypeSafe's decision model: input priced, output free
}

TOKEN_KEYS = ["input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"]


def cost_usd(model, input_tokens, output_tokens):
    """The list price of one call, or None for a model this table does not know."""
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000.0


def read_profile(path):
    """The profiler's JSON-lines file as a list of events."""
    events = []
    with open(path) as profile_file:
        for line in profile_file:
            if line.strip():
                events.append(json.loads(line))
    return events


def spend_summary(events):
    """USD and tokens per node (the profiled function) and per model, over the events that
    carry token counts. A call's own `cost_usd` is trusted when present; otherwise the price
    table prices it, and a model the table does not know is counted in tokens only."""
    empty = {"calls": 0, "usd": 0.0}
    for key in TOKEN_KEYS:
        empty[key] = 0
    by_function = {}
    by_model = {}
    by_function_model = {}
    unpriced = set()
    total = dict(empty)
    for event in events:
        usage = event.get("extra") or event
        if "input_tokens" not in usage and "output_tokens" not in usage:
            continue
        model = usage.get("model") or "unknown"
        function = event.get("func") or event.get("type") or "unknown"
        usd = usage.get("cost_usd")
        if usd is None:
            usd = cost_usd(model, usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0)
        if usd is None:
            unpriced.add(model)
            usd = 0.0
        for table, key in ((by_function, function), (by_model, model), (by_function_model, (function, model))):
            if key not in table:
                table[key] = dict(empty)
            table[key]["calls"] += 1
            table[key]["usd"] += usd
            for token_key in TOKEN_KEYS:
                table[key][token_key] += usage.get(token_key, 0) or 0
        total["calls"] += 1
        total["usd"] += usd
        for token_key in TOKEN_KEYS:
            total[token_key] += usage.get(token_key, 0) or 0
    return {"total": total, "by_function": by_function, "by_model": by_model,
            "by_function_model": by_function_model, "unpriced_models": sorted(unpriced)}


SPEND_COLUMNS = ["func", "model", "calls"] + TOKEN_KEYS + ["cost_usd", "priced"]


def spend_rows(summary):
    """The summary as rows for a CSV: one per (function, model), sorted by USD, then a total row.
    `priced` says whether the row's USD is real (the node's own figure or the price table) or
    the model is unpriced and the row counts tokens only."""
    rows = []
    pairs = summary["by_function_model"]

    def usd_of(pair):
        return -pairs[pair]["usd"]

    ordered = sorted(pairs, key=usd_of)
    for function, model in ordered:
        cell = pairs[(function, model)]
        row = [function, model, cell["calls"]]
        for token_key in TOKEN_KEYS:
            row.append(cell[token_key])
        row.append("{:.4f}".format(cell["usd"]))
        row.append("no" if model in summary["unpriced_models"] else "yes")
        rows.append(row)
    total = summary["total"]
    total_row = ["TOTAL", "", total["calls"]]
    for token_key in TOKEN_KEYS:
        total_row.append(total[token_key])
    total_row.append("{:.4f}".format(total["usd"]))
    total_row.append("partly" if summary["unpriced_models"] else "yes")
    rows.append(total_row)
    return rows


def profile_logs(paths):
    """The profiler log files under `paths`: a file is taken as is, a directory is searched for
    ChiaProfileCollector.log files (the same rule as `chia viz-profile --format table`)."""
    import pathlib
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    found = []
    for path in paths:
        path = pathlib.Path(path)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(path.rglob("ChiaProfileCollector.log")))
        else:
            raise FileNotFoundError("no such profiler log or directory: {}".format(path))
    return found


def render_spend_table(log_paths, output=None):
    """CLI entry point for `chia viz-profile --format spend`: one or more profiler logs or
    directories (searched for ChiaProfileCollector.log), one CSV row per node and model with
    calls, tokens by kind and USD, then a total row. Unpriced models are named on stderr."""
    import csv
    import sys
    events = []
    for log_path in profile_logs(log_paths):
        events.extend(read_profile(str(log_path)))
    summary = spend_summary(events)
    rows = spend_rows(summary)
    if output:
        handle = open(output, "w", newline="")
    else:
        handle = sys.stdout
    writer = csv.writer(handle)
    writer.writerow(SPEND_COLUMNS)
    for row in rows:
        writer.writerow(row)
    if output:
        handle.close()
    if summary["unpriced_models"]:
        print("unpriced models (tokens counted, 0 USD): " + ", ".join(summary["unpriced_models"]), file=sys.stderr)
    print("{} model calls, {:.4f} USD".format(summary["total"]["calls"], summary["total"]["usd"]), file=sys.stderr)


class SpendExhausted(RuntimeError):
    """Raised by a charge that would take the loop past its cap."""


class LLMSpend:
    """A loop-wide cap on what model calls may spend, in USD and in tokens (input plus
    output). Every model node charges it after a call; the charge that would pass a cap
    raises `SpendExhausted`, so a loop stops before it overspends instead of after. State is
    a small JSON file under an exclusive lock, so the processes of one run share one cap and
    a run that dies leaves its ledger of spend behind. Either cap may be None: uncapped."""

    def __init__(self, path, usd_cap=None, token_cap=None):
        self.path = path
        self.usd_cap = usd_cap
        self.token_cap = token_cap
        if not os.path.exists(path):
            self._write({"usd": 0.0, "tokens": 0, "calls": 0, "by_model": {}})

    def _write(self, state):
        with open(self.path, "w") as state_file:
            json.dump(state, state_file, indent=1)

    def _read(self):
        with open(self.path) as state_file:
            return json.load(state_file)

    def charge(self, model, input_tokens, output_tokens, usd=None):
        """Record one call. `usd` defaults to the price table's figure; an unpriced model
        charges 0 USD and its tokens. Returns the running totals."""
        if usd is None:
            usd = cost_usd(model, input_tokens, output_tokens) or 0.0
        tokens = input_tokens + output_tokens
        with open(self.path + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            state = self._read()
            over_usd = self.usd_cap is not None and state["usd"] + usd > self.usd_cap
            over_tokens = self.token_cap is not None and state["tokens"] + tokens > self.token_cap
            if over_usd or over_tokens:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                raise SpendExhausted("spent {:.2f} USD and {} tokens; a call of {:.4f} USD and {} tokens passes the cap ({} USD, {} tokens)".format(
                    state["usd"], state["tokens"], usd, tokens, self.usd_cap, self.token_cap))
            state["usd"] += usd
            state["tokens"] += tokens
            state["calls"] += 1
            per_model = state["by_model"].setdefault(model, {"calls": 0, "usd": 0.0, "tokens": 0})
            per_model["calls"] += 1
            per_model["usd"] += usd
            per_model["tokens"] += tokens
            self._write(state)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        return state

    def total(self):
        return self._read()

    def remaining(self):
        """What is left before each cap; None where there is no cap."""
        state = self._read()
        usd_left = None if self.usd_cap is None else self.usd_cap - state["usd"]
        tokens_left = None if self.token_cap is None else self.token_cap - state["tokens"]
        return {"usd": usd_left, "tokens": tokens_left}


# On a Ray cluster the same cap runs as one detached actor every worker charges:
#     spend = ray.remote(LLMSpend).options(name="llm_spend", lifetime="detached").remote(path, 20.0)
#     ray.get(spend.charge.remote(model, tokens_in, tokens_out))
# The file behind it makes the actor's state survive the actor.
