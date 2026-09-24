"""Deciders: the cheap, calibrated judge behind CHIA's gates.

A decider answers typed questions about a small state: for a `noul` question a probability that
the answer is yes, for a `score` question a level with its confidence. It reads text, never
numbers to compute with, and it is priced so that asking it about a model call costs about a
thousandth of the call. `JevDecider` is TypeSafe's Jev; `ConstantDecider` answers fixed values
for tests; any object with the same `decide(state, questions)` method serves.
"""

import json
import os
import time
import urllib.error
import urllib.request


class Decider:
    """What a decider answers: for each question, a probability (`noul`) or a score with its
    per-level probabilities and confidence (`score`). `state` is what it judges."""

    def decide(self, state, questions):
        raise NotImplementedError


class ConstantDecider(Decider):
    """Answers every yes/no question with one probability and every score with one level."""

    def __init__(self, noul=0.9, score=1, confidence=0.8, needed=None):
        self.noul = noul
        self.score = score
        self.confidence = confidence
        self.needed = needed or {}     # per section name, overrides `noul`

    def decide(self, state, questions):
        answers = {}
        for name in questions:
            question = questions[name]
            if question["type"] == "noul":
                bare_name = name.split(":", 1)[1] if ":" in name else name
                answers[name] = {"noul": self.needed.get(bare_name, self.noul)}
            else:
                answers[name] = {"score": self.score, "confidence": self.confidence}
        return answers


class JevDecider(Decider):
    """TypeSafe's Jev, a decision model that answers typed questions with calibrated
    probabilities: one POST per prompt gated, every section's question in the same call.
    Input costs 0.042 USD per million tokens and output is free, so a gated call costs about
    one fiftieth of the call it gates. The key comes from `TYPESAFE_API_KEY` or the argument."""

    ENDPOINT = "https://api.typesafe.ai/v1/systemone"
    PRICE_PER_MILLION_INPUT = 0.042

    def __init__(self, api_key=None, model="jev-latest", timeout_seconds=30, retries=3):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("JevDecider needs an API key (TYPESAFE_API_KEY or api_key=)")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.last_usage = {}

    def decide(self, state, questions):
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        request = urllib.request.Request(self.ENDPOINT, data=body, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    answer = json.loads(response.read())
                break
            except urllib.error.HTTPError as error:
                # 429 is the rate limit, 529 the service overloaded: wait and try again.
                if error.code in (429, 529) and attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
        usage = answer.get("usage", {})
        self.last_usage = {"model": answer.get("model", self.model),
                           "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                           "cost_usd": usage.get("input_tokens", 0) * self.PRICE_PER_MILLION_INPUT / 1_000_000.0}
        return answer.get("answers", {})


