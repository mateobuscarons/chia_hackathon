"""Gemini on Vertex for one JSON answer, thinking counted.

CHIA's Vertex node (`chia.models.vertex`) drives a tool loop under a fixed generation config and
counts `candidates_token_count` as the output. A loop that asks for a JSON answer at a chosen
temperature and thinking budget, and needs the bill to include the thinking tokens (billed as
output), uses this node: one call, no tools, and a profiler event carrying input, output (answer
plus thinking), thinking, model and USD at the list price of `chia.trace.spend`. Rate limits and
server errors are retried with a backoff; a malformed answer is the caller's to retry, since
every attempt is a billed call.
"""

import time

from chia.base.ChiaFunction import ChiaFunction
from chia.trace.profiler import get_profiler
from chia.trace.spend import cost_usd

RETRY_DELAYS = (15, 30, 60, 120, 180, 240)


class VertexGeminiJSON:
    """One node per model and role; `prompt` is the CHIA function."""

    def __init__(self, model, system_message="", project=None, location="global", temperature=0.7,
                 max_output_tokens=16384, timeout_seconds=120):
        self.model = model
        self.system_message = system_message
        self.project = project
        self.location = location
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds

    @ChiaFunction(resources={"vertex_creds": 0.01})
    def prompt(self, user_message, thinking_budget=None, info=None):
        """The answer text and its usage record. `info` (a caller's label, round, run tag) joins
        the profiler event beside the tokens, the model and the USD."""
        from google import genai
        from google.genai import errors
        client = genai.Client(vertexai=True, project=self.project, location=self.location,
                              http_options={"timeout": self.timeout_seconds * 1000})
        config = {"response_mime_type": "application/json", "temperature": self.temperature,
                  "max_output_tokens": self.max_output_tokens}
        if self.system_message:
            config["system_instruction"] = self.system_message
        if thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": thinking_budget}
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=user_message, config=config)
                break
            except errors.APIError as error:
                transient = error.code == 429 or error.code >= 500
                if not transient or attempt == len(RETRY_DELAYS):
                    raise
                time.sleep(RETRY_DELAYS[attempt])
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count or 0
        thinking_tokens = usage.thoughts_token_count or 0
        output_tokens = (usage.candidates_token_count or 0) + thinking_tokens
        record = {"model": self.model, "input_tokens": input_tokens, "output_tokens": output_tokens,
                  "thinking_tokens": thinking_tokens, "cost_usd": cost_usd(self.model, input_tokens, output_tokens)}
        if info:
            record.update(info)
        profiler = get_profiler()
        if profiler.enabled:
            profiler.add_info(record)
        return response.text or "", record
