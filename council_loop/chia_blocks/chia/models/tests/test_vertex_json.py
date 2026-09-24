"""Tier 0: the JSON Gemini node against a stub client: the call it makes, the usage it records,
the retry on a rate limit. No network."""

import types

import pytest

from chia.models import vertex_json
from chia.models.vertex_json import VertexGeminiJSON


class StubModels:
    """Answers in order; an exception in the list is raised in its turn."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        usage = types.SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=7)
        return types.SimpleNamespace(text=answer, usage_metadata=usage)


def stub_client(monkeypatch, answers):
    from google import genai
    models = StubModels(answers)

    def make_client(**kwargs):
        return types.SimpleNamespace(models=models)

    monkeypatch.setattr(genai, "Client", make_client)
    return models


def no_sleep(seconds):
    return None


def test_prompt_makes_one_json_call_and_counts_thinking_as_output(monkeypatch):
    models = stub_client(monkeypatch, ['{"ok": true}'])
    node = VertexGeminiJSON("gemini-3.1-pro-preview", system_message="be brief", project="p", location="global")
    text, record = node.prompt("hello", thinking_budget=1024, info={"label": "analyst", "round": 3})
    assert text == '{"ok": true}'
    call = models.calls[0]
    assert call["model"] == "gemini-3.1-pro-preview" and call["contents"] == "hello"
    assert call["config"]["response_mime_type"] == "application/json"
    assert call["config"]["temperature"] == 0.7 and call["config"]["max_output_tokens"] == 16384
    assert call["config"]["system_instruction"] == "be brief"
    assert call["config"]["thinking_config"] == {"thinking_budget": 1024}
    assert record["input_tokens"] == 10 and record["thinking_tokens"] == 7 and record["output_tokens"] == 12
    assert record["cost_usd"] == pytest.approx((10 * 2.00 + 12 * 12.00) / 1e6)
    assert record["model"] == "gemini-3.1-pro-preview" and record["label"] == "analyst" and record["round"] == 3


def test_prompt_retries_a_rate_limit_and_raises_on_a_client_error(monkeypatch):
    from google.genai import errors
    monkeypatch.setattr(vertex_json.time, "sleep", no_sleep)
    models = stub_client(monkeypatch, [errors.APIError(429, {"error": {"message": "slow down"}}), '{"ok": 1}'])
    node = VertexGeminiJSON("gemini-3.1-pro-preview")
    text, record = node.prompt("hello")
    assert text == '{"ok": 1}' and len(models.calls) == 2
    assert "thinking_config" not in models.calls[-1]["config"]
    assert "system_instruction" not in models.calls[-1]["config"]
    models = stub_client(monkeypatch, [errors.APIError(400, {"error": {"message": "bad request"}})])
    with pytest.raises(errors.APIError):
        node.prompt("hello")
    assert len(models.calls) == 1
