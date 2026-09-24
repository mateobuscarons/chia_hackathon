"""Tier 0: the spend summary and the cap, no Ray, no key."""

import pytest

from chia.trace.spend import LLMSpend, SpendExhausted, cost_usd, spend_summary


def test_cost_usd_prices_known_models_only():
    assert cost_usd("gemini-3.1-pro-preview", 1_000_000, 0) == pytest.approx(2.0)
    assert cost_usd("gemini-3.1-pro-preview", 0, 1_000_000) == pytest.approx(12.0)
    assert cost_usd("some-local-model", 1000, 1000) is None


def test_spend_summary_sums_per_node_and_model_and_flags_unpriced():
    events = [
        {"type": "complete", "func": "VertexGeminiLLM.prompt", "extra": {"model": "gemini-3.1-pro-preview", "input_tokens": 1_000_000, "output_tokens": 0}},
        {"type": "complete", "func": "VertexGeminiLLM.prompt", "extra": {"model": "gemini-3.1-pro-preview", "input_tokens": 0, "output_tokens": 1_000_000}},
        {"type": "complete", "func": "ClaudeLLM.prompt", "extra": {"model": "claude-sonnet-4-5", "input_tokens": 10, "output_tokens": 10, "cost_usd": 0.5}},
        {"type": "complete", "func": "OllamaLLM.prompt", "extra": {"model": "llama3", "input_tokens": 700, "output_tokens": 100}},
        {"type": "complete", "func": "build_champsim", "exec_time_s": 6.0},
    ]
    summary = spend_summary(events)
    assert summary["total"]["calls"] == 4
    assert summary["total"]["usd"] == pytest.approx(14.5)
    assert summary["by_function"]["VertexGeminiLLM.prompt"]["usd"] == pytest.approx(14.0)
    assert summary["by_model"]["claude-sonnet-4-5"]["usd"] == pytest.approx(0.5)
    assert summary["by_model"]["llama3"]["input_tokens"] == 700
    assert summary["unpriced_models"] == ["llama3"]


def test_cap_raises_before_passing_and_persists(tmp_path):
    path = str(tmp_path / "spend.json")
    cap = LLMSpend(path, usd_cap=5.0)
    cap.charge("gemini-3.1-pro-preview", 1_000_000, 0)
    cap.charge("gemini-3.1-pro-preview", 1_000_000, 0)
    assert cap.remaining()["usd"] == pytest.approx(1.0)
    with pytest.raises(SpendExhausted):
        cap.charge("gemini-3.1-pro-preview", 1_000_000, 0)
    assert cap.total()["calls"] == 2
    assert LLMSpend(path, usd_cap=5.0).total()["usd"] == pytest.approx(4.0)


def test_token_cap_and_uncapped_remaining(tmp_path):
    cap = LLMSpend(str(tmp_path / "s.json"), token_cap=1000)
    cap.charge("unknown-model", 600, 300)
    assert cap.remaining() == {"usd": None, "tokens": 100}
    with pytest.raises(SpendExhausted):
        cap.charge("unknown-model", 100, 100)


def test_spend_rows_and_csv_from_a_log_directory(tmp_path):
    import csv
    import json
    from chia.trace.spend import render_spend_table, spend_rows
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    events = [
        {"type": "complete", "func": "prompt", "extra": {"model": "gemini-3.1-pro-preview", "input_tokens": 1_000_000, "output_tokens": 0}},
        {"type": "complete", "func": "prompt", "extra": {"model": "llama3", "input_tokens": 700, "output_tokens": 100}},
        {"type": "complete", "func": "run_champsim", "exec_time_s": 6.0},
    ]
    with open(log_dir / "ChiaProfileCollector.log", "w") as log_file:
        for event in events:
            log_file.write(json.dumps(event) + "\n")
    rows = spend_rows(spend_summary(events))
    assert rows[0][:3] == ["prompt", "gemini-3.1-pro-preview", 1] and rows[0][-2:] == ["2.0000", "yes"]
    assert rows[1][:2] == ["prompt", "llama3"] and rows[1][-1] == "no"
    assert rows[-1][0] == "TOTAL" and rows[-1][2] == 2 and rows[-1][-1] == "partly"
    out = tmp_path / "spend.csv"
    render_spend_table([str(tmp_path)], output=str(out))
    with open(out) as csv_file:
        table = list(csv.reader(csv_file))
    assert table[0][0] == "func" and table[-1][0] == "TOTAL" and table[-1][3] == "1000700"


def test_cap_charge_of_an_unpriced_model_counts_tokens_only(tmp_path):
    cap = LLMSpend(str(tmp_path / "c.json"), usd_cap=1.0, token_cap=None)
    state = cap.charge("some-local-model", 5000, 500)
    assert state["usd"] == 0.0 and state["tokens"] == 5500 and state["by_model"]["some-local-model"]["calls"] == 1


def test_spend_summary_trusts_a_node_written_cost_over_the_table():
    events = [{"type": "complete", "func": "prompt", "extra": {"model": "gemini-3.1-pro-preview", "input_tokens": 1_000_000, "output_tokens": 0, "cost_usd": 0.5}}]
    assert spend_summary(events)["total"]["usd"] == pytest.approx(0.5)
