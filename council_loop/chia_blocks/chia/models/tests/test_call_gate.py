"""Tier 0: the call gate with a constant decider, no API."""

import pytest

from chia.models.call_gate import CallGate, ConstantDecider, gate_and_log

QUESTIONS = {"prefetch": {"instructions": "Does the sheet point at the prefetchers?", "criteria": {"true": "yes", "false": "no"}},
             "concurrency": {"instructions": "Does the sheet point at the queues?", "criteria": {"true": "yes", "false": "no"}}}


def test_shadow_mode_consults_everyone_and_records_probabilities():
    gate = CallGate(ConstantDecider(needed={"prefetch": 0.8, "concurrency": 0.2}), threshold=0.4, mode="shadow")
    verdicts = gate.gate({"sheet": "L1D misses dominate"}, QUESTIONS)
    assert verdicts["prefetch"] == {"probability": 0.8, "consult": True}
    assert verdicts["concurrency"] == {"probability": 0.2, "consult": True}


def test_apply_mode_skips_under_the_threshold_only():
    gate = CallGate(ConstantDecider(needed={"prefetch": 0.8, "concurrency": 0.2}), threshold=0.4, mode="apply")
    verdicts = gate_and_log(gate, {"sheet": "L1D misses dominate"}, QUESTIONS, label="round 3")
    assert verdicts["prefetch"]["consult"] is True
    assert verdicts["concurrency"]["consult"] is False


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        CallGate(ConstantDecider(), mode="maybe")


def test_outcome_logging_is_a_no_op_without_a_profiler():
    from chia.models.call_gate import log_outcome
    log_outcome("round 3", "prefetch", {"probability": 0.2, "consult": False}, made=False, outcome="skipped")


def test_gate_and_log_returns_verdicts_for_every_question_even_when_the_decider_omits_one():
    from chia.models.call_gate import CallGate, gate_and_log
    from chia.models.decider import Decider

    class Partial(Decider):
        def decide(self, state, questions):
            return {"consult:prefetch": {"noul": 0.2}}      # concurrency left unanswered

    verdicts = gate_and_log(CallGate(Partial(), threshold=0.4, mode="apply"), {}, QUESTIONS)
    assert verdicts["prefetch"]["consult"] is False
    assert verdicts["concurrency"] == {"probability": 1.0, "consult": True}   # unanswered: consulted
