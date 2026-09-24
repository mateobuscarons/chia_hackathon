"""Tier 0: the ledger's pure functions, no Ray, no simulator."""

import json

import pytest

from chia.trace.ledger import Ledger, convergence_speedup, equal_quality_speedup, markdown, rank_agreement, ranks, report


def filled(path, values, refused=()):
    book = Ledger(path, "ipc")
    for index, value in enumerate(values):
        violations = ["cap"] if index in refused else ()
        book.record({"knob": index}, {"ipc": value}, "test", index // 5, "because", violations, {"usd": 0.01, "seconds": 30})
    return book


def test_record_writes_one_json_line_per_row_and_reloads(tmp_path):
    path = str(tmp_path / "run.jsonl")
    book = filled(path, [0.9, 1.0, 1.1])
    lines = open(path).read().strip().split("\n")
    assert len(lines) == 3
    assert json.loads(lines[1])["value"] == 1.0
    assert json.loads(lines[1])["rationale"] == "because"
    reloaded = Ledger(path, "ipc")
    assert [row["value"] for row in reloaded.rows] == [0.9, 1.0, 1.1]
    assert book.rows[2]["index"] == 2


def test_best_so_far_ignores_refused_designs(tmp_path):
    book = filled(str(tmp_path / "a.jsonl"), [0.9, 1.5, 1.0, 1.2], refused=(1,))
    assert book.best_so_far() == [0.9, 0.9, 1.0, 1.2]
    assert book.final() == 1.2
    assert book.feasible_rate() == 0.75


def test_evaluations_to_and_plateau(tmp_path):
    book = filled(str(tmp_path / "a.jsonl"), [0.9, 1.0, 1.3, 1.3, 1.3, 1.3, 1.3])
    assert book.evaluations_to(1.3) == 3
    assert book.evaluations_to(2.0) is None
    assert book.plateau(4) == 3
    assert book.plateau(5) is None
    assert book.wasted_share() == pytest.approx(4 / 6)


def test_speedups_read_against_the_slow_ledger(tmp_path):
    fast = filled(str(tmp_path / "fast.jsonl"), [0.9, 1.3, 1.3, 1.3, 1.3])
    slow = filled(str(tmp_path / "slow.jsonl"), [0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])
    assert equal_quality_speedup(fast, slow) == pytest.approx(10 / 2)
    assert convergence_speedup(fast, slow, 3) == pytest.approx(5 / 2)
    never = filled(str(tmp_path / "never.jsonl"), [0.9, 1.0])
    assert equal_quality_speedup(never, slow) is None
    assert convergence_speedup(never, slow, 3) is None


def test_rank_agreement_perfect_reversed_and_ties():
    assert ranks([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0]
    same = rank_agreement([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12])
    assert same["spearman"] == pytest.approx(1.0)
    assert same["kendall"] == pytest.approx(1.0)
    assert same["top_overlap"] == 1.0
    opposite = rank_agreement([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1])
    assert opposite["spearman"] == pytest.approx(-1.0)
    assert opposite["kendall"] == pytest.approx(-1.0)


def test_report_and_markdown(tmp_path):
    fast = filled(str(tmp_path / "council.jsonl"), [0.9, 1.3, 1.3, 1.3, 1.3])
    slow = filled(str(tmp_path / "random.jsonl"), [0.9, 1.0, 1.1, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3])
    table = report({"council": fast, "random": slow}, window=3)
    assert table["reference"] == "random"
    assert table["rows"]["council"]["equal_quality_speedup"] == pytest.approx(5.0)
    assert table["rows"]["council"]["usd"] == pytest.approx(0.05)
    text = markdown(table)
    assert "| council |" in text
    assert "reference: random" in text


def test_report_and_markdown_read_two_ledgers_against_a_reference(tmp_path):
    from chia.trace.ledger import Ledger, markdown, report
    slow = Ledger(str(tmp_path / "slow.jsonl"), "ipc")
    fast = Ledger(str(tmp_path / "fast.jsonl"), "ipc")
    for index, value in enumerate([1.0, 1.1, 1.1, 1.2, 1.2, 1.2, 1.2, 1.2]):
        slow.record({"k": index}, {"ipc": value}, "random", 1 + index // 5)
    for index, value in enumerate([1.0, 1.25, 1.25]):
        fast.record({"k": index}, {"ipc": value}, "council", 1, cost={"usd": 0.1, "seconds": 60.0})
    table = report({"slow": slow, "fast": fast}, reference="slow", window=3)
    assert table["rows"]["fast"]["to_reference"] == 2
    assert table["rows"]["fast"]["equal_quality_speedup"] == 4.0
    assert table["rows"]["slow"]["plateau"] == 4 and table["rows"]["fast"]["plateau"] is None
    assert table["rows"]["fast"]["usd"] == pytest.approx(0.3) and table["rows"]["fast"]["seconds"] == 180.0
    text = markdown(table)
    assert "| fast |" in text and "4.0x" in text and "reference: slow" in text


def test_rank_agreement_perfect_and_reversed():
    from chia.trace.ledger import rank_agreement
    same = rank_agreement([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0], top=2)
    assert same["spearman"] == 1.0 and same["kendall"] == 1.0 and same["top_overlap"] == 1.0
    reversed_ = rank_agreement([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], top=2)
    assert reversed_["spearman"] == -1.0 and reversed_["kendall"] == -1.0 and reversed_["top_overlap"] == 0.0


def test_a_refused_candidate_is_recorded_but_never_the_best(tmp_path):
    from chia.trace.ledger import Ledger
    book = Ledger(str(tmp_path / "b.jsonl"), "ipc")
    book.record({"k": 0}, {"ipc": 1.0}, "stock", 0)
    book.record({"k": 1}, {"ipc": 1.9}, "council", 1, violations=["power 1.2 W over the 1.0 W cap"])
    book.record({"k": 2}, {"ipc": 1.2}, "council", 1)
    assert book.final() == 1.2 and book.feasible_rate() == 2 / 3
    assert len(Ledger(str(tmp_path / "b.jsonl"), "ipc").rows) == 3
