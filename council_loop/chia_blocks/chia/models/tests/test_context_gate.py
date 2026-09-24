"""Tier 0: the context gate with a constant decider, no API."""

from chia.models.context_gate import ConstantDecider, ContextGate, PromptSections, estimate_tokens

PROMPT = """## Role
You are the analyst.

## Chip
3.8 GHz, 22 nm.

## The team's recent designs
  D1 ipc=1.10
  D2 ipc=1.16

## What each value of each knob has done
  l2_sets: 256: best 1.5

## Task
Write the sheet."""


def test_sections_split_and_render_round_trip():
    sections = PromptSections.from_markdown(PROMPT)
    assert sections.names() == ["Role", "Chip", "The team's recent designs", "What each value of each knob has done", "Task"]
    assert sections.render() == PROMPT
    assert sections.render(["Role", "Task"]) == "## Role\nYou are the analyst.\n\n## Task\nWrite the sheet."


def test_preamble_before_the_first_header_is_kept():
    sections = PromptSections.from_markdown("hello\n## A\na\n## B\nb")
    assert sections.names() == ["preamble", "A", "B"]
    assert sections.render() == "hello\n## A\na\n## B\nb"


def test_shadow_mode_reports_but_sends_the_prompt_unchanged():
    decider = ConstantDecider(noul=0.9, score=2, needed={"Chip": 0.2})
    result = ContextGate(decider, threshold=0.5, mode="shadow").gate(PROMPT)
    assert result["text"] == PROMPT
    assert result["dropped"] == {"Chip": 0.2}
    assert "Role" in result["kept"] and "Task" in result["kept"]
    assert result["difficulty"] == "hard" and result["thinking_budget"] == 8192
    assert result["tokens_after"] < result["tokens_before"] == estimate_tokens(PROMPT)


def test_apply_mode_drops_the_unneeded_sections_and_keeps_role_and_task():
    decider = ConstantDecider(noul=0.1, score=0, needed={"Chip": 0.95})
    result = ContextGate(decider, threshold=0.5, mode="apply").gate(PROMPT)
    assert result["kept"] == ["Role", "Chip", "Task"]
    assert "## The team's recent designs" not in result["text"]
    assert result["text"].startswith("## Role") and result["text"].endswith("Write the sheet.")
    assert result["difficulty"] == "easy" and result["thinking_budget"] == 1024


def test_the_decider_never_sees_numbers_only_heads_and_the_task():
    seen = {}

    class Spy(ConstantDecider):
        def decide(self, state, questions):
            seen.update(state)
            return super().decide(state, questions)

    ContextGate(Spy(), mode="shadow").gate(PROMPT)
    assert seen["task"] == "Write the sheet."
    assert set(seen["sections"]) == {"Chip", "The team's recent designs", "What each value of each knob has done"}
    assert seen["sections"]["Chip"] == "3.8 GHz, 22 nm."


def test_descriptions_replace_section_heads_and_prefixes_keep_sections():
    from chia.models.context_gate import ContextGate, ConstantDecider
    seen = {}

    class Recording(ConstantDecider):
        def decide(self, state, questions):
            seen.update(state["sections"])
            return super().decide(state, questions)

    def describe(name):
        return {"Chip": "the chip card, the same every round"}.get(name)

    gate = ContextGate(Recording(needed={"Chip": 0.1, "Recent designs": 0.9}), threshold=0.5, mode="apply",
                       always_kept=("Role", "Task", "Once more"), describe=describe, tiers={"easy": 1, "medium": 2, "hard": 3})
    prompt = "## Role\nr\n## Chip\n0.61 0.44 numbers\n## Recent designs\nD1 D2\n## Once more (again)\nask\n## Task\ndo"
    gated = gate.gate(prompt)
    assert seen["Chip"] == "the chip card, the same every round"     # the description, not the numbers
    assert seen["Recent designs"] == "D1 D2"                          # no description: the head
    assert "Once more (again)" in gated["kept"] and "Chip" in gated["dropped"]
    assert gated["thinking_budget"] == 2                              # the caller's tiers (score 1 -> medium)
    assert "## Chip" not in gated["text"] and "## Once more (again)" in gated["text"]
