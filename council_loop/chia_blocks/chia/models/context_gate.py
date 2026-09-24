"""A context gate: a cheap decider triages a prompt's sections before an expensive model reads it.

An agent's prompt is assembled from sections - the chip, the counters, the ledgers, the recent
designs, the task. Most of a call's cost is those sections, and not every section is needed for
every task. The gate asks a cheap, calibrated decider one yes/no question per section ("is this
section needed to perform the task?") and one question about the task's difficulty, then either
drops the sections judged unnecessary (`apply`) or only records what it would have dropped
(`shadow`). Shadow mode costs nothing but the decider's cents and produces the evidence to decide
whether apply mode is worth it; apply mode saves the tokens and sets the model's thinking budget
by the difficulty.

The decider is the interface in `chia.models.decider`: `JevDecider` (TypeSafe's Jev, a decision
model with calibrated probabilities) is the one shipped; `ConstantDecider` answers fixed values for tests. The
decider never sees numbers to judge, only what a section is about and what the task asks: that is
where such models are reliable.

`PromptSections` splits a Markdown prompt on its `## ` headers, so any prompt already written in
sections can be gated without restructuring the code that writes it.
"""

import os
import re

from chia.models.decider import ConstantDecider, Decider, JevDecider  # noqa: F401  (re-exported)

try:
    from chia.trace.profiler import get_profiler
except ImportError:
    get_profiler = None

# Characters per token, for the estimate the gate reports before a model has counted anything.
CHARS_PER_TOKEN = 4.0
ALWAYS_KEPT = ("preamble", "Role", "Task")
THINKING_TIERS = {"easy": 1024, "medium": 4096, "hard": 8192}


class PromptSections:
    """An ordered list of (name, text) sections of one prompt."""

    def __init__(self, sections):
        self.sections = list(sections)

    @classmethod
    def from_markdown(cls, text):
        """Split on lines that start with `## `; text before the first header is `preamble`.
        The header line stays with its section, so `render()` returns the prompt unchanged."""
        sections = []
        name = "preamble"
        lines = []
        for line in text.split("\n"):
            header = re.match(r"## (.+)", line)
            if header:
                if lines or name != "preamble":
                    sections.append((name, "\n".join(lines)))
                name = header.group(1).strip()
                lines = [line]
            else:
                lines.append(line)
        sections.append((name, "\n".join(lines)))
        if sections and sections[0][0] == "preamble" and sections[0][1].strip() == "":
            sections = sections[1:]
        return cls(sections)

    def names(self):
        return [name for name, _ in self.sections]

    def render(self, kept=None):
        """The prompt from the kept sections (all when `kept` is None), in their order."""
        parts = []
        for name, text in self.sections:
            if kept is None or name in kept:
                parts.append(text)
        return "\n".join(parts)

    def head(self, text, characters=300):
        body = text.split("\n", 1)[1] if "\n" in text else ""
        return body.strip()[:characters]


def estimate_tokens(text):
    return int(len(text) / CHARS_PER_TOKEN)


class ContextGate:
    """Gate one prompt. `threshold` is the probability under which a section is judged
    unnecessary; sections in `always_kept` are never asked about."""

    def __init__(self, decider, threshold=0.5, mode="shadow", always_kept=ALWAYS_KEPT, describe=None, tiers=None):
        """`describe(name)` returns a one-line description of what a section holds, or None: the
        decider judges the description rather than the section's first characters, which for a
        table of numbers say nothing it can read. `tiers` maps easy|medium|hard to a thinking
        budget; the caller's mapping, since what a budget means differs by model and role.
        A section whose name starts with an `always_kept` entry is never asked about."""
        if mode not in ("shadow", "apply"):
            raise ValueError("mode must be shadow or apply")
        self.decider = decider
        self.threshold = threshold
        self.mode = mode
        self.always_kept = tuple(always_kept)
        self.describe = describe
        self.tiers = dict(tiers or THINKING_TIERS)

    def kept_always(self, name):
        for prefix in self.always_kept:
            if name.startswith(prefix):
                return True
        return False

    def gate(self, prompt_text):
        """Returns a dict: `text` (the prompt to send: unchanged in shadow mode), `kept`,
        `dropped` (name -> probability), `probabilities`, `difficulty` (easy|medium|hard),
        `thinking_budget`, `tokens_before`, `tokens_after` (an estimate; the model's own count
        is the measurement), `mode`."""
        sections = PromptSections.from_markdown(prompt_text)
        task_text = ""
        for name, text in sections.sections:
            if name == "Task":
                task_text = sections.head(text, 1500)
        state = {"task": task_text, "sections": {}}
        questions = {}
        for name, text in sections.sections:
            if self.kept_always(name):
                continue
            description = self.describe(name) if self.describe is not None else None
            state["sections"][name] = description if description else sections.head(text)
            questions["needed:" + name] = {
                "type": "noul",
                "instructions": "Is the section named '{}' (described under sections) needed to perform the task well?".format(name),
                "criteria": {"true": "the task cannot be done well without what this section holds",
                             "false": "the task can be done as well without it"}}
        questions["difficulty"] = {
            "type": "score",
            "instructions": "How hard is the task to decide well from these sections?",
            "criteria": ["easy: one section decides it", "medium: several sections must be weighed",
                         "hard: the sections conflict or the evidence is thin"]}
        answers = self.decider.decide(state, questions)
        probabilities = {}
        dropped = {}
        kept = []
        for name, _ in sections.sections:
            if self.kept_always(name):
                kept.append(name)
                continue
            probability = float(answers.get("needed:" + name, {}).get("noul", 1.0))
            probabilities[name] = probability
            if probability < self.threshold:
                dropped[name] = probability
            else:
                kept.append(name)
        score = answers.get("difficulty", {}).get("score", 1)
        difficulty = ["easy", "medium", "hard"][min(2, max(0, int(round(float(score)))))]
        text = prompt_text if self.mode == "shadow" else sections.render(kept)
        return {"text": text, "kept": kept, "dropped": dropped, "probabilities": probabilities,
                "difficulty": difficulty, "thinking_budget": self.tiers[difficulty],
                "tokens_before": estimate_tokens(prompt_text), "tokens_after": estimate_tokens(sections.render(kept)),
                "mode": self.mode}


def gate_and_log(gate, prompt_text, label=None):
    """Gate one prompt and, when CHIA's profiler is running, log what the gate saw: tokens
    before and after, the sections dropped with their probabilities, the difficulty, the mode
    and the decider's own usage. Returns the gate's result."""
    result = gate.gate(prompt_text)
    if get_profiler is not None:
        profiler = get_profiler()
        if profiler.enabled:
            usage = getattr(gate.decider, "last_usage", {})
            profiler.log_event("context_gate", label=label, mode=result["mode"],
                               tokens_before=result["tokens_before"], tokens_after=result["tokens_after"],
                               dropped=result["dropped"], difficulty=result["difficulty"],
                               decider_model=usage.get("model"), decider_input_tokens=usage.get("input_tokens", 0),
                               decider_cost_usd=usage.get("cost_usd", 0.0))
    return result
