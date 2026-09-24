"""A call gate: a cheap decider says which optional agent calls are worth making this round.

A loop that consults several agents a round (specialists, reviewers, critics) pays for every
call, and many calls end in "nothing to add": the agent reads its context and holds. The gate
asks a cheap, calibrated decider one yes/no question per agent ("does the evidence warrant
consulting this agent now?"). In `shadow` mode every call is still made and the gate only
records what it would have skipped, so the saving and its price in lost proposals can be read
before anything is skipped. In `apply` mode the calls rated under the threshold are not made.
The decider is `chia.models.decider.Decider`: TypeSafe's Jev in production, a constant for tests. What each skipped or made call then produced is the
loop's knowledge; `log_outcome` writes it beside the verdict so the two can be read together.
"""

from chia.models.decider import ConstantDecider, Decider, JevDecider  # noqa: F401  (re-exported)

try:
    from chia.trace.profiler import get_profiler
except ImportError:
    get_profiler = None


class CallGate:
    """Gate a round's optional calls. `threshold` is the probability under which a call is not
    worth making; `mode` is shadow or apply."""

    def __init__(self, decider, threshold=0.4, mode="shadow"):
        if mode not in ("shadow", "apply"):
            raise ValueError("mode must be shadow or apply")
        self.decider = decider
        self.threshold = threshold
        self.mode = mode

    def gate(self, state, questions):
        """`state` is what the decider judges (short texts, no raw numbers); `questions` maps an
        agent's name to its question (`instructions` and `criteria` for true and false). Returns
        name -> {"probability", "consult"}; `consult` is always True in shadow mode."""
        typed = {}
        for name in questions:
            typed["consult:" + name] = dict(questions[name], type="noul")
        answers = self.decider.decide(state, typed)
        verdicts = {}
        for name in questions:
            probability = float(answers.get("consult:" + name, {}).get("noul", 1.0))
            if self.mode == "shadow":
                consult = True
            else:
                consult = probability >= self.threshold
            verdicts[name] = {"probability": probability, "consult": consult}
        return verdicts


def gate_and_log(gate, state, questions, label=None):
    """Gate one round and, when CHIA's profiler is running, log one `call_gate` event: the mode,
    the threshold, every verdict and the decider's own usage. Returns the verdicts."""
    verdicts = gate.gate(state, questions)
    if get_profiler is not None:
        profiler = get_profiler()
        if profiler.enabled:
            usage = getattr(gate.decider, "last_usage", {})
            profiler.log_event("call_gate", label=label, mode=gate.mode, threshold=gate.threshold, verdicts=verdicts,
                               decider_model=usage.get("model"), decider_input_tokens=usage.get("input_tokens", 0),
                               decider_cost_usd=usage.get("cost_usd", 0.0))
    return verdicts


def log_outcome(label, name, verdict, made, outcome):
    """What a gated call came to: `made` says whether the call was made, `outcome` what it produced
    (the loop's own words: held, proposed, the measured gain). One `call_gate_outcome` event."""
    if get_profiler is None:
        return
    profiler = get_profiler()
    if profiler.enabled:
        profiler.log_event("call_gate_outcome", label=label, name=name, probability=verdict["probability"],
                           consult=verdict["consult"], made=made, outcome=outcome)
