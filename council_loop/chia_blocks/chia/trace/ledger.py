"""An evaluation ledger for any iterative loop: one row per candidate the loop evaluated.

A loop that searches, repairs or evolves a design evaluates candidates one after another, and
every such loop in CHIA's examples keeps its own private record of them. This module is that
record once: who proposed a candidate and why, what it measured, which constraints it broke and
what it cost. From the rows it answers the questions a search is compared on - the best value
so far after N evaluations, evaluations to reach a target, the plateau, the share of wasted
evaluations - and compares two ledgers the way the sample-efficiency literature does. Rows are
appended to a JSON-lines file, so a run that dies keeps everything it measured.

Nothing here knows what a candidate is. A candidate is a dict of knobs; metrics is a dict of
numbers; the objective is the name of the metric to maximise.
"""

import json
import os
import time

# The pure ledger works without CHIA installed; with it, every row is also a profiler event and
# the node below records from any worker of a cluster.
try:
    from chia.trace.profiler import get_profiler
except ImportError:
    get_profiler = None
try:
    from chia.base.ChiaFunction import ChiaFunction
except ImportError:
    ChiaFunction = None


class Ledger:
    """The rows of one run, in evaluation order, mirrored to a JSON-lines file."""

    def __init__(self, path, objective):
        self.path = path
        self.objective = objective
        self.rows = []
        if os.path.exists(path):
            with open(path) as ledger_file:
                for line in ledger_file:
                    if line.strip():
                        self.rows.append(json.loads(line))

    def record(self, candidate, metrics, source, round_index, rationale=None, violations=(), cost=None):
        """Append one evaluation and return its row. `source` says who proposed the candidate
        (an arm, an agent, a sweep), `rationale` why, `violations` the constraints it broke
        (empty means feasible), `cost` what the evaluation spent (seconds, tokens, USD)."""
        row = {
            "index": len(self.rows),
            "time": time.time(),
            "round": round_index,
            "source": source,
            "rationale": rationale,
            "candidate": candidate,
            "metrics": metrics,
            "value": metrics[self.objective],
            "violations": list(violations),
            "cost": cost or {},
        }
        self.rows.append(row)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a") as ledger_file:
            ledger_file.write(json.dumps(row) + "\n")
        if get_profiler is not None:
            profiler = get_profiler()
            if profiler.enabled:
                profiler.log_event("candidate", index=row["index"], source=source, value=row["value"],
                                   violations=len(row["violations"]), ledger=self.path)
        return row

    def best_so_far(self):
        """The best feasible value after each evaluation, one entry per row; None until the
        first feasible row. The curve every sample-efficiency comparison is read from."""
        curve = []
        best = None
        for row in self.rows:
            if not row["violations"]:
                if best is None or row["value"] > best:
                    best = row["value"]
            curve.append(best)
        return curve

    def final(self):
        """The best feasible value the run reached, or None."""
        curve = self.best_so_far()
        if not curve:
            return None
        return curve[-1]

    def evaluations_to(self, target):
        """How many evaluations it took to reach `target` (a value at least as good), or None
        if the run never did."""
        curve = self.best_so_far()
        for index in range(len(curve)):
            if curve[index] is not None and curve[index] >= target:
                return index + 1
        return None

    def plateau(self, window):
        """Evaluations until the best stopped improving for `window` further evaluations, the
        convergence point AgentDSE compares methods at. None when the run ended before a
        plateau of that length could be seen."""
        curve = self.best_so_far()
        last_gain = None
        for index in range(len(curve)):
            if curve[index] is not None and (index == 0 or curve[index] != curve[index - 1]):
                last_gain = index
        if last_gain is None or len(curve) - 1 - last_gain < window:
            return None
        return last_gain + 1

    def wasted_share(self):
        """The share of evaluations after the first that did not raise the best so far."""
        curve = self.best_so_far()
        if len(curve) < 2:
            return 0.0
        wasted = 0
        for index in range(1, len(curve)):
            if curve[index] == curve[index - 1]:
                wasted += 1
        return wasted / (len(curve) - 1)

    def feasible_rate(self):
        """The share of evaluated candidates that broke no constraint."""
        if not self.rows:
            return 0.0
        feasible = 0
        for row in self.rows:
            if not row["violations"]:
                feasible += 1
        return feasible / len(self.rows)


def equal_quality_speedup(fast, slow):
    """Evaluations the slow ledger spent, over the evaluations the fast ledger needed to reach
    the slow ledger's final value (AgentDSE's Table VI). None when the fast one never did."""
    target = slow.final()
    if target is None:
        return None
    needed = fast.evaluations_to(target)
    if needed is None:
        return None
    return len(slow.rows) / needed


def convergence_speedup(fast, slow, window):
    """Evaluations to plateau, slow over fast (AgentDSE's Table VII). None when either run
    ended before a plateau of `window` evaluations could be seen."""
    fast_plateau = fast.plateau(window)
    slow_plateau = slow.plateau(window)
    if fast_plateau is None or slow_plateau is None:
        return None
    return slow_plateau / fast_plateau


def ranks(values):
    """Ranks from 1, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for tied in range(position, end + 1):
            result[order[tied]] = average
        position = end + 1
    return result


def rank_agreement(low, high, top=5):
    """How two evaluators rank the same candidates, e.g. a short simulation against a long
    one: Spearman's rho, Kendall's tau, and the share of `low`'s top `top` that are in
    `high`'s top `top`. `low` and `high` are the two values of the same candidates, in order."""
    count = len(low)
    low_ranks = ranks(low)
    high_ranks = ranks(high)
    low_mean = sum(low_ranks) / count
    high_mean = sum(high_ranks) / count
    covariance = 0.0
    low_spread = 0.0
    high_spread = 0.0
    for index in range(count):
        covariance += (low_ranks[index] - low_mean) * (high_ranks[index] - high_mean)
        low_spread += (low_ranks[index] - low_mean) ** 2
        high_spread += (high_ranks[index] - high_mean) ** 2
    spearman = covariance / ((low_spread * high_spread) ** 0.5) if low_spread > 0 and high_spread > 0 else 0.0
    concordant = 0
    discordant = 0
    for first in range(count):
        for second in range(first + 1, count):
            sign = (low[first] - low[second]) * (high[first] - high[second])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    kendall = (concordant - discordant) / (count * (count - 1) / 2.0) if count > 1 else 0.0
    low_top = set(sorted(range(count), key=lambda index: -low[index])[:top])
    high_top = set(sorted(range(count), key=lambda index: -high[index])[:top])
    overlap = len(low_top & high_top) / float(min(top, count)) if count else 0.0
    return {"spearman": spearman, "kendall": kendall, "top_overlap": overlap, "count": count}


def report(ledgers, reference=None, window=20):
    """One row per ledger, read against a reference ledger (default: the one with the most
    evaluations): final value, evaluations, evaluations to the reference's final value, plateau,
    wasted share, feasible rate, the two speedups, and what the run spent in USD and seconds
    (from each row's `cost`, keys "usd" and "seconds")."""
    if reference is None:
        reference = max(ledgers, key=lambda name: len(ledgers[name].rows))
    slow = ledgers[reference]
    rows = {}
    for name in ledgers:
        ledger = ledgers[name]
        usd = 0.0
        seconds = 0.0
        for row in ledger.rows:
            usd += row["cost"].get("usd", 0.0)
            seconds += row["cost"].get("seconds", 0.0)
        to_reference = None
        if slow.final() is not None:
            to_reference = ledger.evaluations_to(slow.final())
        rows[name] = {
            "final": ledger.final(),
            "evaluations": len(ledger.rows),
            "to_reference": to_reference,
            "plateau": ledger.plateau(window),
            "wasted_share": ledger.wasted_share(),
            "feasible_rate": ledger.feasible_rate(),
            "equal_quality_speedup": 1.0 if name == reference else equal_quality_speedup(ledger, slow),
            "convergence_speedup": 1.0 if name == reference else convergence_speedup(ledger, slow, window),
            "usd": usd,
            "seconds": seconds,
        }
    return {"reference": reference, "window": window, "rows": rows}


def markdown(report_):
    """The report as a Markdown table."""
    lines = ["| run | final | evaluations | to reference | plateau | wasted | feasible | speedup (equal quality) | speedup (convergence) | USD | sim min |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in report_["rows"]:
        row = report_["rows"][name]
        cells = [name]
        for key, form in [("final", "{:.4f}"), ("evaluations", "{}"), ("to_reference", "{}"), ("plateau", "{}"),
                          ("wasted_share", "{:.0%}"), ("feasible_rate", "{:.0%}"),
                          ("equal_quality_speedup", "{:.1f}x"), ("convergence_speedup", "{:.1f}x"), ("usd", "{:.2f}")]:
            if row[key] is None:
                cells.append("-")
            else:
                cells.append(form.format(row[key]))
        cells.append("{:.0f}".format(row["seconds"] / 60.0))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("reference: {} (speedups read against its final value; plateau = no gain over the next {} evaluations)".format(
        report_["reference"], report_["window"]))
    return "\n".join(lines)


if ChiaFunction is not None:

    class LedgerNode:
        """The ledger as a CHIA node. The file is the state, so the node carries only its path and
        objective; `record` runs wherever the loop calls it, so a loop on a cluster records from
        any worker, and `report` reads several ledgers at once."""

        def __init__(self, path, objective):
            self.path = path
            self.objective = objective

        @staticmethod
        @ChiaFunction()
        def record(path, objective, candidate, metrics, source, round_index, rationale=None, violations=(), cost=None):
            return Ledger(path, objective).record(candidate, metrics, source, round_index, rationale, violations, cost)

        @staticmethod
        @ChiaFunction()
        def report(paths, objective, reference=None, window=20):
            ledgers = {}
            for name in paths:
                ledgers[name] = Ledger(paths[name], objective)
            return report(ledgers, reference, window)
