"""The random-forest optimizer: one surrogate and one search, used everywhere.

  python -m loop.forest <designs> <batch> <trace> [trace ...]   # the reference search

`run()` is that search. `loop/run.py` calls it for the `bo` arm (from the stock chip)
and for the `pooled_bo` arm (from the design the memory hands over); `reference()`
calls it long to set the ceiling a cell is scored against. Budget, batch and warm-up
are arguments, so the settings an arm and the reference disagree on are visible at
the call site rather than living in a second copy of the loop - comparing two curves
grown under different warm-ups is how a wrong ratio gets published.

The surrogate is a RANDOM FOREST, not a Gaussian process. Replaying both offline
against the cached tables, on the 300-design uniform sample (the only pool that
is a fair draw from the space - pools filled by earlier searches are so rich that
drawing at random already reaches 92-96%, so no method can be told apart in
them), 60 seeds, 16 designs:

    surrogate        D4    D8   D12   D16   area   D16 min..max
    forest          58%   78%   86%   92%     72       62..100
    gaussian proc   53%   71%   82%   88%     68       28..100
    random search   57%   73%   78%   81%     67       49..100

The forest is four points of area above the GP, which is itself one point above
drawing at random, and it removes the seeds that end at 28% of the gap. For a
ceiling that reliability is the point: a reference that depends on a lucky seed
is not a reference. (Measured on the same data: the GP is NOT specifically weak
on the categorical knobs, so the usual explanation for this does not apply here.
What the forest does better is not established.)

Uncertainty comes from the spread across trees, which is what makes expected
improvement possible without a Gaussian process.
"""

import math
import os
import sys
import time

import numpy
from sklearn.ensemble import RandomForestRegressor

from loop import champsim_problem
from loop.champsim_problem import candidate_pool
from loop.configs import SEARCH_SPACE, typed_knobs

# The initial design, before the forest chooses anything. At a 16-design budget
# the sweep found any warm-up beats none and the size barely matters; at 100
# designs nothing was measured, so this is the conventional ten percent.
WARM_UP_SHARE = 0.10
TREES = 100


# ---------------------------------------------------------------- the surrogate ----

def encode(knobs):
    """A size or a width becomes one column holding log2 of its value; a prefetcher
    or a policy becomes one column per value, holding 1 for the value it has.

    Trees can split on a value's index directly, so the one-hot widening looks
    unnecessary here - but it was measured, and indexing the values is worse
    (D16 88% against 92%), because an index imposes an order on policies that have
    none and a single split then has to separate values that are not adjacent."""
    row = []
    for knob in sorted(SEARCH_SPACE):
        values = SEARCH_SPACE[knob]
        if isinstance(values[0], str):
            for value in values:
                if str(value) == str(knobs[knob]):
                    row.append(1.0)
                else:
                    row.append(0.0)
        else:
            row.append(math.log2(float(knobs[knob])))
    return row


def encode_many(knobs_list):
    rows = []
    for knobs in knobs_list:
        rows.append(encode(knobs))
    return numpy.array(rows)


def fit_forest(knobs_list, values):
    forest = RandomForestRegressor(n_estimators=TREES, min_samples_leaf=1, max_features=0.8,
                                   bootstrap=True, random_state=0, n_jobs=-1)
    forest.fit(encode_many(knobs_list), numpy.array(values))
    return forest


def predict(forest, knobs_list):
    """Mean and spread over the trees: the forest's predictive distribution."""
    rows = encode_many(knobs_list)
    per_tree = []
    for tree in forest.estimators_:
        per_tree.append(tree.predict(rows))
    stacked = numpy.array(per_tree)
    spread = numpy.maximum(stacked.std(axis=0), 1e-6)
    return stacked.mean(axis=0), spread


def expected_improvement(means, spreads, best_so_far):
    scores = []
    for mean, spread in zip(means, spreads):
        z = (mean - best_so_far) / spread
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        scores.append((mean - best_so_far) * cdf + spread * pdf)
    return numpy.array(scores)


# ---------------------------------------------------------------- the candidates ----

def neighbours(knobs, problem):
    """Every runnable design one knob away: where a search finishes, and what a
    random sample of a 6.6-million-design space never contains."""
    found = {}
    for knob in SEARCH_SPACE:
        for value in SEARCH_SPACE[knob]:
            if str(value) == str(knobs[knob]):
                continue
            design = dict(knobs)
            design[knob] = value
            if problem["is_candidate"](design):
                found[problem["name_of"](design)] = design
    return found


# ---------------------------------------------------------------- the search ----

WARM_UP_DESIGNS = 3       # the initial design at an arm's budget, as the sweep set it


def run(problem, budget, batch, seed, tag, start_design=None, warm_up=WARM_UP_DESIGNS):
    """`budget` designs bought in batches of `batch`. Every forest search in the
    project goes through here, so the settings an arm and the reference disagree on
    are arguments rather than two copies of one loop.

    start_design: bought before anything else. `pooled_bo` passes the design the
    memory hands over, so the only difference from `bo` is where the search begins;
    the surrogate, the warm-up and the candidate pool are identical.

    warm_up: designs taken from the pool before the forest chooses anything. The
    arms use WARM_UP_DESIGNS; a long reference search passes a share of its own
    budget. State a search's warm-up and batch whenever its curve is compared with
    another's - they decide where the curve bends, and a comparison across two
    different settings is not a comparison.

    The candidate pool is drawn with the run's own seed and keeps the whole space in
    view, not only the incumbent's neighbours: on this chip the best design measured
    sits SIX knobs from the memory's handover, and everything within five of it is
    capped well below the ceiling, so a search that only walks one knob at a time
    cannot get there from a good start."""
    objective = problem["objective"]
    started = time.time()
    candidates = candidate_pool(problem, seed)
    candidates.pop(problem["name_of"](problem["stock"]), None)
    broken = set()          # designs the simulator could not measure; never offered again

    stock_metrics = problem["evaluate"](problem["stock"])
    history = [{"index": 0, "round": 0, "name": problem["name_of"](problem["stock"]), "knobs": problem["stock"],
                "metrics": stock_metrics, "source": "stock", "hypothesis": None}]
    measured_knobs = [problem["stock"]]
    measured_values = [stock_metrics[objective]]
    best = stock_metrics[objective]
    best_knobs = problem["stock"]
    print("[{}] {} designs, batches of {}, {} to warm up | {} | stock {:.4f}".format(
        tag, budget, batch, warm_up, "+".join(problem["workloads"]), best), flush=True)

    def one_at_a_time(knobs_list):
        kept = []
        metrics = []
        for knobs in knobs_list:
            name = problem["name_of"](knobs)
            try:
                metrics.append(problem["evaluate"](knobs))
                kept.append(knobs)
            except Exception as error:
                broken.add(name)
                candidates.pop(name, None)
                print("[{}] dropped {}: {}".format(tag, name, repr(error)[-160:]), flush=True)
        return kept, metrics

    def buy(knobs_list, source, round_number):
        """Measure a batch and record it. A design that makes the simulator produce
        no usable stats used to end the whole run - one seed died that way in two
        separate cells, because the candidate pool is drawn from the seed and so the
        same bad design came back every time. Retry design by design instead, drop
        whatever still fails, and let the loop pick a replacement: the budget counts
        designs that produced a measurement."""
        nonlocal best, best_knobs
        try:
            metrics_list = problem["evaluate_many"](knobs_list)
        except Exception as error:
            print("[{}] batch failed ({}), retrying one at a time".format(tag, repr(error)[-160:]), flush=True)
            knobs_list, metrics_list = one_at_a_time(knobs_list)
        for knobs, metrics in zip(knobs_list, metrics_list):
            name = problem["name_of"](knobs)
            candidates.pop(name, None)
            entry = {"index": len(history), "round": round_number, "name": name, "knobs": knobs,
                     "metrics": metrics, "source": source, "hypothesis": None}
            history.append(entry)
            measured_knobs.append(knobs)
            measured_values.append(metrics[objective])
            if metrics[objective] > best:
                best = metrics[objective]
                best_knobs = knobs
            print("[{}] round {} | D{} | {}={:.4f} | best {:.4f} | {} | {:.0f} min".format(
                tag, round_number, entry["index"], objective, metrics[objective], best, source,
                (time.time() - started) / 60), flush=True)

    if start_design is not None:
        buy([typed_knobs(start_design)], "memory", 0)

    opening = []
    for name in candidates:
        if len(opening) == min(warm_up, budget - (len(history) - 1)):
            break
        opening.append(candidates[name])
    if len(opening) > 0:
        buy(opening, "warm-up", 0)

    round_number = 0
    while len(history) - 1 < budget:
        round_number += 1
        for name, knobs in neighbours(best_knobs, problem).items():
            if name not in candidates and name not in broken:
                candidates[name] = knobs
        wanted = min(batch, budget - (len(history) - 1))
        chosen = choose(candidates, measured_knobs, measured_values, best, wanted)
        knobs_list = []
        for name in chosen:
            knobs_list.append(candidates[name])
        buy(knobs_list, "forest", round_number)
    return {"designs": history, "rounds": []}


def reference(traces, how_many, batch):
    """The ceiling a cell is scored against. Same search, run long, and it must stay
    a mechanism separate from the arms: scoring against a design an arm found bounds
    the metric at that arm, which has happened once and cost every arm five points."""
    problem = champsim_problem.make_suite_problem(traces, allow_simulation=True)
    objective = problem["objective"]
    result = run(problem, how_many, batch, 0, "reference", warm_up=max(1, int(how_many * WARM_UP_SHARE)))
    stock = result["designs"][0]["metrics"][objective]
    best = stock
    best_knobs = result["designs"][0]["knobs"]
    for entry in result["designs"]:
        if entry["metrics"][objective] > best:
            best = entry["metrics"][objective]
            best_knobs = entry["knobs"]
    print("reference: best {:.4f} (+{:.1f}% on the stock chip) over {} designs".format(
        best, 100.0 * (best / stock - 1.0), how_many), flush=True)
    print("reference: best design " + problem["name_of"](best_knobs), flush=True)
    return best


def choose(candidates, measured_knobs, measured_values, best_so_far, wanted):
    """`wanted` designs by expected improvement. Within one batch a pick joins the
    training set at its own predicted value, so the next pick of the same batch
    does not simply repeat it."""
    believed_knobs = list(measured_knobs)
    believed_values = list(measured_values)
    names = []
    for name in candidates:
        names.append(name)
    knobs_list = []
    for name in names:
        knobs_list.append(candidates[name])

    chosen = []
    while len(chosen) < wanted:
        forest = fit_forest(believed_knobs, believed_values)
        means, spreads = predict(forest, knobs_list)
        scores = expected_improvement(means, spreads, best_so_far)
        best_index = None
        for index, name in enumerate(names):
            if name in chosen:
                continue
            if best_index is None or scores[index] > scores[best_index]:
                best_index = index
        if best_index is None:
            break
        chosen.append(names[best_index])
        believed_knobs.append(knobs_list[best_index])
        believed_values.append(float(means[best_index]))
    return chosen


if __name__ == "__main__":
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        from loop import run
        run.start_chia()
    reference(sys.argv[3:], int(sys.argv[1]), int(sys.argv[2]))
