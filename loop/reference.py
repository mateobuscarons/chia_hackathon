"""The reference search: the ceiling a suite's arms are scored against.

  python -m loop.reference <designs> <batch> <trace> [trace ...]

Self-contained on purpose. The arms' searcher (`loop/bo.py`) is tuned for a
16-design budget and is part of what the experiment measures; the reference is
not an arm and must not drift with it, so it carries its own surrogate here and
shares only the problem definition and the result tables.

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
from loop.configs import SEARCH_SPACE, random_feasible_designs

# The initial design, before the forest chooses anything. At a 16-design budget
# the sweep found any warm-up beats none and the size barely matters; at 100
# designs nothing was measured, so this is the conventional ten percent.
WARM_UP_SHARE = 0.10
CANDIDATE_POOL = 20000
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


def starting_pool(problem):
    pool = {}
    for knobs in random_feasible_designs(CANDIDATE_POOL, seed=0):
        if problem["is_candidate"](knobs):
            pool[problem["name_of"](knobs)] = knobs
    pool.pop(problem["name_of"](problem["stock"]), None)
    return pool


# ---------------------------------------------------------------- the search ----

def search(traces, how_many, batch):
    problem = champsim_problem.make_suite_problem(traces, allow_simulation=True)
    objective = problem["objective"]
    started = time.time()

    stock_metrics = problem["evaluate"](problem["stock"])
    stock = stock_metrics[objective]
    measured_knobs = [problem["stock"]]
    measured_values = [stock]
    best = stock
    best_knobs = problem["stock"]

    candidates = starting_pool(problem)
    warm_up = max(1, int(how_many * WARM_UP_SHARE))
    print("reference: {} designs, batches of {}, {} random to start | {} | stock {:.4f}".format(
        how_many, batch, warm_up, "+".join(problem["workloads"]), stock), flush=True)

    bought = 0
    while bought < how_many:
        wanted = min(batch, how_many - bought)
        if bought < warm_up:
            wanted = min(wanted, warm_up - bought)
            chosen = []
            for name in candidates:
                if len(chosen) == wanted:
                    break
                chosen.append(name)
            source = "random"
        else:
            for name, knobs in neighbours(best_knobs, problem).items():
                if name not in candidates:
                    candidates[name] = knobs
            chosen = choose(candidates, measured_knobs, measured_values, best, wanted)
            source = "forest"

        knobs_list = []
        for name in chosen:
            knobs_list.append(candidates[name])
        metrics_list = problem["evaluate_many"](knobs_list)
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            candidates.pop(name, None)
            measured_knobs.append(knobs)
            measured_values.append(metrics[objective])
            bought += 1
            if metrics[objective] > best:
                best = metrics[objective]
                best_knobs = knobs
            print("[reference] D{} | {}={:.4f} | best {:.4f} | {} | {:.0f} min".format(
                bought, objective, metrics[objective], best, source, (time.time() - started) / 60), flush=True)

    print("reference: best {:.4f} (+{:.1f}% on the stock chip) over {} designs in {:.0f} min".format(
        best, 100.0 * (best / stock - 1.0), how_many, (time.time() - started) / 60), flush=True)
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
    search(sys.argv[3:], int(sys.argv[1]), int(sys.argv[2]))
