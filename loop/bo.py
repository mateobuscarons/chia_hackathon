"""The statistical baseline: textbook Bayesian optimisation. A Gaussian process
fit on every design measured so far; each round picks `per_round` designs by
expected improvement (batch via the kriging believer) over a seeded random
sample of the feasible space plus every unmeasured design one or two knobs away
from the incumbent (the random sample never holds the incumbent's neighbours,
where a search finishes).

The surrogate: a Gaussian process in log-speedup units. An ORDINAL knob (sets, ways,
MSHRs) becomes one column, log2(value) rescaled to [0, 1]; a CATEGORICAL knob
(prefetcher, replacement) is one-hot encoded; Matern 5/2 with one length scale per
column. The target is log(objective / stock objective).
"""

import math
import random
import warnings

import numpy
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from loop.champsim_problem import candidate_pool
from loop.configs import SEARCH_SPACE


# ---------------------------------------------------------------- the surrogate ----

MIN_STD = 0.005

# Noise (in normalised-target units, since normalize_y=True) attached to every
# measurement: ChampSim is deterministic, so only a hair above zero.
REAL_NOISE = 1e-6

# With few points the kernel optimizer hits its bounds; harmless, and noisy.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def is_ordinal(values):
    for value in values:
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
    return True


def build_columns(search_space):
    """[(knob, kind, payload)]: kind "ordinal" (payload = (log2 min, log2 max))
    or "onehot" (payload = the value as a string)."""
    columns = []
    for knob in search_space:
        values = search_space[knob]
        if is_ordinal(values) and len(values) > 1:
            logs = []
            for value in values:
                logs.append(math.log2(float(value)))
            columns.append((knob, "ordinal", (min(logs), max(logs))))
        else:
            for value in values:
                columns.append((knob, "onehot", str(value)))
    return columns


def encode(knobs, columns):
    """One feature row for one config."""
    row = []
    for knob, kind, payload in columns:
        if kind == "ordinal":
            low, high = payload
            position = math.log2(float(knobs[knob]))
            if high > low:
                row.append((position - low) / (high - low))
            else:
                row.append(0.0)
        else:
            if str(knobs[knob]) == payload:
                row.append(1.0)
            else:
                row.append(0.0)
    return row


def encode_many(knobs_list, columns):
    rows = []
    for knobs in knobs_list:
        rows.append(encode(knobs, columns))
    return numpy.array(rows)


def to_log_speedup(value, reference):
    return math.log(max(value, 1e-9) / reference)


def fit(history, search_space, objective, honest_std=True, reference=None):
    """Fit the GP on every measured design in `history`. reference: the stock
    design's objective (defaults to the first history entry's value). honest_std
    adds the leave-one-out error to every predicted std; the textbook `bo` arm
    passes False and uses the posterior as is."""
    if reference is None:
        first = history[0]
        reference = first.get("reference", first["metrics"][objective])
    columns = build_columns(search_space)

    rows = []
    targets = []
    noises = []
    for entry in history:
        rows.append(encode(entry["knobs"], columns))
        own_reference = entry.get("reference", reference)
        targets.append(to_log_speedup(entry["metrics"][objective], own_reference))
        noises.append(REAL_NOISE)
    real_count = len(rows)

    regressor = fit_regressor(rows, targets, noises, len(columns))

    loo_std = 0.0
    if honest_std and real_count >= 3:
        squared_errors = []
        for left_out in range(real_count):
            other_rows = rows[:left_out] + rows[left_out + 1:]
            other_targets = targets[:left_out] + targets[left_out + 1:]
            other_noises = noises[:left_out] + noises[left_out + 1:]
            partial = fit_regressor(other_rows, other_targets, other_noises, len(columns),
                                    restarts=0, kernel_from=regressor)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prediction = partial.predict(numpy.array([rows[left_out]]))[0]
            squared_errors.append((prediction - targets[left_out]) ** 2)
        loo_std = math.sqrt(sum(squared_errors) / len(squared_errors))
    return {"regressor": regressor, "columns": columns, "loo_std": loo_std, "reference": reference}


def fit_regressor(rows, targets, noises, dimension, restarts=3, kernel_from=None):
    """kernel_from: reuse an already-fitted kernel (the leave-one-out refits do
    this: same hyper-parameters, one fewer point, so they cost a solve, not an
    optimisation)."""
    if kernel_from is not None:
        kernel = kernel_from.kernel_
        optimizer = None
    else:
        kernel = default_kernel(dimension)
        optimizer = "fmin_l_bfgs_b"
    # With fewer than 3 real points the marginal likelihood has no information
    # about length scales: keep the default hyper-parameters instead of letting
    # the optimizer drift to a degenerate (near-singular) kernel.
    if len(rows) < 3:
        optimizer = None
    # alpha = per-point noise: this is how a rule's virtual point is trusted less.
    regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=numpy.array(noises),
                                         n_restarts_optimizer=restarts, optimizer=optimizer,
                                         random_state=0)
    regressor.fit(numpy.array(rows), numpy.array(targets))
    if not finite_on_training(regressor, rows):
        # Degenerate fit: fall back to the default hyper-parameters, no optimisation.
        regressor = GaussianProcessRegressor(kernel=default_kernel(dimension), normalize_y=True,
                                             alpha=numpy.array(noises), optimizer=None, random_state=0)
        regressor.fit(numpy.array(rows), numpy.array(targets))
    return regressor


def default_kernel(dimension):
    """Features live in [0, 1]; a length scale above ~5 already means "flat", so the
    upper bounds stay tight to keep the covariance matrix well conditioned."""
    length_scales = numpy.ones(dimension)
    return (ConstantKernel(1.0, (1e-2, 1e1))
            * Matern(length_scale=length_scales, length_scale_bounds=(5e-2, 5.0), nu=2.5)
            + WhiteKernel(1e-5, (1e-7, 1e-1)))


def finite_on_training(regressor, rows):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means, stds = regressor.predict(numpy.array(rows), return_std=True)
    return bool(numpy.isfinite(means).all() and numpy.isfinite(stds).all())


def predict_many(model, knobs_list):
    """Vectorised prediction: (means, stds) as numpy arrays."""
    if len(knobs_list) == 0:
        return numpy.array([]), numpy.array([])
    rows = encode_many(knobs_list, model["columns"])
    # Apple's BLAS raises a spurious "divide by zero in matmul" flag on the tiny
    # covariances a short length scale produces; the values are finite (checked).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log_means, log_stds = model["regressor"].predict(rows, return_std=True)
    log_stds = numpy.sqrt(log_stds ** 2 + model["loo_std"] ** 2)
    # Back to objective units: value = reference * exp(log speedup); the std
    # follows by the delta method (std of value ~ value * std of log value).
    means = model["reference"] * numpy.exp(numpy.clip(log_means, -5.0, 5.0))
    stds = means * log_stds
    stds = numpy.maximum(stds, MIN_STD)
    return means, stds


def predict(model, knobs):
    means, stds = predict_many(model, [knobs])
    return float(means[0]), float(stds[0])


def probability_at_least(model, knobs, threshold):
    mean, std = predict(model, knobs)
    return normal_tail(mean, std, threshold)


def normal_tail(mean, std, threshold):
    """P(X >= threshold) for X ~ Normal(mean, std)."""
    z = (threshold - mean) / std
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------- the search ----


def neighbourhood(incumbent_knobs, problem):
    """Every runnable design one or two knob changes away from the incumbent."""
    designs = {}

    def consider(knobs):
        if problem["is_candidate"](knobs):
            designs[problem["name_of"](knobs)] = knobs

    one_step = []
    for knob in SEARCH_SPACE:
        for value in SEARCH_SPACE[knob]:
            if str(value) == str(incumbent_knobs[knob]):
                continue
            design = dict(incumbent_knobs)
            design[knob] = value
            one_step.append(design)
            consider(design)
    for design in one_step:
        for knob in SEARCH_SPACE:
            if str(design[knob]) != str(incumbent_knobs[knob]):
                continue
            for value in SEARCH_SPACE[knob]:
                if str(value) == str(incumbent_knobs[knob]):
                    continue
                second = dict(design)
                second[knob] = value
                consider(second)
    return designs


def expected_improvement(mean, std, best_so_far):
    if std <= 0.0:
        return max(0.0, mean - best_so_far)
    z = (mean - best_so_far) / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (mean - best_so_far) * cdf + std * pdf


def pick_by_expected_improvement(candidates, model, history, objective, how_many, seed):
    """Batch of `how_many` by the kriging believer: each pick joins the training
    set at its predicted mean and the GP is refit before the next pick."""
    names = list(candidates.keys())
    random.Random(seed).shuffle(names)
    knobs_list = []
    for name in names:
        knobs_list.append(candidates[name])
    believed = list(history)
    best_so_far = max(entry["metrics"][objective] for entry in history)
    chosen = []
    for pick in range(how_many):
        if pick > 0:
            model = fit(believed, SEARCH_SPACE, objective, honest_std=False, reference=model["reference"])
        means, stds = predict_many(model, knobs_list)
        best_name = None
        best_score = None
        for index, name in enumerate(names):
            if name in chosen:
                continue
            score = expected_improvement(float(means[index]), float(stds[index]), best_so_far)
            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        chosen.append(best_name)
        believer_index = names.index(best_name)
        believed.append({"knobs": candidates[best_name], "metrics": {objective: float(means[believer_index])}})
    return chosen


def run_bo(problem, rounds, per_round, seed, tag):
    objective = problem["objective"]
    candidates = candidate_pool(problem, seed)
    stock_metrics = problem["evaluate"](problem["stock"])
    history = [{"index": 0, "round": 0, "name": problem["name_of"](problem["stock"]), "knobs": problem["stock"],
                "metrics": stock_metrics, "source": "stock", "hypothesis": None}]
    candidates.pop(history[0]["name"], None)
    for round_number in range(1, rounds + 1):
        incumbent = history[0]
        for entry in history:
            if entry["metrics"][objective] > incumbent["metrics"][objective]:
                incumbent = entry
        measured = set()
        for entry in history:
            measured.add(entry["name"])
        for name, knobs in neighbourhood(incumbent["knobs"], problem).items():
            if name not in measured:
                candidates[name] = knobs
        model = fit(history, SEARCH_SPACE, objective, honest_std=False, reference=stock_metrics[objective])
        chosen = pick_by_expected_improvement(candidates, model, history, objective, per_round, seed * 1000 + round_number)
        knobs_list = []
        for name in chosen:
            knobs_list.append(candidates[name])
        metrics_list = problem["evaluate_many"](knobs_list)
        for name, knobs, metrics in zip(chosen, knobs_list, metrics_list):
            entry = {"index": len(history), "round": round_number, "name": name, "knobs": knobs, "metrics": metrics,
                     "source": "bo", "hypothesis": None}
            history.append(entry)
            candidates.pop(name, None)
            print("[{}] round {} | D{} | {}={:.4f} | bo".format(tag, round_number, entry["index"], objective, metrics[objective]), flush=True)
    return {"designs": history, "rounds": []}
