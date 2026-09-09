"""Gaussian-process surrogate (scikit-learn), shared by every arm.

Interface used by the loop:
  fit(history, search_space, objective, honest_std, reference) -> model
  predict(model, knobs) -> (mean, std)
  predict_many(model, [knobs]) -> (means, stds)      one vectorised call
  probability_at_least(model, knobs, threshold) -> P(objective >= threshold)

Encoding: an ORDINAL knob (all values numeric: sets, ways) becomes one column,
log2(value) rescaled to [0, 1], so the GP knows 1024 sits between 512 and 2048.
A CATEGORICAL knob (prefetcher, replacement, soc) is one-hot encoded.
Kernel: Matern 5/2 with one length scale per column (ARD), so the GP can learn
that some knobs matter more than others.

honest_std=True adds the leave-one-out error to every predicted std: a GP fit
on a handful of points is overconfident, and the surrogate's BETS must not be.
The pure Bayesian-optimisation baseline (`bo` arm) passes honest_std=False and
uses the GP posterior as is.

Target space: the GP is fit on log(objective / reference), where reference is
the baseline objective of the problem each run came from (an entry may carry
its own "reference"; otherwise the current problem's reference is used). So a
run from a chip with twice the IPC contributes "+20% over its own baseline",
not an absolute number, and pooled data from other chips stops dragging the
predictions for this chip toward their scale. Predictions are converted back
to objective units for the current problem.
"""

import math
import warnings

import numpy
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

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
    """Fit the GP on every measured run in `history` (this problem's and, for the
    pooled arm, other chips'). Playbook rules never enter the fit: they shift the
    predicted mean afterwards (forecast.shift_for), so the GP's uncertainty stays honest.
    reference: the current problem's baseline objective (defaults to the first
    history entry's own reference, else its value)."""
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
