"""Gaussian-process surrogate: the strong statistical baseline (scikit-learn).

Same three functions as loop/surrogate.py so the loop can swap them:
fit(history) -> model, predict(model, knobs) -> (mean, std),
probability_at_least(model, knobs, threshold) -> P(objective >= threshold).
Knobs are one-hot encoded; every knob value becomes one 0/1 column.
"""

import math
import warnings

import numpy
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

MIN_STD = 0.005

# With few points the kernel optimizer hits its bounds; harmless, and noisy.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def encode(knobs, columns):
    """One row of 0/1 features. columns = [(knob, value), ...] in fixed order."""
    row = []
    for knob, value in columns:
        if str(knobs[knob]) == value:
            row.append(1.0)
        else:
            row.append(0.0)
    return row


# Noise attached to a virtual point from a rule the GP fully trusts / distrusts.
# Real measurements get the tiny value; a rule's point gets more, so real data
# overrides a rule as soon as it arrives.
REAL_NOISE = 1e-6
PRIOR_NOISE_MIN = 1e-3
PRIOR_NOISE_MAX = 1e-1


def fit(history, search_space, objective, priors=None):
    """priors: optional list of {"knobs", "value", "confidence"} - virtual
    experiments from playbook rules. They warm-start the GP; their noise grows
    as confidence drops, and they never count in the leave-one-out error."""
    if priors is None:
        priors = []
    columns = []
    for knob in search_space:
        for value in search_space[knob]:
            columns.append((knob, str(value)))

    rows = []
    targets = []
    noises = []
    for entry in history:
        rows.append(encode(entry["knobs"], columns))
        targets.append(entry["metrics"][objective])
        noises.append(REAL_NOISE)
    real_count = len(rows)
    for prior in priors:
        rows.append(encode(prior["knobs"], columns))
        targets.append(prior["value"])
        distrust = 1.0 - prior["confidence"]
        noises.append(PRIOR_NOISE_MIN + distrust * (PRIOR_NOISE_MAX - PRIOR_NOISE_MIN))

    regressor = fit_regressor(rows, targets, noises)

    # A GP fit on a handful of points is overconfident: its own std ignores
    # how wrong it is out of sample. Leave-one-out error is added to every
    # prediction's std so the surrogate's bets are honest about that.
    squared_errors = []
    if real_count >= 3:
        for left_out in range(real_count):
            other_rows = rows[:left_out] + rows[left_out + 1:]
            other_targets = targets[:left_out] + targets[left_out + 1:]
            other_noises = noises[:left_out] + noises[left_out + 1:]
            partial = fit_regressor(other_rows, other_targets, other_noises)
            prediction = partial.predict(numpy.array([rows[left_out]]))[0]
            squared_errors.append((prediction - targets[left_out]) ** 2)
    loo_std = 0.0
    if len(squared_errors) > 0:
        loo_std = math.sqrt(sum(squared_errors) / len(squared_errors))
    return {"regressor": regressor, "columns": columns, "loo_std": loo_std}


def fit_regressor(rows, targets, noises):
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-4)
    # alpha = per-point noise: this is how a rule's virtual point is trusted less.
    regressor = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=numpy.array(noises),
                                         n_restarts_optimizer=3, random_state=0)
    regressor.fit(numpy.array(rows), numpy.array(targets))
    return regressor


def predict(model, knobs):
    row = numpy.array([encode(knobs, model["columns"])])
    mean, std = model["regressor"].predict(row, return_std=True)
    total_std = math.sqrt(float(std[0]) ** 2 + model["loo_std"] ** 2)
    return float(mean[0]), max(total_std, MIN_STD)


def probability_at_least(model, knobs, threshold):
    mean, std = predict(model, knobs)
    z = (threshold - mean) / std
    return 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
