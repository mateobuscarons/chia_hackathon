"""The candidate surrogates, all answering the same question.

Every one of them predicts the same target - the log speedup of a design over
the chip's own baseline run for one workload - so they can be compared on one
protocol: fit on chips A and B, predict chip C.

  knob     what bo_pooled sees today: the knob columns plus which chip it is.
  physics  the trace profile's own numbers plus the predicted miss ratio at
           each of this design's cache sizes.
  minimal  only what changes with the design: predicted miss-ratio deltas
           against the chip's baseline, so the model cannot memorise which
           program this is.

The CPI stack lives in loop.cpi_stack; it is not a feature set, it is a formula.
"""

import math

import numpy

from loop import surrogate_gp, trace_profile
from loop.configs import SEARCH_SPACE_C

SOCS = ["A_mobile", "B_midrange", "C_server"]
POLICY_KNOBS = ["l1d_prefetcher", "l2_prefetcher", "llc_prefetcher", "llc_replacement"]


def capacity_kb(knobs, level):
    return knobs[level + "_sets"] * knobs[level + "_ways"] * 64 / 1024.0


def design_capacities(knobs):
    """The three capacities a miss-ratio curve is read at: the L1D on its own,
    the L2 on its own, and everything the LLC backs up."""
    l1d_kb = capacity_kb(knobs, "l1d")
    l2_kb = capacity_kb(knobs, "l2")
    llc_kb = capacity_kb(knobs, "llc")
    return [l1d_kb, l2_kb, l2_kb + llc_kb]


def policy_columns(knobs):
    columns = []
    for knob in POLICY_KNOBS:
        for value in SEARCH_SPACE_C[knob]:
            if str(knobs[knob]) == str(value):
                columns.append(1.0)
            else:
                columns.append(0.0)
    return columns


def depth_columns(knobs):
    columns = []
    columns.append(math.log2(knobs["l2_mshr"]) / 6.0)
    columns.append(math.log2(knobs["llc_mshr"]) / 7.0)
    return columns


def chip_columns(chip):
    columns = []
    columns.append(chip["rob_size"] / 512.0)
    columns.append(chip["dram_cycles"] / 200.0)
    return columns


def physics_features(row, chip):
    profile = row["profile"]
    knobs = row["knobs"]
    features = []
    # What the program does, whatever chip it runs on.
    features.append(math.log2(profile["mem_accesses_per_kinstr"] + 1) / 10.0)
    features.append(profile["write_fraction"])
    features.append(profile["stride_regular_fraction"])
    features.append(profile["reuse_local_fraction"])
    features.append(math.log2(profile["footprint_kb"] + 1) / 16.0)
    # How this design's geometry treats it, straight from the miss-ratio curve.
    for level_kb in design_capacities(knobs):
        features.append(trace_profile.predicted_miss_ratio(profile, level_kb))
        ratio = profile["footprint_kb"] / level_kb
        features.append(min(math.log2(ratio + 1e-9), 8.0) / 8.0)
    # The same for the chip's untouched geometry: the target is relative to it.
    for level_kb in design_capacities(row["baseline_knobs"]):
        features.append(trace_profile.predicted_miss_ratio(profile, level_kb))
    features = features + depth_columns(knobs)
    features = features + policy_columns(knobs)
    features = features + chip_columns(chip)
    return features


def minimal_features(row, chip):
    profile = row["profile"]
    knobs = row["knobs"]
    features = []
    design_kbs = design_capacities(knobs)
    baseline_kbs = design_capacities(row["baseline_knobs"])
    for index in range(len(design_kbs)):
        design_miss = trace_profile.predicted_miss_ratio(profile, design_kbs[index])
        baseline_miss = trace_profile.predicted_miss_ratio(profile, baseline_kbs[index])
        features.append(design_miss - baseline_miss)
        features.append(baseline_miss)
    features.append(math.log2(profile["mem_accesses_per_kinstr"] + 1) / 10.0)
    features = features + depth_columns(knobs)
    features = features + policy_columns(knobs)
    features = features + chip_columns(chip)
    return features


def knob_features(row, chip):
    space = dict(SEARCH_SPACE_C)
    space["soc"] = SOCS
    columns = surrogate_gp.build_columns(space)
    knobs = dict(row["knobs"])
    knobs["soc"] = row["soc"]
    return surrogate_gp.encode(knobs, columns)


FEATURIZERS = {"knob": knob_features, "physics": physics_features, "minimal": minimal_features}


def fit_and_predict(train_rows, test_rows, chips, featurizer, targets):
    """One GP, fitted on the training rows, asked about the test rows."""
    train_x = []
    for row in train_rows:
        train_x.append(featurizer(row, chips[row["soc"]]))
    noises = []
    for row in train_rows:
        noises.append(surrogate_gp.REAL_NOISE)
    dimension = len(train_x[0])
    regressor = surrogate_gp.fit_regressor(train_x, targets, noises, dimension, restarts=1)
    test_x = []
    for row in test_rows:
        test_x.append(featurizer(row, chips[row["soc"]]))
    return numpy.array(regressor.predict(numpy.array(test_x)))


def miss_count_features(row, chip):
    """Features for predicting a design's MISS COUNTS, not its speed.

    Deliberately blind to the chip: the same design on chips A, B and C misses
    within a few percent of the same number of times (measured Sep 8: 1.8-4.2%
    between B and C, rank correlation 0.99), while its IPC moves 26%. So the
    miss-count model has nothing to transfer - it is the same function on every
    chip - and every chip-dependent effect is left to the CPI stack's formula.
    """
    profile = row["profile"]
    knobs = row["knobs"]
    features = []
    features.append(math.log2(profile["mem_accesses_per_kinstr"] + 1) / 10.0)
    features.append(profile["write_fraction"])
    features.append(profile["stride_regular_fraction"])
    features.append(profile["reuse_local_fraction"])
    features.append(math.log2(profile["footprint_kb"] + 1) / 16.0)
    # The miss-ratio curve read at this design's capacities: the strong prior.
    for level_kb in design_capacities(knobs):
        features.append(trace_profile.predicted_miss_ratio(profile, level_kb))
        features.append(math.log2(level_kb) / 16.0)
    features = features + depth_columns(knobs)
    features = features + policy_columns(knobs)
    return features


def fit_and_predict_targets(train_rows, test_rows, featurizer, targets):
    """Same as fit_and_predict but for a target that is not log speedup, and
    for a featurizer that does not need the chip."""
    train_x = []
    for row in train_rows:
        train_x.append(featurizer(row, None))
    noises = []
    for row in train_rows:
        noises.append(surrogate_gp.REAL_NOISE)
    regressor = surrogate_gp.fit_regressor(train_x, targets, noises, len(train_x[0]), restarts=1)
    test_x = []
    for row in test_rows:
        test_x.append(featurizer(row, None))
    return numpy.array(regressor.predict(numpy.array(test_x)))
