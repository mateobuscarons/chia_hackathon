"""Offline tests on the cached simulation rows. No simulator, no cost.

Anything the plan needs to know before a design is bought gets tested here
first, on the 2,644 per-workload rows already measured for chips A, B and C.
Every test uses the same protocol: fit on chips A and B, predict chip C, and
score on the objective the loop actually optimises - the geometric mean of IPC
over the four workloads, for designs measured on all four.

  ceiling      the CPI stack given the MEASURED miss counts of every row. This
               is its upper bound: it separates "is the miss-cost formula
               right" from "does the trace's miss-ratio curve predict the miss
               counts". If the formula fails here, no better curve saves it.
  timeliness   the same, plus one fitted number per prefetcher value, because a
               prefetcher makes misses cheaper rather than rarer. A
               measurement, not a decision.
  compare      the CPI stack against the GP surrogates on one protocol: the
               knob GP (what bo_pooled sees today), the physics-feature GP and
               the minimal-physics GP.
  misses       layer 1: does the trace's miss-ratio curve predict the per-level
               miss counts the simulator measured? Everything above assumes it
               does. Two candidate hierarchy mappings are measured side by
               side, split by whether a prefetcher was running.

  fair         the head-to-head the paper needs: the CPI stack fed PREDICTED
               miss counts, so it knows exactly as little about an unsimulated
               design as the GPs do.

  learned      the same, but the miss counts come from a model trained on A+B
               instead of from the miss-ratio curve. Miss counts are chip
               invariant, so that model has nothing to transfer; only the cost
               of a miss changes with the chip, and that is the formula.

  admit        the workload admission gate: from the free trace profile alone,
               can this workload teach anything about cache capacity? Validated
               against the measured variance shares on chip C.

Usage: python -m loop.offline ceiling | timeliness | compare | misses | fair | learned | admit
"""

import glob
import json
import math
import os
import sys

import numpy
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from loop import champsim_problem, cpi_stack, surrogates, trace_profile
from loop.configs import SEARCH_SPACE_C, config_name

SOCS = ["A_mobile", "B_midrange", "C_server"]
TRACES = ["605.mcf_s-665B", "619.lbm_s-2676B", "620.omnetpp_s-874B", "623.xalancbmk_s-700B"]
LEVELS = ["L1D", "L2C", "LLC"]
BASE_CONFIG = "champsim/champsim_config.json"
# The frozen Tier C reference: best suite geomean among the 447 designs measured
# on all four workloads (verified Sep 8; chip C baseline is 0.5990).
FIXED_REFERENCE = 0.7491

# One entry per free coefficient: (name, start, lower bound, upper bound).
BASE_COEFFICIENTS = [
    ("l2_cost", 1.0, 0.0, 5.0),
    ("llc_cost", 1.0, 0.0, 5.0),
    ("dram_cost", 1.0, 0.0, 5.0),
    ("overlap_base", 0.3, 0.0, 1.0),
    ("overlap_stride", 0.5, -2.0, 2.0),
]


def timeliness_coefficients():
    """One factor per prefetcher value; "no" is the reference and stays at zero."""
    specs = []
    for value in SEARCH_SPACE_C["l2_prefetcher"]:
        if value == "no":
            continue
        specs.append(("timeliness_l2_" + value, 0.5, 0.0, 20.0))
    for value in SEARCH_SPACE_C["llc_prefetcher"]:
        if value == "no":
            continue
        specs.append(("timeliness_llc_" + value, 0.2, 0.0, 20.0))
    return specs


# What chip C's space can actually buy, in KB, under its 4608 KB area cap. The
# LLC's miss-ratio curve is read at (L2 + LLC); the L2's at its own capacity.
BUYABLE_LLC_SMALLEST_KB = 256 * 4 * 64 / 1024.0 + 1024 * 8 * 64 / 1024.0
BUYABLE_LLC_LARGEST_KB = 4608.0
# A workload must be able to move at least this many misses per kilo-instruction
# by capacity alone, or no capacity decision made on it means anything.
MOVABLE_MPKI_FLOOR = 1.0
# Measured share of chip C's IPC variance owned by the two capacity knobs
# (Tier A full factorial). The gate has to agree with these or it is wrong.
CAPACITY_VARIANCE = {"605.mcf_s-665B": 0.46, "619.lbm_s-2676B": 0.01, "620.omnetpp_s-874B": 0.86}


def admit():
    """Can this workload teach anything about cache capacity? Free, no simulator.

    Two numbers decide it. `movable` is how many misses per kilo-instruction
    capacity alone can remove, from the smallest to the largest cache the search
    space can buy. `cold` is the share of its misses that are compulsory - a
    block touched for the first time in the measured window - which no design
    can ever remove. A workload with a big MPKI but nothing movable measures
    only its own cold misses, and a long enough window is the only fix.
    """
    print("== workload admission gate: chip C can buy an LLC read at {:.0f}-{:.0f} KB"
          .format(BUYABLE_LLC_SMALLEST_KB, BUYABLE_LLC_LARGEST_KB))
    print("  {:<12s} {:>10s} {:>9s} {:>9s} {:>8s} {:>9s} {:>8s}  {}".format(
        "trace", "footprint", "MPKI@min", "MPKI@max", "movable", "unfixable", "cap.var",
        "verdict"))
    for path in sorted(glob.glob("results/profile_*.json")):
        with open(path) as profile_file:
            profile = json.load(profile_file)
        name = os.path.basename(path).replace("profile_", "").replace(".json", "")
        accesses = profile["mem_accesses_per_kinstr"]
        smallest = trace_profile.predicted_miss_ratio(profile, BUYABLE_LLC_SMALLEST_KB)
        largest = trace_profile.predicted_miss_ratio(profile, BUYABLE_LLC_LARGEST_KB)
        movable = accesses * (smallest - largest)
        curve = profile["miss_ratio_curve"]
        floor = min(curve[size] for size in curve)
        # Compulsory misses: blocks touched for the first time in this window.
        # No cache on any chip removes them; only a longer window does.
        unfixable_mpki = accesses * floor

        if movable >= MOVABLE_MPKI_FLOOR:
            verdict = "ADMIT"
        else:
            verdict = "REJECT: capacity moves only {:.2f} MPKI".format(movable)
        known = CAPACITY_VARIANCE.get(name)
        if known is None:
            known_text = "       -"
        else:
            known_text = "{:8.2f}".format(known)
        print("  {:<12s} {:8.0f}KB {:9.1f} {:9.1f} {:8.2f} {:9.1f} {}  {}".format(
            name.split(".")[1].split("_")[0], profile["footprint_kb"], accesses * smallest,
            accesses * largest, movable, unfixable_mpki, known_text, verdict))
    print()
    print("  unfixable = compulsory MPKI in this window. When it swallows almost all of")
    print("  MPKI@min, the workload is being measured before it starts reusing anything,")
    print("  and the fix is a longer window, not a bigger cache. cap.var is the measured")
    print("  ground truth (chip C variance share of the capacity knobs) the gate must match.")


def knobs_equal(knobs, reference_knobs):
    for knob in reference_knobs:
        if str(knobs.get(knob)) != str(reference_knobs[knob]):
            return False
    return True


def level_mpki_of(metrics):
    values = {}
    for level in LEVELS:
        values[level] = metrics[level + "_mpki"]
    return values


def load_profiles():
    profiles = {}
    for trace in TRACES:
        with open("results/profile_{}.json".format(trace)) as profile_file:
            profiles[trace] = json.load(profile_file)
    return profiles


def load_rows():
    """Every cached (chip, workload, design) row, carrying its own chip and
    workload baseline so a design can be scored relative to the untouched chip."""
    profiles = load_profiles()
    rows = []
    for soc in SOCS:
        baseline_knobs = champsim_problem.profiled_baseline(soc, "C")
        for trace in TRACES:
            path = "results/tierC_{}_{}.json".format(soc, trace)
            with open(path) as table_file:
                table = json.load(table_file)
            baseline_ipc = None
            baseline_mpki = None
            for name in table:
                entry = table[name]
                if entry["metrics"] is None:
                    continue
                if knobs_equal(entry["knobs"], baseline_knobs):
                    baseline_ipc = entry["metrics"]["ipc"]
                    baseline_mpki = level_mpki_of(entry["metrics"])
            if baseline_ipc is None:
                raise SystemExit("no baseline row for {} {}".format(soc, trace))
            for name in table:
                entry = table[name]
                if entry["metrics"] is None or "ipc" not in entry["metrics"]:
                    continue
                rows.append({
                    "soc": soc,
                    "trace": trace,
                    "knobs": entry["knobs"],
                    "ipc": entry["metrics"]["ipc"],
                    "level_mpki": level_mpki_of(entry["metrics"]),
                    "baseline_ipc": baseline_ipc,
                    "baseline_knobs": baseline_knobs,
                    "baseline_level_mpki": baseline_mpki,
                    "profile": profiles[trace],
                })
    return rows


def split_by_chip(rows):
    train_rows = []
    test_rows = []
    for row in rows:
        if row["soc"] == "C_server":
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def true_log_speedup(row):
    return math.log(row["ipc"] / row["baseline_ipc"])


def truths_of(rows):
    values = []
    for row in rows:
        values.append(true_log_speedup(row))
    return numpy.array(values)


def chips_for(rows):
    chips = {}
    for row in rows:
        if row["soc"] not in chips:
            chips[row["soc"]] = cpi_stack.chip_parameters(row["soc"], BASE_CONFIG)
    return chips


def coefficients_from(vector, specs):
    coefficients = {}
    for index in range(len(specs)):
        coefficients[specs[index][0]] = vector[index]
    return coefficients


def stack_predictions(rows, chips, coefficients):
    predictions = []
    for row in rows:
        predictions.append(cpi_stack.predicted_log_speedup(row, chips[row["soc"]], coefficients))
    return numpy.array(predictions)


def fit_coefficients(rows, specs=None):
    """Least squares on log speedup: a handful of numbers, shared by every chip
    and program. `specs` selects which coefficients are free."""
    if specs is None:
        specs = BASE_COEFFICIENTS
    chips = chips_for(rows)
    truths = truths_of(rows)

    starts = []
    lower_bounds = []
    upper_bounds = []
    for spec in specs:
        starts.append(spec[1])
        lower_bounds.append(spec[2])
        upper_bounds.append(spec[3])

    def residuals(vector):
        coefficients = coefficients_from(vector, specs)
        return stack_predictions(rows, chips, coefficients) - truths

    solution = least_squares(residuals, starts, bounds=(lower_bounds, upper_bounds))
    return coefficients_from(solution.x, specs)


def per_workload_line(label, rows, predictions):
    truths = truths_of(rows)
    mae = float(numpy.abs(predictions - truths).mean())
    rank_correlation = float(spearmanr(predictions, truths).correlation)
    print("  {:26s} n={:5d}  MAE={:.4f}  spearman={:.3f}".format(
        label, len(rows), mae, rank_correlation))


def within_workload_line(label, test_rows, predictions):
    """Rank quality inside one workload, averaged over the four.

    The pooled number flatters a model that only learns which workload it is
    looking at: mcf's speedups span -0.31..+0.51 and xalancbmk's -0.83..+0.02,
    so getting the four levels right already buys most of the pooled rank.
    """
    correlations = []
    for trace in TRACES:
        trace_predictions = []
        trace_truths = []
        for index in range(len(test_rows)):
            if test_rows[index]["trace"] != trace:
                continue
            trace_predictions.append(predictions[index])
            trace_truths.append(true_log_speedup(test_rows[index]))
        correlations.append(float(spearmanr(trace_predictions, trace_truths).correlation))
    average = sum(correlations) / len(correlations)
    print("  {:26s} mean within-workload spearman={:.3f}   per workload:".format(label, average),
          end="")
    for index in range(len(TRACES)):
        print("  {}={:.3f}".format(TRACES[index].split(".")[1].split("_")[0], correlations[index]),
              end="")
    print()


def suite_line(label, test_rows, predictions):
    """The real objective: one design scored on all four workloads at once.

    The suite score is the geometric mean of per-workload IPC, so in log space
    it is the plain mean of the four per-workload log speedups. Only designs
    measured on all four workloads count.
    """
    predicted_parts = {}
    true_parts = {}
    ipc_parts = {}
    for index in range(len(test_rows)):
        row = test_rows[index]
        name = config_name(row["knobs"], row["soc"])
        if name not in predicted_parts:
            predicted_parts[name] = []
            true_parts[name] = []
            ipc_parts[name] = []
        predicted_parts[name].append(predictions[index])
        true_parts[name].append(true_log_speedup(row))
        ipc_parts[name].append(math.log(row["ipc"]))

    suite_predictions = []
    suite_truths = []
    suite_ipcs = []
    for name in predicted_parts:
        if len(predicted_parts[name]) != len(TRACES):
            continue
        suite_predictions.append(sum(predicted_parts[name]) / len(TRACES))
        suite_truths.append(sum(true_parts[name]) / len(TRACES))
        suite_ipcs.append(math.exp(sum(ipc_parts[name]) / len(TRACES)))

    suite_predictions = numpy.array(suite_predictions)
    suite_truths = numpy.array(suite_truths)
    suite_ipcs = numpy.array(suite_ipcs)
    rank_correlation = float(spearmanr(suite_predictions, suite_truths).correlation)
    order = numpy.argsort(-suite_predictions)

    best_in_top = []
    for top in [1, 5, 10]:
        picked = order[:top]
        best_in_top.append(float(suite_ipcs[picked].max()))
    top_rank = int((suite_truths > suite_truths[int(order[0])]).sum())

    print("  {:26s} n={:4d}  spearman={:.3f}  best real IPC in top1/5/10 = "
          "{:.4f}/{:.4f}/{:.4f}  ({:.1f}%/{:.1f}%/{:.1f}% of reference)  rank of top pick={}".format(
              label, len(suite_predictions), rank_correlation,
              best_in_top[0], best_in_top[1], best_in_top[2],
              100.0 * best_in_top[0] / FIXED_REFERENCE,
              100.0 * best_in_top[1] / FIXED_REFERENCE,
              100.0 * best_in_top[2] / FIXED_REFERENCE,
              top_rank))


# Two readings of "how much cache does this level see". Mapping "own" gives each
# level its own capacity; "nested" gives it everything above it as well.
CAPACITY_MAPPINGS = ["own", "nested"]


def predicted_level_mpki(knobs, profile, mapping):
    """Miss counts per kilo-instruction, from the miss-ratio curve alone."""
    l1d_kb = surrogates.capacity_kb(knobs, "l1d")
    l2_kb = surrogates.capacity_kb(knobs, "l2")
    llc_kb = surrogates.capacity_kb(knobs, "llc")
    if mapping == "own":
        capacities = {"L1D": l1d_kb, "L2C": l2_kb, "LLC": l2_kb + llc_kb}
    else:
        capacities = {"L1D": l1d_kb, "L2C": l1d_kb + l2_kb, "LLC": l1d_kb + l2_kb + llc_kb}
    accesses = profile["mem_accesses_per_kinstr"]
    predicted = {}
    for level in LEVELS:
        miss_ratio = trace_profile.predicted_miss_ratio(profile, capacities[level])
        predicted[level] = accesses * miss_ratio
    return predicted


def prefetching_on(knobs):
    for knob in ["l1d_prefetcher", "l2_prefetcher", "llc_prefetcher"]:
        if str(knobs[knob]) != "no":
            return True
    return False


def misses():
    """Layer 1, one workload and one level at a time.

    `ratio` is measured / predicted: 1.0 would mean the curve is exact, 2.0
    that the simulator sees twice the misses the curve promises. `spearman` is
    what the mean function actually needs - whether the curve orders designs
    the way the simulator does, inside one workload.
    """
    rows = load_rows()
    print("== layer 1: miss-ratio curve versus the simulator's miss counts, chip C")
    for mapping in CAPACITY_MAPPINGS:
        print("  -- capacity mapping: {}".format(mapping))
        for level in LEVELS:
            for trace in TRACES:
                for prefetching in [False, True]:
                    measured_values = []
                    predicted_values = []
                    for row in rows:
                        if row["soc"] != "C_server" or row["trace"] != trace:
                            continue
                        if prefetching_on(row["knobs"]) != prefetching:
                            continue
                        measured_values.append(row["level_mpki"][level])
                        predicted = predicted_level_mpki(row["knobs"], row["profile"], mapping)
                        predicted_values.append(predicted[level])
                    if len(measured_values) < 8:
                        continue
                    measured_array = numpy.array(measured_values)
                    predicted_array = numpy.array(predicted_values)
                    ratio = float(measured_array.mean() / max(predicted_array.mean(), 1e-9))
                    rank = float(spearmanr(predicted_array, measured_array).correlation)
                    if prefetching:
                        state = "prefetch on "
                    else:
                        state = "prefetch off"
                    print("    {:4s} {:10s} {}  n={:4d}  measured={:6.1f}  predicted={:6.1f}  "
                          "ratio={:5.2f}  spearman={:+.3f}".format(
                              level, trace.split(".")[1].split("_")[0], state,
                              len(measured_values), measured_array.mean(),
                              predicted_array.mean(), ratio, rank))


def rows_with_predicted_misses(rows, mapping="own"):
    """The same rows with the simulator's miss counts replaced by the trace
    curve's. This is what the loop can actually know about a design it has not
    run: only the baseline row keeps its measured counts, because the baseline
    is simulated on every chip before anything else."""
    predicted_rows = []
    for row in rows:
        predicted_row = dict(row)
        predicted_row["level_mpki"] = predicted_level_mpki(row["knobs"], row["profile"], mapping)
        predicted_row["baseline_level_mpki"] = predicted_level_mpki(
            row["baseline_knobs"], row["profile"], mapping)
        predicted_rows.append(predicted_row)
    return predicted_rows


def fair():
    """Every model given the same knowledge: knobs, the chip's spec, the trace
    profile, and the chip's baseline run. Nothing measured about the design."""
    rows = load_rows()
    train_rows, test_rows = split_by_chip(rows)
    chips = chips_for(rows)
    train_targets = truths_of(train_rows)

    print("== fair head-to-head: fit on A+B (n={}), predict C (n={}), suite objective"
          .format(len(train_rows), len(test_rows)))
    print("   reference = {:.4f}; chip C baseline = 0.5990".format(FIXED_REFERENCE))

    names = []
    predictions_by_name = {}
    for name in ["knob", "minimal"]:
        featurizer = surrogates.FEATURIZERS[name]
        label = name + " GP"
        names.append(label)
        predictions_by_name[label] = surrogates.fit_and_predict(
            train_rows, test_rows, chips, featurizer, train_targets)

    predicted_train = rows_with_predicted_misses(train_rows)
    predicted_test = rows_with_predicted_misses(test_rows)
    coefficients = fit_coefficients(predicted_train, BASE_COEFFICIENTS)
    names.append("CPI stack (predicted)")
    predictions_by_name["CPI stack (predicted)"] = stack_predictions(
        predicted_test, chips, coefficients)

    oracle_coefficients = fit_coefficients(train_rows, BASE_COEFFICIENTS)
    names.append("CPI stack (measured)")
    predictions_by_name["CPI stack (measured)"] = stack_predictions(
        test_rows, chips, oracle_coefficients)

    print("  fitted on predicted counts:", end="")
    for spec in BASE_COEFFICIENTS:
        print("  {}={:.3f}".format(spec[0], coefficients[spec[0]]), end="")
    print()
    print("  -- within one workload at a time")
    for label in names:
        within_workload_line(label, test_rows, predictions_by_name[label])
    print("  -- suite objective")
    for label in names:
        suite_line(label, test_rows, predictions_by_name[label])


def learned_miss_counts(train_rows, test_rows):
    """One model per level, trained on the old chips, predicting log MPKI.

    Log, because a miss count is multiplicative: half the misses matters the
    same whether the level sees 10 or 100 per kilo-instruction.
    """
    predicted_by_level = {}
    for level in LEVELS:
        targets = []
        for row in train_rows:
            targets.append(math.log(max(row["level_mpki"][level], 1e-3)))
        predictions = surrogates.fit_and_predict_targets(
            train_rows, test_rows, surrogates.miss_count_features, targets)
        predicted_by_level[level] = numpy.exp(predictions)
    return predicted_by_level


def rows_with_learned_misses(train_rows, test_rows):
    """Test rows whose miss counts come from the model, not the simulator.

    A baseline row keeps its measured counts: the baseline design is simulated
    on every chip before the search starts.
    """
    predicted_by_level = learned_miss_counts(train_rows, test_rows)
    learned_rows = []
    for index in range(len(test_rows)):
        row = test_rows[index]
        learned_row = dict(row)
        level_mpki = {}
        for level in LEVELS:
            level_mpki[level] = float(predicted_by_level[level][index])
        learned_row["level_mpki"] = level_mpki
        learned_rows.append(learned_row)
    return learned_rows


def miss_count_quality(test_rows, learned_rows):
    print("  -- how good the learned miss counts are on chip C")
    for level in LEVELS:
        measured = []
        learned = []
        for index in range(len(test_rows)):
            measured.append(test_rows[index]["level_mpki"][level])
            learned.append(learned_rows[index]["level_mpki"][level])
        measured_array = numpy.array(measured)
        learned_array = numpy.array(learned)
        relative = numpy.abs(learned_array - measured_array) / numpy.maximum(measured_array, 1e-9)
        rank = float(spearmanr(learned_array, measured_array).correlation)
        print("    {:4s} mean |relative error| = {:5.1f}%   spearman = {:.3f}".format(
            level, 100.0 * float(relative.mean()), rank))


def learned():
    """The strongest version of the mechanism: learn the misses, compute the cost."""
    rows = load_rows()
    train_rows, test_rows = split_by_chip(rows)
    chips = chips_for(rows)
    train_targets = truths_of(train_rows)

    print("== learn the misses, compute the cost: fit on A+B (n={}), predict C (n={})"
          .format(len(train_rows), len(test_rows)))
    print("   reference = {:.4f}; chip C baseline = 0.5990".format(FIXED_REFERENCE))

    names = []
    predictions_by_name = {}
    for name in ["knob", "minimal"]:
        label = name + " GP"
        names.append(label)
        predictions_by_name[label] = surrogates.fit_and_predict(
            train_rows, test_rows, chips, surrogates.FEATURIZERS[name], train_targets)

    coefficients = fit_coefficients(train_rows, BASE_COEFFICIENTS)
    learned_rows = rows_with_learned_misses(train_rows, test_rows)
    miss_count_quality(test_rows, learned_rows)
    names.append("stack + learned misses")
    predictions_by_name["stack + learned misses"] = stack_predictions(
        learned_rows, chips, coefficients)

    curve_rows = rows_with_predicted_misses(test_rows)
    names.append("stack + curve misses")
    predictions_by_name["stack + curve misses"] = stack_predictions(
        curve_rows, chips, fit_coefficients(rows_with_predicted_misses(train_rows),
                                            BASE_COEFFICIENTS))

    names.append("stack + measured (oracle)")
    predictions_by_name["stack + measured (oracle)"] = stack_predictions(
        test_rows, chips, coefficients)

    print("  -- within one workload at a time")
    for label in names:
        within_workload_line(label, test_rows, predictions_by_name[label])
    print("  -- suite objective")
    for label in names:
        suite_line(label, test_rows, predictions_by_name[label])


def ceiling(specs=None, title="step 1: CPI stack on MEASURED miss counts"):
    if specs is None:
        specs = BASE_COEFFICIENTS
    rows = load_rows()
    train_rows, test_rows = split_by_chip(rows)
    chips = chips_for(rows)

    coefficients = fit_coefficients(train_rows, specs)
    print("== {}, fit on A+B, tested on C".format(title))
    for spec in specs:
        print("  {:26s} = {:7.3f}".format(spec[0], coefficients[spec[0]]))

    print("  -- per workload")
    per_workload_line("train A+B (all)", train_rows,
                      stack_predictions(train_rows, chips, coefficients))
    test_predictions = stack_predictions(test_rows, chips, coefficients)
    per_workload_line("test C (all)", test_rows, test_predictions)
    for trace in TRACES:
        per_trace_rows = []
        per_trace_predictions = []
        for index in range(len(test_rows)):
            if test_rows[index]["trace"] == trace:
                per_trace_rows.append(test_rows[index])
                per_trace_predictions.append(test_predictions[index])
        per_workload_line("test C " + trace.split(".")[1].split("_")[0], per_trace_rows,
                          numpy.array(per_trace_predictions))
    print("  -- suite objective (the one the loop optimises)")
    suite_line("test C suite", test_rows, test_predictions)


def compare():
    """The head-to-head: every surrogate on one protocol and one objective."""
    rows = load_rows()
    train_rows, test_rows = split_by_chip(rows)
    chips = chips_for(rows)
    train_targets = truths_of(train_rows)

    print("== fit on A+B (n={}), predict C (n={}), scored on the suite objective"
          .format(len(train_rows), len(test_rows)))
    print("   reference = {:.4f} (best of the 447 cached designs); chip C baseline = 0.5990"
          .format(FIXED_REFERENCE))

    names = []
    predictions_by_name = {}
    for name in ["knob", "physics", "minimal"]:
        featurizer = surrogates.FEATURIZERS[name]
        label = name + " GP"
        names.append(label)
        predictions_by_name[label] = surrogates.fit_and_predict(
            train_rows, test_rows, chips, featurizer, train_targets)

    coefficients = fit_coefficients(train_rows, BASE_COEFFICIENTS)
    names.append("CPI stack")
    predictions_by_name["CPI stack"] = stack_predictions(test_rows, chips, coefficients)

    print("  -- per workload, pooled over the four (mixes workloads: see below)")
    for label in names:
        per_workload_line(label, test_rows, predictions_by_name[label])
    print("  -- WITHIN one workload at a time (the honest per-workload rank)")
    for label in names:
        within_workload_line(label, test_rows, predictions_by_name[label])
    print("  -- suite objective")
    for label in names:
        suite_line(label, test_rows, predictions_by_name[label])


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "ceiling"
    if step == "ceiling":
        ceiling()
    elif step == "timeliness":
        ceiling(BASE_COEFFICIENTS + timeliness_coefficients(),
                "step 1b: same stack plus a fitted timeliness factor per prefetcher")
    elif step == "compare":
        compare()
    elif step == "misses":
        misses()
    elif step == "fair":
        fair()
    elif step == "learned":
        learned()
    elif step == "admit":
        admit()
    else:
        raise SystemExit("unknown step: " + step)
