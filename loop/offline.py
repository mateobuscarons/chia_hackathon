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

  headroom     gate G-headroom: on the simulated rows of the held-out suite, is
               there any IPC spread for a search to find? And - the real prize -
               does the admission gate's per-workload prediction (which knob
               family has leverage) match what the simulator says?
  coverage     the descriptor-range table: does each held-out workload sit
               INSIDE the range the training suite covers? A condition fired
               inside the training range is interpolating; outside it, the rule
               is guessing and the paper must say so.
  admit        the workload admission gate: from the free trace profile alone,
               can this workload teach anything about cache capacity? Validated
               against the measured variance shares on chip C.

Usage: python -m loop.offline ceiling | timeliness | compare | misses | fair | learned
                            | admit | coverage | headroom
"""

import glob
import json
import math
import random
import os
import sys

import numpy
from scipy.optimize import least_squares
from scipy.stats import spearmanr

from loop import champsim_problem, cpi_stack, surrogates, trace_profile
from loop.configs import SEARCH_SPACE, config_name

SOCS = ["A_mobile", "B_midrange", "C_server"]
TRACES = ["605.mcf_s-665B", "619.lbm_s-2676B", "620.omnetpp_s-874B", "623.xalancbmk_s-700B"]
LEVELS = ["L1D", "L2C", "LLC"]
BASE_CONFIG = "champsim/champsim_config.json"
# The frozen Tier C reference: best suite geomean among the 447 designs measured
# on all four workloads (chip C baseline is 0.5990).
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
    for value in SEARCH_SPACE["l2_prefetcher"]:
        if value == "no":
            continue
        specs.append(("timeliness_l2_" + value, 0.5, 0.0, 20.0))
    for value in SEARCH_SPACE["llc_prefetcher"]:
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
# Below this many misses per kilo-instruction there is nothing for any knob to
# fix, whatever family it belongs to.
POLICY_MPKI_FLOOR = 5.0
# Measured share of chip C's IPC variance owned by the two capacity knobs
# (dense 81-design sweep). The gate has to agree with these or it is wrong.
CAPACITY_VARIANCE = {"605.mcf_s-665B": 0.46, "619.lbm_s-2676B": 0.01, "620.omnetpp_s-874B": 0.86}
# The same, for the L2 prefetcher knob.
PREFETCHER_VARIANCE = {"605.mcf_s-665B": 0.46, "619.lbm_s-2676B": 0.96, "620.omnetpp_s-874B": 0.00}


def admit():
    """Can this workload teach anything about cache capacity? Free, no simulator.

    Two channels, because a workload can be worth simulating for two reasons.

    The CAPACITY channel is `movable`: how many misses per kilo-instruction
    capacity alone can remove, from the smallest to the largest cache the search
    space can buy. This channel is validated - its verdicts match the measured
    variance shares on chip C.

    The POLICY channel is just a miss floor: with enough misses there is
    something for a prefetcher or a replacement policy to fix, even when no
    cache size helps. This channel deliberately does NOT claim to know WHICH
    policy. Stride regularity is reported because it is suggestive, but it is
    not trustworthy on its own: mcf's stride regularity is only 0.157 and yet
    the L2 prefetcher owns 0.46 of its variance on chip C, because SPP learns
    repeated irregular paths that a stride count cannot see.

    `unfixable` is the compulsory MPKI - blocks touched for the first time in
    the measured window, which no design removes.
    """
    print("== workload admission gate: chip C can buy an LLC read at {:.0f}-{:.0f} KB"
          .format(BUYABLE_LLC_SMALLEST_KB, BUYABLE_LLC_LARGEST_KB))
    print("  {:<12s} {:>9s} {:>8s} {:>8s} {:>9s} {:>7s} {:>7s} {:>7s}  {}".format(
        "trace", "footprint", "MPKI", "movable", "unfixable", "stride", "cap.var", "pf.var",
        "admit as"))
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

        channels = []
        if movable >= MOVABLE_MPKI_FLOOR:
            channels.append("capacity")
        if accesses * smallest >= POLICY_MPKI_FLOOR:
            channels.append("policy")
        if len(channels) == 0:
            admitted_as = "REJECT (nothing any knob can fix)"
        else:
            admitted_as = " + ".join(channels)

        capacity_known = CAPACITY_VARIANCE.get(name)
        prefetcher_known = PREFETCHER_VARIANCE.get(name)
        if capacity_known is None:
            capacity_text = "      -"
        else:
            capacity_text = "{:7.2f}".format(capacity_known)
        if prefetcher_known is None:
            prefetcher_text = "      -"
        else:
            prefetcher_text = "{:7.2f}".format(prefetcher_known)
        short_name = name.split(".")[1].split("_")[0]
        if name.startswith("bfs.") or name.startswith("pr."):
            short_name = name.split("-")[0]
        print("  {:<12s} {:7.0f}KB {:8.1f} {:8.2f} {:9.1f} {:7.3f} {} {}  {}".format(
            short_name, profile["footprint_kb"], accesses * smallest, movable, unfixable_mpki,
            profile["stride_regular_fraction"], capacity_text, prefetcher_text, admitted_as))
    print()
    print("  admit as capacity  -> the capacity knobs have leverage here (validated channel)")
    print("  admit as policy    -> enough misses for a prefetcher or replacement policy to")
    print("                        matter, WITHOUT claiming which one; the simulator says.")
    print("  cap.var / pf.var   = measured chip C variance shares of the capacity knobs and")
    print("                        the L2 prefetcher (dense 81-design sweep), the ground truth.")
    print("  stride is reported but NOT used to decide: mcf is 0.157 and still 0.46 pf.var.")


# The suites, decided from the admission gate above.
TRAINING_SUITE = ["605.mcf_s-665B", "620.omnetpp_s-874B", "619.lbm_s-2676B",
                  "649.fotonik3d_s-10881B", "627.cam4_s-490B"]
HELD_OUT_SUITE = ["bfs.urand-36B", "pr.urand-129B", "bfs.kron-128B"]
COVERAGE_DESCRIPTORS = ["footprint_kb", "mem_accesses_per_kinstr", "stride_regular_fraction",
                        "reuse_local_fraction", "write_fraction"]


# What the admission gate predicted for the held-out suite, written down before
# a single design was simulated on it (decision 11: the gate is a prediction
# instrument, not a selection filter).
GATE_PREDICTION = {
    "bfs.urand-36B": "capacity should move it (movable 33.6 MPKI)",
    "pr.urand-129B": "capacity should NOT move it (movable 0.00); prefetching should",
    "bfs.kron-128B": "capacity should NOT move it (movable 0.01)",
}
CAPACITY_KNOBS = ["l2_sets", "l2_ways", "llc_sets", "llc_ways"]
POLICY_KNOBS_ALL = ["l1d_prefetcher", "l2_prefetcher", "llc_prefetcher", "llc_replacement"]


def one_knob_difference(knobs, baseline_knobs):
    """The single knob this design changes from the baseline, or None."""
    changed = []
    for knob in baseline_knobs:
        if str(knobs.get(knob)) != str(baseline_knobs[knob]):
            changed.append(knob)
    if len(changed) == 1:
        return changed[0]
    return None


def headroom():
    """Does the held-out suite have anything for a search to find, and does the
    gate's prediction hold once the simulator has spoken?"""
    baseline_knobs = champsim_problem.profiled_baseline("C_server", "C")
    baseline_name = config_name(baseline_knobs, "C_server")

    tables = {}
    for trace in HELD_OUT_SUITE:
        path = "results/tierC_C_server_{}.json".format(trace)
        with open(path) as table_file:
            tables[trace] = json.load(table_file)

    print("== gate G-headroom: chip C, held-out suite")
    print("   {:<16s} {:>9s} {:>9s} {:>9s} {:>9s}  {}".format(
        "workload", "baseline", "worst", "best", "spread", "designs"))
    baseline_ipc = {}
    for trace in HELD_OUT_SUITE:
        table = tables[trace]
        values = []
        for name in table:
            metrics = table[name]["metrics"]
            if metrics is None or "ipc" not in metrics:
                continue
            values.append(metrics["ipc"])
        if baseline_name in table and table[baseline_name]["metrics"] is not None:
            baseline_ipc[trace] = table[baseline_name]["metrics"]["ipc"]
        else:
            baseline_ipc[trace] = None
        if len(values) == 0:
            print("   {:<16s} no rows yet".format(trace))
            continue
        spread = 100.0 * (max(values) / min(values) - 1.0)
        if baseline_ipc[trace] is None:
            baseline_text = "        -"
        else:
            baseline_text = "{:9.4f}".format(baseline_ipc[trace])
        print("   {:<16s} {} {:9.4f} {:9.4f} {:8.1f}% {:9d}".format(
            trace, baseline_text, min(values), max(values), spread, len(values)))

    # The suite score: geometric mean over the three workloads.
    names_in_all = None
    for trace in HELD_OUT_SUITE:
        usable = set()
        for name in tables[trace]:
            if tables[trace][name]["metrics"] is not None:
                usable.add(name)
        if names_in_all is None:
            names_in_all = usable
        else:
            names_in_all = names_in_all & usable
    suite_scores = {}
    for name in names_in_all:
        total = 0.0
        for trace in HELD_OUT_SUITE:
            total += math.log(tables[trace][name]["metrics"]["ipc"])
        suite_scores[name] = math.exp(total / len(HELD_OUT_SUITE))
    if len(suite_scores) == 0:
        print("   no design measured on all three yet")
        return
    best_name = max(suite_scores, key=lambda n: suite_scores[n])
    print()
    print("   suite geomean over {} designs measured on all three:".format(len(suite_scores)))
    if baseline_name in suite_scores:
        base = suite_scores[baseline_name]
        print("     baseline {:.4f}, best {:.4f} -> headroom {:.1f}%".format(
            base, suite_scores[best_name], 100.0 * (suite_scores[best_name] / base - 1.0)))
    else:
        print("     best {:.4f} (baseline not measured on all three)".format(
            suite_scores[best_name]))

    # The pre-registered check: which knob family actually moves each workload?
    print()
    print("== does the gate's prediction hold? one-knob effects versus the baseline")
    for trace in HELD_OUT_SUITE:
        if baseline_ipc[trace] is None:
            continue
        print("   {} - gate said: {}".format(trace, GATE_PREDICTION[trace]))
        best_capacity = 0.0
        best_policy = 0.0
        for name in tables[trace]:
            metrics = tables[trace][name]["metrics"]
            if metrics is None or "ipc" not in metrics:
                continue
            knob = one_knob_difference(tables[trace][name]["knobs"], baseline_knobs)
            if knob is None:
                continue
            effect = 100.0 * (metrics["ipc"] / baseline_ipc[trace] - 1.0)
            if knob in CAPACITY_KNOBS and effect > best_capacity:
                best_capacity = effect
            if knob in POLICY_KNOBS_ALL and effect > best_policy:
                best_policy = effect
        print("     best single capacity change: {:+.1f}%   best single policy change: {:+.1f}%"
              .format(best_capacity, best_policy))


def coverage():
    """Is the held-out class inside the training suite's descriptor ranges?

    Every rule condition is written on these descriptors. A condition that
    fires on a held-out workload whose descriptor sits inside the training
    range is interpolating, which is a fair thing to expect it to get right.
    Outside the range it is extrapolating, and the paper has to admit it.
    """
    profiles = {}
    for name in TRAINING_SUITE + HELD_OUT_SUITE:
        with open("results/profile_{}.json".format(name)) as profile_file:
            profiles[name] = json.load(profile_file)

    print("== descriptor coverage: training suite range vs each held-out workload")
    print("   training suite:", ", ".join(TRAINING_SUITE))
    for descriptor in COVERAGE_DESCRIPTORS:
        training_values = []
        for name in TRAINING_SUITE:
            training_values.append(profiles[name][descriptor])
        low = min(training_values)
        high = max(training_values)
        print("  {:<26s} training range {:10.3f} .. {:10.3f}".format(descriptor, low, high))
        for name in HELD_OUT_SUITE:
            value = profiles[name][descriptor]
            if value < low:
                position = "EXTRAPOLATING below"
            elif value > high:
                position = "EXTRAPOLATING above"
            else:
                position = "interpolating"
            print("    {:<24s} {:10.3f}   {}".format(name, value, position))


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


def suite_scores(test_rows, predictions):
    """The real objective: one design scored on all four workloads at once.

    The suite score is the geometric mean of per-workload IPC, so in log space
    it is the plain mean of the four per-workload log speedups. Only designs
    measured on all four workloads count. Returns (number of designs, spearman,
    best real IPC in the top 1 / 5 / 10 predicted, rank of the top pick).
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
    return len(suite_predictions), rank_correlation, best_in_top, top_rank


def suite_line(label, test_rows, predictions):
    count, rank_correlation, best_in_top, top_rank = suite_scores(test_rows, predictions)
    print("  {:26s} n={:4d}  spearman={:.3f}  best real IPC in top1/5/10 = "
          "{:.4f}/{:.4f}/{:.4f}  ({:.1f}%/{:.1f}%/{:.1f}% of reference)  rank of top pick={}".format(
              label, count, rank_correlation,
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


REPLAY_OBSERVED = [0, 4, 8, 12, 16, 24]
REPLAY_DRAWS = 5


def suite_design_names(test_rows):
    """Designs measured on every workload of the suite, in a fixed order."""
    counts = {}
    for row in test_rows:
        name = config_name(row["knobs"], row["soc"])
        counts[name] = counts.get(name, 0) + 1
    names = []
    for name in sorted(counts):
        if counts[name] == len(TRACES):
            names.append(name)
    return names


def replay():
    """Decision 21, offline: as observations on chip C accumulate, does a GP that
    learns MISS COUNTS (chip-blind, so every A/B row is usable as is) and hands
    them to the CPI stack rank the remaining designs better than today's knob GP
    on log speedup with a chip one-hot? k observed designs are drawn at random
    from the cached C designs; the rest are scored on the suite objective."""
    rows = load_rows()
    train_rows, test_rows = split_by_chip(rows)
    chips = chips_for(rows)
    coefficients = fit_coefficients(train_rows, BASE_COEFFICIENTS)
    names = suite_design_names(test_rows)
    rows_by_name = {}
    for row in test_rows:
        rows_by_name.setdefault(config_name(row["knobs"], row["soc"]), []).append(row)
    print("== replay on chip C: {} suite designs; {} draws per k; reference {:.4f}".format(
        len(names), REPLAY_DRAWS, FIXED_REFERENCE))
    print("  {:>4s}  {:>28s}  {:>28s}".format("k", "knob GP (today)", "miss-count GP + stack"))
    print("  {:>4s}  {:>13s} {:>14s}  {:>13s} {:>14s}".format("", "spearman", "top-5 %ref", "spearman", "top-5 %ref"))
    for observed_count in REPLAY_OBSERVED:
        scores = {"knob": [], "misses": []}
        for draw in range(REPLAY_DRAWS):
            shuffled = list(names)
            random.Random(draw).shuffle(shuffled)
            observed_names = set(shuffled[:observed_count])
            observed_rows = []
            remaining_rows = []
            for name in names:
                if name in observed_names:
                    observed_rows.extend(rows_by_name[name])
                else:
                    remaining_rows.extend(rows_by_name[name])
            fit_rows = train_rows + observed_rows
            # Today's pooled GP: knobs + chip one-hot, target log speedup.
            knob_predictions = surrogates.fit_and_predict(
                fit_rows, remaining_rows, chips, surrogates.FEATURIZERS["knob"], truths_of(fit_rows))
            scores["knob"].append(suite_scores(remaining_rows, knob_predictions))
            # Miss counts learned on every row (chip-blind), cost from the stack.
            learned_rows = rows_with_learned_misses(fit_rows, remaining_rows)
            stack = stack_predictions(learned_rows, chips, coefficients)
            scores["misses"].append(suite_scores(remaining_rows, stack))
        line = "  {:>4d}".format(observed_count)
        for model in ["knob", "misses"]:
            spearman_mean = 0.0
            top5_mean = 0.0
            for count, spearman, best_in_top, top_rank in scores[model]:
                spearman_mean += spearman / len(scores[model])
                top5_mean += 100.0 * best_in_top[1] / FIXED_REFERENCE / len(scores[model])
            line += "  {:>13.3f} {:>13.1f}%".format(spearman_mean, top5_mean)
        print(line, flush=True)
        if observed_count == 0:
            break_note = "  (k=0 is the offline table's setting: zero designs seen on C)"
            print(break_note)


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


def language(playbook_path):
    """Gate G-language, free: do the playbook's rules fire where the held-out class
    can use them and stay silent where it cannot? For every held-out workload,
    read the rules against chip C and the workload's descriptors, then compare
    each speaking rule's claimed direction with the one-knob effects already
    measured on that workload's cached designs. Written down before any test
    cell runs; the test cells then measure how much the rules cut the search."""
    from loop import forecast, playbook
    from loop.configs import SEARCH_SPACE
    store = playbook.load(playbook_path)
    baseline = champsim_problem.profiled_baseline("C_server", "C")
    print("== G-language: {} rules from {} read against chip C and the held-out workloads".format(
        len(store["rules"]), playbook_path))
    for trace in HELD_OUT_SUITE:
        with open("results/profile_{}.json".format(trace)) as profile_file:
            profile = json.load(profile_file)
        descriptors = champsim_problem.chip_descriptors(profile, baseline, SEARCH_SPACE, "C_server")
        speaking = forecast.speaking_rules(store["rules"], descriptors)
        table = champsim_problem.load_table("results/tierC_C_server_{}.json".format(trace))
        designs = []
        for entry in table.values():
            if entry["metrics"] is not None:
                designs.append({"knobs": entry["knobs"], "metrics": entry["metrics"]})
        effects = forecast.one_knob_effects(designs, "ipc")
        print("-- {} | movable_llc {:.1f} movable_l2 {:.1f} stride {:.2f} | {} of {} rules speak | {} cached designs".format(
            trace, descriptors["movable_llc_mpki"], descriptors["movable_l2_mpki"],
            descriptors["stride_regular_fraction"], len(speaking), len(store["rules"]), len(designs)))
        for rule in speaking:
            claim = rule["claim"]
            matches = []
            for effect in effects:
                if effect["knob"] != claim["knob"]:
                    continue
                is_direction = str(claim["value"]) in ["up", "down"]
                if is_direction or effect["to"] == str(claim["value"]) or effect["from"] == str(claim["value"]):
                    matches.append("{}->{} {:+.1f}%".format(effect["from"], effect["to"], effect["gain_pct"]))
            shown = "no controlled pair cached"
            if len(matches) > 0:
                shown = "; ".join(matches[:4])
            print("   {} {} {} claims {:+.1f}% | measured: {}".format(
                rule["id"], claim["knob"], claim["value"], claim["gain_pct"], shown))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "ceiling"
    if step == "language":
        language(sys.argv[2])
        raise SystemExit(0)
    if step == "replay":
        replay()
        raise SystemExit(0)
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
    elif step == "coverage":
        coverage()
    elif step == "headroom":
        headroom()
    else:
        raise SystemExit("unknown step: " + step)
