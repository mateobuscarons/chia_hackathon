"""The loop: the searches over one chip, same budget, same seeds, same fidelity.

  python -m council_loop.search <cell> <tag> [arm,arm]     # run the arms -> results/runs/<soc>_<cell>_<tag>.json
  python -m council_loop.search preview <cell>             # every prompt of an opening round, no model call
  python -m council_loop.search gate <cell> [samples] [seed]   # the brief gate: is there a search problem here

Arms, each measuring BUDGET designs from the stock chip (design D0):
  random      uniform draws from the feasible set, BO_BATCH a round
  council     the council (council_loop.council): the statistician, the analyst, four specialists,
              one wave of WAVE designs a round after OPENINGS opening designs
Every run is scored from its ledger (council_loop.ledgers, the CHIA ledger block).

Env: SEEDS (default 2), FIRST_SEED, BUDGET (default 60), BO_BATCH (designs per random round,
default 1), PARALLEL_RUNS, SIM_THREADS, ANALYST_MODEL, LOOP_WARMUP / LOOP_SIM (the run's
fidelity), WAVE / OPENINGS / PLATEAU (the council's).
"""

import json
import math
import multiprocessing
import os
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy
import ray
from chia.trace.profiler import get_collector, reset_profiler, start_collector, stop_collector
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from council_loop import analyst
from council_loop import chip
from council_loop import socs
from council_loop.simulate import tree_paths
from council_loop.suite import (CHAMPSIM_ROOT, SIMULATION_INSTRUCTIONS, WARMUP_INSTRUCTIONS, make_suite_problem,
                               workload_of, workloads_for)
from council_loop.space import SEARCH_SPACE, knobs_changed, random_feasible_designs, violations

TRACE = {"mcf": "traces/605.mcf_s-665B.champsimtrace.xz",
         "lbm": "traces/619.lbm_s-2676B.champsimtrace.xz",
         "llama2": "traces/llama2.c-llama2_7b.1.champsimtrace.gz",
         "sd": "traces/stable-diffusion.cpp-v1-5-pruned-emaonly.1.champsimtrace.gz",
         "clip": "traces/clip_trace_1.champsimtrace.gz",
         "whisper": "traces/whisper_trace_1.champsimtrace.gz",
         "vit": "traces/vit.cpp-large-ggml-model-f16.gguf.armadillo.1.champsimtrace.gz",
         "biogpt": "traces/biogpt.cpp-ggml-model-tocilizumab.1.champsimtrace.gz"}

# The workload a cell measures a design on. `llama2` is the paper's trace; the other five ML
# traces are the ones the brief gate screened; `smoke` is the launch check on two cheap traces.
CELLS = {
    "llama2": [TRACE["llama2"]],
    "sd": [TRACE["sd"]],
    "clip": [TRACE["clip"]],
    "whisper": [TRACE["whisper"]],
    "vit": [TRACE["vit"]],
    "biogpt": [TRACE["biogpt"]],
    "smoke": [TRACE["mcf"], TRACE["lbm"]],
}
ARMS = ["random", "council"]

BUDGET = int(os.environ.get("BUDGET", "60"))
SMOKE_BUDGET = 2          # the gate measures this many designs per arm, whatever BUDGET says
# The forest scores a fixed random sample of the feasible designs (the space holds
# 6.6 million); the agent may name ANY feasible design.
CANDIDATE_POOL = 20000
PARALLEL_RUNS = int(os.environ.get("PARALLEL_RUNS", "6"))

REPORT_DIR = "results/runs"
PROFILE_DIR = "results/profiles"    # one profiler log per run: results/profiles/<tag>/ChiaProfileCollector.log


def start_ray(runs):
    """One local Ray for this process. The resources are the ones CHIA's nodes ask for: a
    `champsim` unit per simulation of `runs` parallel runs plus one build per tree, and the
    credentials unit of the model node. A run process connects to it under its own name."""
    simulations = int(os.environ.get("SIM_THREADS", "4")) * runs
    trees = len(tree_paths(CHAMPSIM_ROOT))
    ray.init(resources={"champsim": simulations + trees, "vertex_creds": 1}, include_dashboard=False, log_to_driver=False)


def report_path(cell_name, tag):
    """One report per (SoC, cell, tag): the name says which chip it was measured on, so a
    report is never read for another chip's."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    return os.path.join(REPORT_DIR, "{}_{}_{}.json".format(socs.NAME, cell_name, tag))


# ---------------------------------------------------------------- the surrogate ----

def encode(knobs):
    """A size or a width becomes one column holding log2 of its value; a prefetcher
    or a policy becomes one column per value, holding 1 for the value it has.

    Trees can split on a value's index directly, so the one-hot widening looks
    unnecessary - but it was measured, and indexing the values is worse (D16 88%
    against 92%), because an index imposes an order on policies that have none and
    a single split then has to separate values that are not adjacent."""
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


def fit_gp(knobs_list, values, kernel=None):
    """A Gaussian process on the same encoding: Matern 5/2 with one length scale per
    column and a noise term, the surrogate the standard BO libraries fit. The kernel's
    hyperparameters are fitted once a batch (`kernel` None); the picks within the batch
    reuse them and refit the posterior only."""
    if kernel is None:
        columns = len(encode(knobs_list[0]))
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=numpy.ones(columns), length_scale_bounds=(1e-2, 1e2), nu=2.5)
                  + WhiteKernel(1e-3, (1e-6, 1e-1)))
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1, random_state=0)
    else:
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)
    with warnings.catch_warnings():
        # A length scale at its bound is the fit saying a column does not matter; not a fault.
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(encode_many(knobs_list), numpy.array(values))
    return model


def predict(model, knobs_list):
    """Mean and spread of the Gaussian process's posterior at these designs."""
    rows = encode_many(knobs_list)
    mean, spread = model.predict(rows, return_std=True)
    return mean, numpy.maximum(spread, 1e-6)


def expected_improvement(means, spreads, best_so_far):
    scores = []
    for mean, spread in zip(means, spreads):
        z = (mean - best_so_far) / spread
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        scores.append((mean - best_so_far) * cdf + spread * pdf)
    return numpy.array(scores)


# ---------------------------------------------------------------- the candidates ----

def candidate_pool(problem, seed):
    """The sample of feasible designs the forest scores, drawn with the run's own
    seed so every seed scores a different sample."""
    pool = {}
    for knobs in random_feasible_designs(CANDIDATE_POOL, seed=seed):
        if problem["is_candidate"](knobs):
            pool[problem["name_of"](knobs)] = knobs
    return pool


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


# ---------------------------------------------------------------- measuring ----

def measure_batch(problem, knobs_list, tag):
    """Measure a batch and return what survived: (knobs kept, metrics, names dropped).

    A design that makes the simulator produce no usable stats used to end a whole
    run, and because the candidate pool is drawn from the seed the same bad design
    came back every time. Retry design by design instead and drop whatever still
    fails: the budget counts designs that produced a measurement."""
    try:
        return knobs_list, problem["evaluate_many"](knobs_list), []
    except Exception as error:
        print("[{}] batch failed ({}), retrying one at a time".format(tag, str(error)[-300:]), flush=True)
    kept = []
    metrics = []
    dropped = []
    for knobs in knobs_list:
        name = problem["name_of"](knobs)
        try:
            metrics.append(problem["evaluate"](knobs))
            kept.append(knobs)
        except Exception as error:
            dropped.append(name)
            print("[{}] dropped {}: {}".format(tag, name, str(error)[-300:]), flush=True)
    return kept, metrics, dropped


# ---------------------------------------------------------------- the forest arms ----



def random_search(problem, budget, batch, seed, tag):
    """Uniform draws from the feasible set, `batch` a round, no model and no memory: the
    lowest rung of the ladder every other arm is read against."""
    started = time.time()
    stock_metrics = problem["evaluate"](problem["stock"])
    history = [{"index": 0, "round": 0, "name": problem["name_of"](problem["stock"]), "knobs": problem["stock"],
                "metrics": stock_metrics, "source": "stock", "hypothesis": None, "violations": []}]
    # Drawn once, in order, so a seed is the same sequence whatever the batch.
    pool = random_feasible_designs(budget + 20, seed=seed)
    stock_name = problem["name_of"](problem["stock"])
    round_number = 0
    while len(history) - 1 < budget:
        round_number += 1
        knobs_list = []
        while len(knobs_list) < min(batch, budget - (len(history) - 1)) and pool:
            knobs = pool.pop(0)
            if problem["name_of"](knobs) != stock_name:
                knobs_list.append(knobs)
        knobs_list, metrics_list, dropped = measure_batch(problem, knobs_list, tag)
        for knobs, metrics in zip(knobs_list, metrics_list):
            history.append({"index": len(history), "round": round_number, "name": problem["name_of"](knobs),
                            "knobs": knobs, "metrics": metrics, "source": "random", "hypothesis": None,
                            "violations": violations(metrics, problem["workloads"], stock_metrics)})
        print("[{}] round {} | {} designs | best {:.4f} | {:.0f} min".format(
            tag, round_number, len(history) - 1, best_so_far(history)[-1], (time.time() - started) / 60), flush=True)
    return {"designs": history, "rounds": []}


# ---------------------------------------------------------------- one run ----

def run_one(arm, cell_name, seed, budget, tag_prefix):
    """One (arm, seed) run in its own process, connected to the cell's Ray under its own name and
    with its own profiler collector: one log per run holding the model calls, the candidates and
    the gate events. Returns only what the report keeps."""
    tag = "{}-{}-{}-s{}".format(socs.NAME, arm, tag_prefix, seed)
    ray.init(address="auto", namespace=tag, log_to_driver=False)
    log_dir = os.path.join(PROFILE_DIR, tag)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)          # a re-run under the same tag starts a fresh log, as its ledger does
    start_collector(log_dir=log_dir, namespace=tag)
    reset_profiler()            # a profiler made before the collector existed would stay disabled
    try:
        return run_arm(arm, cell_name, seed, budget, tag)
    finally:
        # Events are fire-and-forget: a blocking call on the collector lands every one sent
        # before the actor stops.
        ray.get(get_collector().get_events.remote())
        stop_collector()
        ray.shutdown()


def run_arm(arm, cell_name, seed, budget, tag):
    problem = make_suite_problem(CELLS[cell_name])
    if arm == "council":
        from council_loop import council
        result = council.search(problem, budget, seed, tag)
    elif arm == "random":
        result = random_search(problem, budget, int(os.environ.get("BO_BATCH", "1")), seed, tag)
    else:
        raise ValueError("unknown arm " + arm)
    result["designs"] = compact(result["designs"], problem)
    return result


def compact(history, problem):
    rows = []
    for entry in history:
        per_workload = {}
        for workload in problem["workloads"]:
            per_workload[workload] = {"ipc": entry["metrics"][workload + ":ipc"],
                                      "LLC_mpki": entry["metrics"][workload + ":LLC_mpki"]}
        rows.append({"index": entry["index"], "round": entry["round"], "name": entry["name"], "knobs": entry["knobs"],
                     "source": entry["source"], "hypothesis": entry.get("hypothesis"),
                     "ipc": entry["metrics"]["ipc"], "mm2": entry["metrics"].get("mm2"),
                     "watts": entry["metrics"].get("watts"), "violations": entry.get("violations", []),
                     "per_workload": per_workload})
    return rows


# ---------------------------------------------------------------- the cell ----

def council_flags():
    """The mechanisms the council is running with, recorded in every report."""
    from council_loop import council          # imported here: council imports this module
    return {"wave": council.WAVE, "openings": council.OPENINGS, "plateau": council.PLATEAU,
            "chip_view": council.CHIP_VIEW,
            # The caps in force for this run. They can be moved after a chip is under
            # design, so a report that does not say which ones it ran under cannot be
            # read against another.
            "area_cap_mm2": socs.AREA_BUDGET_MM2, "power_cap_w": socs.POWER_BUDGET_W,
            "rounds": council.MAX_ROUNDS}


def run_cell(cell_name, tag, arms=None):
    if arms is None:
        arms = ARMS
    seeds = int(os.environ.get("SEEDS", "2"))
    first_seed = int(os.environ.get("FIRST_SEED", "0"))
    budget = BUDGET
    if cell_name == "smoke":
        budget = SMOKE_BUDGET
    workloads = []
    for entry in workloads_for(CELLS[cell_name]):
        workloads.append(workload_of(entry)[0])
    output_path = report_path(cell_name, tag)
    report = {"cell": cell_name, "tag": tag, "workloads": workloads, "arms": arms, "seeds": seeds,
              "first_seed": first_seed, "budget": budget,
              "bo_batch": int(os.environ.get("BO_BATCH", "1")),
              "fidelity": [WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS],
              "model": analyst.MODEL,
              "soc": socs.NAME, "chip": chip.NAME,
              "area_budget_kb": socs.AREA_BUDGET_KB, "card": chip.card(),
              "flags": council_flags(),
              "runs": {}, "failed": []}
    for arm in arms:
        report["runs"][arm] = {}
    print("cell {} ({}) -> {} | workloads {} | arms {} | seeds {}..{} | {} designs per run".format(
        cell_name, tag, output_path, workloads, arms, first_seed, first_seed + seeds - 1, budget), flush=True)
    started = time.time()
    start_ray(PARALLEL_RUNS)
    # Spawned, not forked: a forked child of a Ray driver inherits a connection it cannot use.
    pool = ProcessPoolExecutor(max_workers=PARALLEL_RUNS, mp_context=multiprocessing.get_context("spawn"))
    handles = []
    # Seed-major: the first wave already covers every arm on seed 0.
    for seed in range(first_seed, first_seed + seeds):
        for arm in arms:
            handles.append((arm, seed, pool.submit(run_one, arm, cell_name, seed, budget, tag)))
    for arm, seed, handle in handles:
        try:
            result = handle.result()
        except Exception as error:
            report["failed"].append({"arm": arm, "seed": seed, "error": repr(error)[-400:]})
            print("!! FAILED {} seed {}: {}".format(arm, seed, repr(error)[-200:]), flush=True)
            continue
        report["runs"][arm][str(seed)] = result
        best = best_so_far(result["designs"])[-1]
        print("== {} seed {}: best {:.4f} over {} designs".format(arm, seed, best, len(result["designs"]) - 1), flush=True)
        write_report(report, output_path)
    report["wall_seconds"] = time.time() - started
    write_report(report, output_path)
    ray.shutdown()
    print("done in {:.1f} h, {} failed".format(report["wall_seconds"] / 3600.0, len(report["failed"])), flush=True)
    return output_path


def write_report(report, output_path):
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=1)


# ---------------------------------------------------------------- the score ----

ROUND_LADDER = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 75]
LEVELS = [90.0, 95.0, 96.0, 99.0, 99.9]


def best_so_far(designs):
    curve = []
    best = None
    for row in designs:
        feasible = not row.get("violations")
        value = row["ipc"] if "ipc" in row else row["metrics"]["ipc"]
        if feasible and (best is None or value > best):
            best = value
        curve.append(best)
    return curve


# ---------------------------------------------------------------- the ceiling ----

def gate(cell_name, samples=48, seed=0):
    """Is there a search problem here at all? Four questions, answered before any model
    call: how much room there is above the stock (headroom), how many knobs separate the
    best design from it (depth), how much of the gain the single best move buys (greedy
    resistance), and how many knobs in the winner LOSE when moved alone (deception). A
    space with headroom but no deception is solved by a sweep and needs no reasoning."""
    problem = make_suite_problem(CELLS[cell_name])
    objective = problem["objective"]
    stock = problem["stock"]
    designs = [stock]
    for knob in SEARCH_SPACE:
        for value in SEARCH_SPACE[knob]:
            if str(value) == str(stock[knob]):
                continue
            move = dict(stock)
            move[knob] = value
            if problem["is_candidate"](move):
                designs.append(move)
    single_moves = len(designs) - 1
    designs += random_feasible_designs(samples, seed=seed)
    print("gate {} (seed {}): {} designs = stock + {} feasible single moves + {} random".format(
        cell_name, seed, len(designs), single_moves, samples), flush=True)
    results = problem["evaluate_many"](designs)
    stock_score = results[0][objective]
    # A cap on power or on a workload's own speed is checked after the simulation, so a
    # design can be the fastest thing measured and still be one this chip may not ship.
    # The gate reports the best FEASIBLE design, the way the search stands on one.
    legal = []
    for index in range(len(designs)):
        broken = violations(results[index], problem["workloads"], results[0])
        legal.append(not broken)
    refused = len(designs) - sum(legal)
    alone = {}
    for index in range(1, single_moves + 1):
        changed = knobs_changed(designs[index], stock)
        knob = list(changed)[0]
        alone[(knob, str(designs[index][knob]))] = results[index][objective]
    feasible = [i for i in range(len(designs)) if legal[i]] or [0]
    best_index = max(feasible, key=lambda i: results[i][objective])
    best, best_score = designs[best_index], results[best_index][objective]
    changed = knobs_changed(best, stock)
    gain = best_score - stock_score
    best_single = max([results[i][objective] for i in range(1, single_moves + 1) if legal[i]]
                      + [stock_score])
    losers = [knob for knob in changed
              if alone.get((knob, str(best[knob])), stock_score) < stock_score]
    print("  headroom          {:+.1f}%   (stock {:.4f} -> best {:.4f})".format(
        100.0 * gain / stock_score, stock_score, best_score), flush=True)
    print("  depth             {} of {} knobs differ from the stock".format(len(changed), len(SEARCH_SPACE)), flush=True)
    print("  greedy resistance the best single move buys {:.0f}% of the gain".format(
        100.0 * (best_single - stock_score) / gain if gain > 0 else 0.0), flush=True)
    print("  deception         {} of {} changed knobs LOSE when moved alone".format(
        len(losers), len(changed)), flush=True)
    print("  refused           {} of {} designs broke a cap after measurement".format(
        refused, len(designs)), flush=True)
    print("  best design       " + problem["name_of"](best), flush=True)
    return {"stock": stock_score, "best": best_score, "changed": changed, "losers": losers}


if __name__ == "__main__":
    if sys.argv[1] == "preview":
        # Every prompt of an opening round on this SoC and cell, without a model call.
        from council_loop import council
        council.preview(make_suite_problem(CELLS[sys.argv[2]]))
    elif sys.argv[1] == "gate":
        start_ray(1)
        gate(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 48,
             int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    else:
        chosen_arms = None
        if len(sys.argv) > 3:
            chosen_arms = sys.argv[3].split(",")
        run_cell(sys.argv[1], sys.argv[2], chosen_arms)
