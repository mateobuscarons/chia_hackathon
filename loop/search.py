"""The loop: the searches over one chip, same budget, same seeds, same fidelity.

  python -m loop.search <cell> <tag> [arm,arm]           # run the arms -> results/runs/<cell>_<tag>.json
  python -m loop.search score results/runs/<cell>_<tag>.json
  python -m loop.search ceiling <designs> <batch> <trace> [trace ...]

Arms, each buying BUDGET designs from the stock chip (design D0):
  bo          random forest + expected improvement, one design per round, so every
              pick is made against measured outcomes
  pooled_bo   the identical search, started from the design the memory hands over
              (loop.memory). One variable differs from `bo`, and it is the memory.
  council     four specialists, one per concern, one concern moving per round,
              started from the same handover (loop.council)
  llm_alone   the agent, PER_ROUND designs a round, reading the chip, the knobs
              and every design measured so far (loop.analyst)

The question is designs-to-level, not level-at-N: how many simulations each arm
needs to get within some share of the best design known. Shares are read by
`score` against the best design measured on every workload of the suite, which is
why `ceiling` exists - it is the same forest search run long, and it must stay a
mechanism separate from the arms. Scoring against a design an arm found bounds the
metric at that arm, which has happened once and cost every arm five points.

Env: SEEDS (default 2), FIRST_SEED, BUDGET (default 16), PARALLEL_RUNS, SIM_THREADS,
MEMORY_PATH (default results/memory.json), ANALYST_MODEL.
"""

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy
from sklearn.ensemble import RandomForestRegressor

from loop import analyst, memory
from loop.suite import make_suite_problem, measured_designs, short_name
from loop.space import SEARCH_SPACE, random_feasible_designs, typed_knobs

TRACE = {"mcf": "traces/605.mcf_s-665B.champsimtrace.xz",
         "lbm": "traces/619.lbm_s-2676B.champsimtrace.xz",
         "omnetpp": "traces/620.omnetpp_s-874B.champsimtrace.xz",
         "bfs.urand": "traces/bfs.urand-36B.champsimtrace.xz",
         "pr.urand": "traces/pr.urand-129B.champsimtrace.xz",
         "bfs.kron": "traces/bfs.kron-128B.champsimtrace.xz",
         "sierra.a.4": "traces/sierra.a.4_0000.champsim.gz",
         "merced": "traces/merced_0000.champsim.gz",
         "tahoe": "traces/tahoe_0000.champsim.gz",
         "whiskey": "traces/whiskey_0000.champsim.gz",
         "bravo": "traces/bravo.a_0000.champsim.gz",
         "delta": "traces/delta_0000.champsim.gz"}

# The suite each cell tests. None of them is in the memory: the memory remembers
# mcf, omnetpp, lbm, bfs.urand, pr.urand and bfs.kron, and nothing else.
CELLS = {
    # The headline: Google datacenter traces, never seen.
    "dc": [TRACE["sierra.a.4"], TRACE["merced"], TRACE["tahoe"]],
    # The confirmation: the three datacenter traces the first suite did not take.
    "dc2": [TRACE["whiskey"], TRACE["bravo"], TRACE["delta"]],
    # The gate before any launch: every arm, one round, two workloads.
    "smoke": [TRACE["mcf"], TRACE["lbm"]],
}
ARMS = ["bo", "pooled_bo", "council", "llm_alone"]

BUDGET = int(os.environ.get("BUDGET", "16"))
PER_ROUND = 2             # the LLM's designs per round; the forest buys one at a time
WARM_UP_DESIGNS = 3       # taken from the pool before the forest chooses anything
TREES = 100
# The forest scores a fixed random sample of the feasible designs (the space holds
# 6.6 million); the agent may name ANY feasible design.
CANDIDATE_POOL = 20000
PARALLEL_RUNS = int(os.environ.get("PARALLEL_RUNS", "6"))


def memory_path():
    return os.environ.get("MEMORY_PATH", "results/memory.json")


REPORT_DIR = "results/runs"


def report_path(cell_name, tag):
    os.makedirs(REPORT_DIR, exist_ok=True)
    return os.path.join(REPORT_DIR, "{}_{}.json".format(cell_name, tag))


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


def fit_forest(knobs_list, values):
    forest = RandomForestRegressor(n_estimators=TREES, min_samples_leaf=1, max_features=0.8,
                                   bootstrap=True, random_state=0, n_jobs=-1)
    forest.fit(encode_many(knobs_list), numpy.array(values))
    return forest


def predict(forest, knobs_list):
    """Mean and spread over the trees: the forest's predictive distribution, which
    is what makes expected improvement possible without a Gaussian process."""
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


def choose(candidates, measured_knobs, measured_values, best_so_far, wanted):
    """`wanted` designs by expected improvement. Within one batch a pick joins the
    training set at its own predicted value, so the next pick of the same batch
    does not simply repeat it."""
    believed_knobs = list(measured_knobs)
    believed_values = list(measured_values)
    names = []
    knobs_list = []
    for name in candidates:
        names.append(name)
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
        print("[{}] batch failed ({}), retrying one at a time".format(tag, repr(error)[-160:]), flush=True)
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
            print("[{}] dropped {}: {}".format(tag, name, repr(error)[-160:]), flush=True)
    return kept, metrics, dropped


# ---------------------------------------------------------------- the forest arms ----

def forest_search(problem, budget, batch, seed, tag, start_design=None, warm_up=WARM_UP_DESIGNS):
    """`budget` designs bought in batches of `batch`. Both forest arms and the
    ceiling go through here, so the settings they disagree on are arguments rather
    than two copies of one loop - comparing curves grown under different warm-ups
    is how a wrong ratio gets published.

    start_design: bought before anything else. `pooled_bo` passes the design the
    memory hands over, so the only difference from `bo` is where the search begins.

    The candidate pool keeps the whole space in view, not only the incumbent's
    neighbours: the best design measured sits SIX knobs from the memory's handover
    and everything within five of it is capped well below the ceiling, so a search
    that only walks one knob at a time cannot get there from a good start."""
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

    def buy(knobs_list, source, round_number):
        """Measure a batch and record it; a design the simulator cannot measure is
        dropped and never offered again, and the loop picks a replacement."""
        nonlocal best, best_knobs
        knobs_list, metrics_list, dropped = measure_batch(problem, knobs_list, tag)
        for name in dropped:
            broken.add(name)
            candidates.pop(name, None)
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


# ---------------------------------------------------------------- the LLM arm ----

def llm_search(problem, budget, per_round, tag):
    """The agent buys `per_round` designs a round. The designs of one round are
    chosen together from the same information, so they are one decision, not two.

    Like the forest arm, the budget counts designs that produced a measurement, so
    a crashed design buys another round. Two guards bound that: a round that can
    name no new design at all ends the run, and no run takes more rounds than its
    budget in designs."""
    objective = problem["objective"]
    started = time.time()
    stock_metrics = problem["evaluate"](problem["stock"])
    history = [{"index": 0, "round": 0, "name": problem["name_of"](problem["stock"]), "knobs": problem["stock"],
                "metrics": stock_metrics, "source": "stock", "hypothesis": None}]
    taken = {history[0]["name"]}
    best = stock_metrics[objective]
    round_logs = []
    print("[{}] {} designs, {} a round | {} | stock {:.4f}".format(
        tag, budget, per_round, "+".join(problem["workloads"]), best), flush=True)
    round_number = 0
    while len(history) - 1 < budget and round_number < budget:
        round_number += 1
        wanted = min(per_round, budget - (len(history) - 1))
        chosen, log = analyst.designs_for_round(problem, history, wanted, taken)
        if len(chosen) == 0:
            print("[{}] round {} could name no new design, stopping".format(tag, round_number), flush=True)
            break
        knobs_list = []
        for design in chosen:
            knobs_list.append(design["knobs"])
        knobs_list, metrics_list, dropped = measure_batch(problem, knobs_list, tag)
        by_name = {}
        for design in chosen:
            by_name[design["name"]] = design
        for knobs, metrics in zip(knobs_list, metrics_list):
            entry = dict(by_name[problem["name_of"](knobs)])
            entry["index"] = len(history)
            entry["round"] = round_number
            entry["metrics"] = metrics
            history.append(entry)
            if metrics[objective] > best:
                best = metrics[objective]
            print("[{}] round {} | D{} | {}={:.4f} | best {:.4f} | {} | {:.0f} min".format(
                tag, round_number, entry["index"], objective, metrics[objective], best, entry["source"],
                (time.time() - started) / 60), flush=True)
        log["round"] = round_number
        log["dropped"] = dropped
        if round_number > 1:
            # Every later prompt is round 1's plus the rows now in `designs`, so
            # storing it again would be the same four sections forty times a cell.
            del log["prompt"]
        round_logs.append(log)
    if len(history) - 1 < budget:
        print("[{}] finished short: {} of {} designs measured".format(tag, len(history) - 1, budget), flush=True)
    return {"designs": history, "rounds": round_logs}


# ---------------------------------------------------------------- one run ----

def run_one(arm, cell_name, seed, budget, tag_prefix):
    """One (arm, seed) run; returns only what the report keeps."""
    problem = make_suite_problem(CELLS[cell_name])
    tag = "{}-{}-s{}".format(arm, tag_prefix, seed)
    if arm == "bo":
        result = forest_search(problem, budget, 1, seed, tag)
    elif arm == "pooled_bo":
        result = forest_search(problem, budget, 1, seed, tag,
                               start_design=memory.handover(memory_path()))
    elif arm == "llm_alone":
        result = llm_search(problem, budget, PER_ROUND, tag)
    elif arm == "council":
        from loop import council
        result = council.search(problem, budget, seed, tag,
                                start_design=memory.handover(memory_path()))
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
                     "ipc": entry["metrics"]["ipc"], "per_workload": per_workload})
    return rows


# ---------------------------------------------------------------- the cell ----

def run_cell(cell_name, tag, arms=None):
    if arms is None:
        arms = ARMS
    seeds = int(os.environ.get("SEEDS", "2"))
    first_seed = int(os.environ.get("FIRST_SEED", "0"))
    budget = BUDGET
    if cell_name == "smoke":
        budget = PER_ROUND
    workloads = []
    for trace_path in CELLS[cell_name]:
        workloads.append(short_name(trace_path))
    output_path = report_path(cell_name, tag)
    report = {"cell": cell_name, "tag": tag, "workloads": workloads, "arms": arms, "seeds": seeds,
              "first_seed": first_seed, "budget": budget, "per_round": PER_ROUND,
              "memory_path": memory_path(), "model": analyst.MODEL, "runs": {}, "failed": []}
    for arm in arms:
        report["runs"][arm] = {}
    print("cell {} ({}) -> {} | workloads {} | arms {} | seeds {}..{} | {} designs per run".format(
        cell_name, tag, output_path, workloads, arms, first_seed, first_seed + seeds - 1, budget), flush=True)
    started = time.time()
    pool = ProcessPoolExecutor(max_workers=PARALLEL_RUNS)
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
        best = max(row["ipc"] for row in result["designs"])
        print("== {} seed {}: best {:.4f} over {} designs".format(arm, seed, best, len(result["designs"]) - 1), flush=True)
        write_report(report, output_path)
    report["wall_seconds"] = time.time() - started
    write_report(report, output_path)
    print("done in {:.1f} h, {} failed".format(report["wall_seconds"] / 3600.0, len(report["failed"])), flush=True)
    return output_path


def write_report(report, output_path):
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=1)


# ---------------------------------------------------------------- the score ----

def score(report_path_):
    """Per arm, the share of the stock-to-best-known gap reached after 1..BUDGET
    designs, mean over seeds with the spread below, and the mean best design. The
    best known design is the best one measured on every workload of the suite in
    the cached tables, so it moves as searches land."""
    with open(report_path_) as report_file:
        report = json.load(report_file)
    problem = make_suite_problem(CELLS[report["cell"]], allow_simulation=False)
    known = measured_designs(problem)
    stock = None
    best_known = None
    for design in known:
        if design["name"] == problem["name_of"](problem["stock"]):
            stock = design["metrics"]["ipc"]
        if best_known is None or design["metrics"]["ipc"] > best_known:
            best_known = design["metrics"]["ipc"]
    budget = report["budget"]
    print("cell {} ({}): {} | stock {:.4f}, best known {:.4f} (+{:.1f}%) over {} designs measured on every workload".format(
        report["cell"], report["tag"], " + ".join(report["workloads"]), stock, best_known,
        100.0 * (best_known / stock - 1.0), len(known)))
    print()
    header = "{:<14s} {:>5s}".format("arm", "seeds")
    for index in range(1, budget + 1):
        header += " {:>7s}".format("D{}".format(index))
    header += " {:>9s}".format("final")
    print("share of the stock-to-best-known gap reached after N designs (mean over seeds; min..max below)")
    print(header)
    for arm in report["arms"]:
        runs = report["runs"].get(arm, {})
        if len(runs) == 0:
            continue
        shares_by_index = []
        for index in range(budget + 1):
            shares_by_index.append([])
        finals = []
        for seed in sorted(runs):
            curve = best_so_far(runs[seed]["designs"])
            for index in range(1, budget + 1):
                value = curve[min(index, len(curve) - 1)]
                shares_by_index[index].append(100.0 * (value - stock) / (best_known - stock))
            finals.append(curve[-1])
        mean_line = "{:<14s} {:>5d}".format(arm, len(runs))
        spread_line = "{:<14s} {:>5s}".format("", "")
        for index in range(1, budget + 1):
            values = shares_by_index[index]
            mean_line += " {:>6.0f}%".format(sum(values) / len(values))
            spread_line += " {:>3.0f}..{:<3.0f}".format(min(values), max(values))
        mean_line += " {:>9.4f}".format(sum(finals) / len(finals))
        print(mean_line)
        if len(runs) > 1:
            print(spread_line)
    if len(report.get("failed", [])) > 0:
        print()
        print("failed runs:", json.dumps(report["failed"]))


def best_so_far(designs):
    curve = []
    best = None
    for row in designs:
        if best is None or row["ipc"] > best:
            best = row["ipc"]
        curve.append(best)
    return curve


# ---------------------------------------------------------------- the ceiling ----

def ceiling(traces, how_many, batch):
    """The reference a cell is scored against: the same forest search, run long,
    from the stock chip and reading no memory. At 100 designs nothing was measured
    about the warm-up, so it takes the conventional ten percent."""
    problem = make_suite_problem(traces)
    objective = problem["objective"]
    result = forest_search(problem, how_many, batch, 0, "ceiling", warm_up=max(1, int(how_many * 0.10)))
    stock = result["designs"][0]["metrics"][objective]
    best = stock
    best_knobs = result["designs"][0]["knobs"]
    for entry in result["designs"]:
        if entry["metrics"][objective] > best:
            best = entry["metrics"][objective]
            best_knobs = entry["knobs"]
    print("ceiling: best {:.4f} (+{:.1f}% on the stock chip) over {} designs".format(
        best, 100.0 * (best / stock - 1.0), how_many), flush=True)
    print("ceiling: best design " + problem["name_of"](best_knobs), flush=True)
    return best


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2])
    elif sys.argv[1] == "ceiling":
        ceiling(sys.argv[4:], int(sys.argv[2]), int(sys.argv[3]))
    else:
        chosen_arms = None
        if len(sys.argv) > 3:
            chosen_arms = sys.argv[3].split(",")
        run_cell(sys.argv[1], sys.argv[2], chosen_arms)
