"""The experiment, one command per cell.

  python -m loop.run memory <cell>                 # the cell's memory from the cached tables of its memory workloads
  python -m loop.run <cell> <tag> [arm,arm,...]    # run the cell's arms; report -> results/run_<cell>_<tag>.json
  LOOP_DISPATCH=chia python -m loop.run <cell> <tag> [arms]   # the same as a CHIA loop: builds and simulations are CHIA
      tasks, runs are Ray tasks, the LLM goes through chia.models.vertex, and the profiler records the task graph
      (results/chia_profiles/, `chia viz-profile`). CHIA_ADDRESS=auto attaches to a cluster started by `chia up`.

Every arm starts from the stock chip (design D0) and buys BUDGET designs in rounds
of PER_ROUND, on the same seeds and at the same fidelity. Arms:
  bo          Gaussian process + expected improvement (loop.bo)
  llm_direct  the plain LLM agent (loop.agent)
  memory      the agent reading the memory's digest, proposing 8 designs a round, a GP fit on the
              run picking the 2 to simulate
Env: SEEDS (default 2), FIRST_SEED, BUDGET (default 8), PARALLEL_RUNS, SIM_THREADS,
ANALYST_MODEL (default gemini-2.5-flash), MEMORY_PATH (default the cell's).
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from loop import agent, bo, memory
from loop.champsim_problem import make_suite_problem, trace_short_name


TRACE = {"mcf": "traces/605.mcf_s-665B.champsimtrace.xz",
         "lbm": "traces/619.lbm_s-2676B.champsimtrace.xz",
         "omnetpp": "traces/620.omnetpp_s-874B.champsimtrace.xz",
         "bfs.urand": "traces/bfs.urand-36B.champsimtrace.xz",
         "pr.urand": "traces/pr.urand-129B.champsimtrace.xz",
         "bfs.kron": "traces/bfs.kron-128B.champsimtrace.xz",
         "sssp.kron": "traces/sssp.kron-246B.champsimtrace.xz",
         "cc.urand": "traces/cc.urand-353B.champsimtrace.xz",
         "cc.twitter": "traces/cc.twitter-15B.champsimtrace.xz",
         "sierra.a.4": "traces/sierra.a.4_0000.champsim.gz",
         "merced": "traces/merced_0000.champsim.gz",
         "tahoe": "traces/tahoe_0000.champsim.gz",
         "whiskey": "traces/whiskey_0000.champsim.gz",
         "bravo": "traces/bravo.a_0000.champsim.gz",
         "delta": "traces/delta_0000.champsim.gz"}

SPEC = [TRACE["mcf"], TRACE["omnetpp"], TRACE["lbm"]]
GAP_SET_1 = [TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"]]
GAP_SET_2 = [TRACE["sssp.kron"], TRACE["cc.urand"], TRACE["cc.twitter"]]
DATACENTER = [TRACE["sierra.a.4"], TRACE["merced"], TRACE["tahoe"]]
# The three admitted datacenter traces the first suite did NOT take: nothing about
# them influenced the memory, the pooled design or the choice of the first suite.
DATACENTER_HELD_OUT = [TRACE["whiskey"], TRACE["bravo"], TRACE["delta"]]

# The memory every cell reads: the workloads the chip has been searched on in
# depth (hundreds of designs each). A test workload is never in its own memory.
MEMORY_WORKLOADS = SPEC + GAP_SET_1

CELLS = {
    # The headline: SPEC and graph searches remembered, Google datacenter traces tested.
    "dc": {"test": DATACENTER, "memory": MEMORY_WORKLOADS},
    # The confirmation: the same memory, the datacenter traces the first suite did not take.
    "dc2": {"test": DATACENTER_HELD_OUT, "memory": MEMORY_WORKLOADS},
    # The fallback and development cell: the same memory, graph set 2 tested.
    "gap2": {"test": GAP_SET_2, "memory": MEMORY_WORKLOADS},
    # The gate before any launch: every arm, one round, two workloads.
    "smoke": {"test": [TRACE["mcf"], TRACE["lbm"]], "memory": [TRACE["omnetpp"], TRACE["bfs.urand"]]},
}
ARMS = ["bo", "llm_direct", "memory"]
# The LLM arms' switches: (use_memory, use_gp).
AGENT_SWITCHES = {"llm_direct": (False, False), "memory": (True, True)}
BUDGET = 8
PER_ROUND = 2
PARALLEL_RUNS = int(os.environ.get("PARALLEL_RUNS", "6"))


def memory_path(cell_name):
    return os.environ.get("MEMORY_PATH", "results/memory_{}.json".format(cell_name))


def report_path(cell_name, tag):
    return "results/run_{}_{}.json".format(cell_name, tag)


def build_memory(cell_name):
    return memory.build(CELLS[cell_name]["memory"], memory_path(cell_name))


# ---------------------------------------------------------------- one run ----

def run_one(arm, cell_name, seed, rounds, tag_prefix):
    """One (arm, seed) run; returns only what the report keeps."""
    problem = make_suite_problem(CELLS[cell_name]["test"])
    tag = "{}-{}-s{}".format(arm, tag_prefix, seed)
    memory_file = memory_path(cell_name)
    if arm == "bo":
        result = bo.run_bo(problem, rounds, PER_ROUND, seed, tag)
    elif arm in AGENT_SWITCHES:
        use_memory, use_gp = AGENT_SWITCHES[arm]
        result = agent.run_agent(problem, rounds, PER_ROUND, tag, memory_file, use_memory, use_gp, seed=seed)
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

class ProcessJobs:
    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=PARALLEL_RUNS)

    def submit(self, function, *arguments):
        return self.pool.submit(function, *arguments)

    def result(self, handle):
        return handle.result()


class RayJobs:
    """The same runs as Ray tasks under CHIA (LOOP_DISPATCH=chia): every
    simulation they launch is a CHIA task too. A run itself needs no CPU: it waits."""

    def __init__(self):
        import ray
        self.ray = ray

    def submit(self, function, *arguments):
        return self.ray.remote(num_cpus=0)(function).remote(*arguments)

    def result(self, handle):
        return self.ray.get(handle)


def run_cell(cell_name, tag, arms=None):
    cell = CELLS[cell_name]
    if arms is None:
        arms = ARMS
    seeds = int(os.environ.get("SEEDS", "2"))
    first_seed = int(os.environ.get("FIRST_SEED", "0"))
    budget = int(os.environ.get("BUDGET", BUDGET))
    rounds = budget // PER_ROUND
    if cell_name == "smoke":
        rounds = 1
    if not os.path.exists(memory_path(cell_name)):
        build_memory(cell_name)
    workloads = []
    for trace_path in cell["test"]:
        workloads.append(trace_short_name(trace_path))
    output_path = report_path(cell_name, tag)
    report = {"cell": cell_name, "tag": tag, "workloads": workloads, "arms": arms, "seeds": seeds, "first_seed": first_seed,
              "budget": rounds * PER_ROUND, "per_round": PER_ROUND, "memory_path": memory_path(cell_name),
              "model": os.environ.get("ANALYST_MODEL", "gemini-2.5-flash"), "runs": {}, "failed": []}
    for arm in arms:
        report["runs"][arm] = {}
    print("cell {} ({}) -> {} | workloads {} | arms {} | seeds {}..{} | {} designs per run".format(
        cell_name, tag, output_path, workloads, arms, first_seed, first_seed + seeds - 1, rounds * PER_ROUND), flush=True)
    started = time.time()
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        jobs = RayJobs()
    else:
        jobs = ProcessJobs()
    handles = []
    # Seed-major: the first wave already covers every arm on seed 0.
    for seed in range(first_seed, first_seed + seeds):
        for arm in arms:
            handles.append((arm, seed, jobs.submit(run_one, arm, cell_name, seed, rounds, cell_name)))
    for arm, seed, handle in handles:
        try:
            result = jobs.result(handle)
        except Exception as error:
            report["failed"].append({"arm": arm, "seed": seed, "error": repr(error)[-400:]})
            print("!! FAILED {} seed {}: {}".format(arm, seed, repr(error)[-200:]), flush=True)
            continue
        report["runs"][arm][str(seed)] = result
        best = max(row["ipc"] for row in result["designs"])
        print("== {} seed {}: best {:.4f} over {} designs".format(arm, seed, best, len(result["designs"]) - 1), flush=True)
        with open(output_path, "w") as report_file:
            json.dump(report, report_file, indent=1)
    report["wall_seconds"] = time.time() - started
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=1)
    print("done in {:.1f} h, {} failed".format(report["wall_seconds"] / 3600.0, len(report["failed"])), flush=True)
    return output_path


def start_chia():
    """Ray in-process with one simulation slot per core and one build slot per
    ChampSim tree copy (Ray also charges each task one CPU, so the pool is cores +
    builds: a build never waits for a simulation), then CHIA's profiler."""
    import ray
    from chia.trace.profiler import start_collector
    from loop.champsim_problem import CHAMPSIM_ROOT
    from loop.simulate import tree_paths
    if os.environ.get("CHIA_ADDRESS") is not None:
        ray.init(address=os.environ["CHIA_ADDRESS"])
    else:
        cores = os.cpu_count()
        builds = len(tree_paths(CHAMPSIM_ROOT))
        ray.init(num_cpus=cores + builds, resources={"champsim": cores, "champsim_build": builds, "vertex_creds": 1},
                 include_dashboard=False, logging_level="ERROR")
    start_collector(log_dir="results/chia_profiles")


if __name__ == "__main__":
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        start_chia()
    if sys.argv[1] == "memory":
        build_memory(sys.argv[2])
        raise SystemExit(0)
    chosen_arms = None
    if len(sys.argv) > 3:
        chosen_arms = sys.argv[3].split(",")
    run_cell(sys.argv[1], sys.argv[2], chosen_arms)
