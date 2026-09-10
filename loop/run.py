"""The transfer experiment, one command per cell.

  python -m loop.run <cell> <tag> [playbook.json|-] [arm,arm,...]

Every cell learns the playbook on chips A+B with the training suite (or reuses
the playbook and frozen pool given by path, which is how the cells share ONE
playbook) and tests every arm on the cell's chip and workloads:
  spec   chip C, the training workloads              (new chip only)
  gap    chip C, the held-out graph workloads        (new chip and new workload class)
  quad   chip D (4 cores, shared LLC), graph mixes   (does it survive a core-count change)
  smoke  chip C, two workloads, two arms, one round  (the gate before any launch)
  gap2b  chip C, the other graph workloads               (memory learned in cell gap, tested here)
  learn  no arms: distill the playbook from the A+B tables, build the memory, save both
         with the frozen pool, and run gate G-language (python -m loop.run learn <tag>)
Every arm starts from the previous chip's (B's) best design on the training suite,
fitted to the cell's chip, and counts every design it buys. 24 designs per run
(= 24 x suite-size simulations). Output: results/experiment_<tag>_<cell>.json
(+ _playbook.json, _pool.json, _memory.json).
Env: SEEDS, FIRST_SEED, ROUNDS, ANALYST_MODEL, PARALLEL_RUNS, SIM_THREADS, MEMORY_PATH,
LOOP_WARMUP / LOOP_SIM (a fast pilot: e.g. 2000000 / 4000000, own tables).
Fast pilot of the headline question, ~45 min on the VM:
  LOOP_WARMUP=2000000 LOOP_SIM=4000000 ROUNDS=8 SEEDS=2 python -m loop.run_chia local gap <tag> <playbook> memory,llm_direct,handoff,rules,bo
Consolidate the memory after a cell (for the next one):
  python -m loop.memory consolidate results/experiment_<tag>_memory.json results/experiment_<tag>_memory_after_gap.json
"""

import os
import sys
import time

from loop import experiment

TRAIN_SOCS = ["A_mobile", "B_midrange"]
TRACE = {"mcf": "traces/605.mcf_s-665B.champsimtrace.xz",
         "lbm": "traces/619.lbm_s-2676B.champsimtrace.xz",
         "omnetpp": "traces/620.omnetpp_s-874B.champsimtrace.xz",
         "fotonik3d": "traces/649.fotonik3d_s-10881B.champsimtrace.xz",
         "cam4": "traces/627.cam4_s-490B.champsimtrace.xz",
         "bfs.urand": "traces/bfs.urand-36B.champsimtrace.xz",
         "pr.urand": "traces/pr.urand-129B.champsimtrace.xz",
         "bfs.kron": "traces/bfs.kron-128B.champsimtrace.xz",
         "bfs.road": "traces/bfs.road-99B.champsimtrace.xz",
         "pr.web": "traces/pr.web-16B.champsimtrace.xz",
         "pr.road": "traces/pr.road-28B.champsimtrace.xz"}
TRAINING_SUITE = [TRACE["mcf"], TRACE["omnetpp"], TRACE["lbm"], TRACE["fotonik3d"], TRACE["cam4"]]
HELD_OUT_SUITE = [TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"]]
# Cell 2b: the other graph workloads. PLACEHOLDER third member: the expert named
# cc; its trace is not fetched yet, pr.road stands in until the user decides.
HELD_OUT_SUITE_2B = [TRACE["bfs.road"], TRACE["pr.web"], TRACE["pr.road"]]
# Mixes for the four-core chip: one trace per core. PLACEHOLDER - the team picks
# the mixes; these rotate which held-out workload runs on two cores.
MIXES = [[TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"], TRACE["bfs.urand"]],
         [TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"], TRACE["pr.urand"]],
         [TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"], TRACE["bfs.kron"]]]

CELLS = {
    "spec": {"test_soc": "C_server", "test_traces": TRAINING_SUITE, "arms": experiment.ARMS,
             "rounds": 12, "seeds": 5},
    "gap": {"test_soc": "C_server", "test_traces": HELD_OUT_SUITE, "arms": experiment.ARMS,
            "rounds": 12, "seeds": 10},
    "gap2b": {"test_soc": "C_server", "test_traces": HELD_OUT_SUITE_2B, "arms": experiment.ARMS,
              "rounds": 12, "seeds": 5},
    "quad": {"test_soc": "D_quad", "test_traces": MIXES, "arms": ["rules", "bo_pooled"],
             "rounds": 12, "seeds": 3},
    "smoke": {"test_soc": "C_server", "test_traces": [TRACE["mcf"], TRACE["lbm"]], "arms": experiment.ARMS,
              "rounds": 1, "seeds": 1, "train_traces": [TRACE["mcf"], TRACE["lbm"]]},
}
PER_ROUND = 2


PREVIOUS_CHIP = "B_midrange"


def previous_best_start(target_soc, traces=None):
    """The previous chip's best design on the training suite (geomean over the
    traces measured on all of them), if it fits the target chip's budget; else the
    target's untouched design, with a warning. This is where every arm starts."""
    from loop import champsim_problem, collect
    from loop.configs import within_budget
    if traces is None:
        traces = TRAINING_SUITE
    designs = collect.top_designs(PREVIOUS_CHIP, 1, traces)
    if len(designs) < 2:
        print("no measured design on {} for the training suite; starting from the untouched chip".format(PREVIOUS_CHIP), flush=True)
        return champsim_problem.profiled_baseline(target_soc, "C")
    best = designs[1]
    if not within_budget(best, target_soc, champsim_problem.BASE_CONFIG):
        print("{}'s best design does not fit {}; starting from the untouched chip".format(PREVIOUS_CHIP, target_soc), flush=True)
        return champsim_problem.profiled_baseline(target_soc, "C")
    return best


def learn_only(tag):
    """Distill the playbook from the training chips' tables (pooled), build the
    initial memory from the same tables, freeze the pooled warm start, and run
    gate G-language. No test arm runs."""
    from loop import offline, playbook, memory
    import json
    store = {"rules": [], "bets": []}
    prior_history = experiment.learn(store, TRAIN_SOCS, [TRAINING_SUITE], "C")
    playbook_path = "results/experiment_{}_playbook.json".format(tag)
    pool_path = "results/experiment_{}_pool.json".format(tag)
    memory_path = "results/experiment_{}_memory.json".format(tag)
    playbook.save(store, playbook_path)
    with open(pool_path, "w") as pool_file:
        json.dump(prior_history, pool_file)
    print("playbook: {} rules, {} rejected -> {} | pool: {} rows -> {}".format(
        len(store["rules"]), len(store.get("rejected_rules", [])), playbook_path, len(prior_history), pool_path), flush=True)
    memory.build_initial(TRAIN_SOCS, TRAINING_SUITE, memory_path)
    offline.language(playbook_path)
    return playbook_path


def run_cell(cell_name, tag, playbook_path=None, arms=None, train_traces=None):
    cell = CELLS[cell_name]
    if arms is None:
        arms = cell["arms"]
    if train_traces is None:
        train_traces = cell.get("train_traces", TRAINING_SUITE)
    # SEEDS=1 runs seed 0 only (a quick look before committing to the whole run);
    # ROUNDS=8 shortens a pilot. With LOOP_WARMUP / LOOP_SIM the short runs land in
    # their own tables, so a fast pilot never mixes with the real cells.
    seeds = int(os.environ.get("SEEDS", cell["seeds"]))
    rounds = int(os.environ.get("ROUNDS", cell["rounds"]))
    # FIRST_SEED=1 SEEDS=2 adds seeds 1-2 to a run that already has seed 0 (same playbook).
    first_seed = int(os.environ.get("FIRST_SEED", "0"))
    output_path = "results/experiment_{}_{}.json".format(tag, cell_name)
    start_knobs = previous_best_start(cell["test_soc"])
    # The memory a cell reads: the one built with its playbook, unless MEMORY_PATH
    # names a consolidated one (cell gap2b reads the memory that includes cell gap).
    memory_path = "results/experiment_{}_memory.json".format(tag)
    if playbook_path is not None:
        memory_path = playbook_path.replace("_playbook.json", "_memory.json")
    memory_path = os.environ.get("MEMORY_PATH", memory_path)
    started = time.time()
    print("cell {} ({}) -> {} | playbook: {} | memory: {} | arms: {} | start: {}".format(
        cell_name, tag, output_path, playbook_path, memory_path, arms,
        {key: start_knobs[key] for key in ["l2_sets", "l2_ways", "l2_prefetcher", "llc_sets", "llc_ways", "llc_replacement"]}), flush=True)
    experiment.run_experiment(train_socs=TRAIN_SOCS, test_soc=cell["test_soc"], traces=cell["test_traces"],
                              rounds=rounds, per_round=PER_ROUND, seeds=seeds,
                              output_path=output_path, train_traces=train_traces,
                              space_name="C", arms=arms, playbook_path=playbook_path,
                              suite=True, first_seed=first_seed, start_knobs=start_knobs, memory_path=memory_path)
    print("done in {:.1f} h".format((time.time() - started) / 3600.0), flush=True)
    return output_path


if __name__ == "__main__":
    cell_name = sys.argv[1]
    tag = sys.argv[2]
    if cell_name == "learn":
        learn_only(tag)
        raise SystemExit(0)
    playbook_path = None
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        playbook_path = sys.argv[3]
    arms = None
    if len(sys.argv) > 4:
        arms = sys.argv[4].split(",")
    run_cell(cell_name, tag, playbook_path, arms)
