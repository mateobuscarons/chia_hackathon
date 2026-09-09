"""The transfer experiment, one command per cell.

  python -m loop.run <cell> <tag> [playbook.json|-] [arm,arm,...]

Every cell learns the playbook on chips A+B with the training suite (or reuses
the playbook and frozen pool given by path, which is how the cells share ONE
playbook) and tests every arm on the cell's chip and workloads:
  spec   chip C, the training workloads              (new chip only)
  gap    chip C, the held-out graph workloads        (new chip and new workload class)
  quad   chip D (4 cores, shared LLC), graph mixes   (does it survive a core-count change)
  smoke  chip C, two workloads, two arms, one round  (the gate before any launch)
  learn  no arms: distill the playbook from the A+B tables, save it with its frozen pool,
         and run gate G-language on it (python -m loop.run learn <tag>)
24 designs per run (= 24 x suite-size simulations). Output:
results/experiment_<tag>_<cell>.json (+ _playbook.json, _pool.json).
Env: SEEDS, FIRST_SEED, ANALYST_MODEL, PARALLEL_RUNS, SIM_THREADS.
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
         "bfs.kron": "traces/bfs.kron-128B.champsimtrace.xz"}
TRAINING_SUITE = [TRACE["mcf"], TRACE["omnetpp"], TRACE["lbm"], TRACE["fotonik3d"], TRACE["cam4"]]
HELD_OUT_SUITE = [TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"]]
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
    "quad": {"test_soc": "D_quad", "test_traces": MIXES, "arms": ["rules", "bo_pooled"],
             "rounds": 12, "seeds": 3},
    "smoke": {"test_soc": "C_server", "test_traces": [TRACE["mcf"], TRACE["lbm"]], "arms": ["random", "bo"],
              "rounds": 1, "seeds": 1, "train_traces": [TRACE["mcf"], TRACE["lbm"]]},
}
PER_ROUND = 2


def learn_only(tag):
    """Distill the playbook from the training chips' tables, freeze the pooled
    warm start next to it, and run gate G-language. No test arm runs."""
    from loop import offline, playbook
    import json
    store = {"rules": [], "bets": []}
    prior_history = experiment.learn(store, TRAIN_SOCS, [TRAINING_SUITE], "C")
    playbook_path = "results/experiment_{}_playbook.json".format(tag)
    pool_path = "results/experiment_{}_pool.json".format(tag)
    playbook.save(store, playbook_path)
    with open(pool_path, "w") as pool_file:
        json.dump(prior_history, pool_file)
    print("playbook: {} rules, {} rejected -> {} | pool: {} rows -> {}".format(
        len(store["rules"]), len(store.get("rejected_rules", [])), playbook_path, len(prior_history), pool_path), flush=True)
    offline.language(playbook_path)
    return playbook_path


def run_cell(cell_name, tag, playbook_path=None, arms=None, train_traces=None):
    cell = CELLS[cell_name]
    if arms is None:
        arms = cell["arms"]
    if train_traces is None:
        train_traces = cell.get("train_traces", TRAINING_SUITE)
    # SEEDS=1 runs seed 0 only (a quick look before committing to the whole run).
    seeds = int(os.environ.get("SEEDS", cell["seeds"]))
    # FIRST_SEED=1 SEEDS=2 adds seeds 1-2 to a run that already has seed 0 (same playbook).
    first_seed = int(os.environ.get("FIRST_SEED", "0"))
    output_path = "results/experiment_{}_{}.json".format(tag, cell_name)
    started = time.time()
    print("cell {} ({}) -> {} | playbook: {} | arms: {}".format(
        cell_name, tag, output_path, playbook_path, arms), flush=True)
    experiment.run_experiment(train_socs=TRAIN_SOCS, test_soc=cell["test_soc"], traces=cell["test_traces"],
                              rounds=cell["rounds"], per_round=PER_ROUND, seeds=seeds,
                              output_path=output_path, train_traces=train_traces,
                              space_name="C", arms=arms, playbook_path=playbook_path,
                              suite=True, first_seed=first_seed)
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
