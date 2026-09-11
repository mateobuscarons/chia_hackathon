"""The transfer experiment, one command per cell.

  python -m loop.run learn <cell> <tag>                       # the cell's playbook, pool and memory from its training tables
  python -m loop.run <cell> <tag> [playbook.json|-] [arm,arm,...]

Headline: workload transfer on ONE chip. Memory written from chip C's own SPEC
results is tested on graph workloads on chip C, then memory that includes those
searches is tested on other graph workloads on chip C. Chip transfer is a small
positive control. Cells:
  w1     memory from chip C's SPEC tables; tested on GAP set 1 on chip C
         (class boundary, chip fixed: facts vs strategies)
  w2     memory = w1's memory + every w1 run, consolidated once and frozen
         (python -m loop.memory consolidate); tested on GAP set 2 on chip C
         (same family: the headline)
  k      memory from chips A+B's SPEC tables; tested on the SPEC suite on chip C
         (chip boundary, workloads fixed: the near-free transfer)
  smoke  every arm, one round, two workloads, through CHIA (the gate before any launch)
Every arm of a cell starts from the same design: the best design measured on the
cell's SOURCE chip over the SPEC suite (chip C itself for w1/w2, chip B for k),
fitted to the target's budget; it counts as design 0, every design bought after
it counts. Rounds of 2 designs; 24 rounds = 48 designs and two seeds while iterating.
Output: results/experiment_<tag>_<cell>.json (+ _playbook.json, _pool.json,
_memory.json from the learn step, _memory_run-*.json written by the agent arms).
Env: SEEDS, FIRST_SEED, ROUNDS, ANALYST_MODEL, PARALLEL_RUNS, SIM_THREADS,
MEMORY_PATH (w2 reads the consolidated memory), LOOP_WARMUP / LOOP_SIM (a short
pilot lands in its own tables).
"""

import os
import sys
import time

from loop import experiment

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
         "pr.road": "traces/pr.road-28B.champsimtrace.xz",
         "sssp.kron": "traces/sssp.kron-246B.champsimtrace.xz",
         "cc.urand": "traces/cc.urand-353B.champsimtrace.xz",
         "cc.twitter": "traces/cc.twitter-15B.champsimtrace.xz"}
# The SPEC suite every chip has measured rows on (chip C has no fotonik3d / cam4
# rows): the source of w1's memory, k's test suite, and the suite the start
# design is chosen on.
SPEC_SUITE = [TRACE["mcf"], TRACE["omnetpp"], TRACE["lbm"]]
# Chips A and B also measured the two capacity workloads added later; k's memory reads all five.
SPEC_SUITE_AB = SPEC_SUITE + [TRACE["fotonik3d"], TRACE["cam4"]]
GAP_SET_1 = [TRACE["bfs.urand"], TRACE["pr.urand"], TRACE["bfs.kron"]]
# Set 2, chosen by measuring 11 designs on six candidates (not by the gate alone):
# +33.5% headroom from the untouched chip, of which 61.8% no single knob reaches
# (GAP set 1: +144.5% but only 29.7% beyond one knob, which is why its arms tied).
# The two shortest-path workloads want a replacement policy, the two connected-
# components workloads want a prefetcher, so no single move satisfies the suite.
GAP_SET_2 = [TRACE["sssp.kron"], TRACE["cc.urand"], TRACE["cc.twitter"]]

CELLS = {
    "w1": {"test_soc": "C_server", "test_traces": GAP_SET_1,
           "memory_socs": ["C_server"], "memory_traces": SPEC_SUITE, "start_soc": "C_server",
           "arms": ["bo", "llm_direct", "memory"],
           "rounds": 24, "seeds": 2},
    # w2 starts from the untouched chip: from the SPEC-best start GAP set 1 had
    # only +6.7% and every arm landed within 1%, inside seed variance.
    "w2": {"test_soc": "C_server", "test_traces": GAP_SET_2,
           "memory_socs": ["C_server"], "memory_traces": SPEC_SUITE, "start_soc": None,
           "arms": ["bo", "llm_direct", "memory"],
           "rounds": 24, "seeds": 2},
    "k": {"test_soc": "C_server", "test_traces": SPEC_SUITE,
          "memory_socs": ["A_mobile", "B_midrange"], "memory_traces": SPEC_SUITE_AB, "start_soc": "B_midrange",
          "arms": ["bo_pooled", "llm_direct", "memory"],
          "rounds": 24, "seeds": 2},
    "smoke": {"test_soc": "C_server", "test_traces": [TRACE["mcf"], TRACE["lbm"]],
              "memory_socs": ["C_server"], "memory_traces": SPEC_SUITE, "start_soc": "C_server",
              "arms": experiment.ARMS, "rounds": 1, "seeds": 1},
}
PER_ROUND = 2
# The `rules` arm runs fewer seeds than the agents (it is a baseline, not a contender).
RULES_SEEDS = 5


def start_design(cell):
    """The best design measured on the cell's source chip over the SPEC suite
    (geomean over the cached rows), if it fits the target chip's budget; else the
    target's untouched chip, with a warning. start_soc None means the cell starts
    from the untouched chip on purpose (w2: the tuned start left too little
    headroom for any arm to separate)."""
    from loop import champsim_problem, collect
    from loop.configs import within_budget
    if cell["start_soc"] is None:
        return champsim_problem.profiled_baseline(cell["test_soc"], "C")
    designs = collect.top_designs(cell["start_soc"], 1, SPEC_SUITE)
    if len(designs) < 2:
        print("no measured design on {} for the SPEC suite; starting from the untouched chip".format(cell["start_soc"]), flush=True)
        return champsim_problem.profiled_baseline(cell["test_soc"], "C")
    best = dict(designs[1])
    # A row measured before a knob joined the space carries that knob at the untouched chip's value.
    untouched = champsim_problem.profiled_baseline(cell["test_soc"], "C")
    for knob in untouched:
        if knob not in best:
            best[knob] = untouched[knob]
    if not within_budget(best, cell["test_soc"], champsim_problem.BASE_CONFIG):
        print("{}'s best design does not fit {}; starting from the untouched chip".format(cell["start_soc"], cell["test_soc"]), flush=True)
        return champsim_problem.profiled_baseline(cell["test_soc"], "C")
    return best


def learn_only(cell_name, tag):
    """The cell's knowledge from its training tables: the playbook (rules, pooled
    over the memory chips), the frozen pool for bo_pooled, and the memory (cases,
    facts, strategies). No test arm runs."""
    from loop import memory, playbook
    import json
    cell = CELLS[cell_name]
    store = {"rules": [], "bets": []}
    prior_history = experiment.learn(store, cell["memory_socs"], [cell["memory_traces"]], "C")
    playbook_path = "results/experiment_{}_playbook.json".format(tag)
    pool_path = "results/experiment_{}_pool.json".format(tag)
    memory_path = "results/experiment_{}_memory.json".format(tag)
    playbook.save(store, playbook_path)
    with open(pool_path, "w") as pool_file:
        json.dump(prior_history, pool_file)
    print("playbook: {} rules, {} rejected -> {} | pool: {} rows -> {}".format(
        len(store["rules"]), len(store.get("rejected_rules", [])), playbook_path, len(prior_history), pool_path), flush=True)
    memory.build(cell["memory_socs"], cell["memory_traces"], memory_path)
    return playbook_path


def run_cell(cell_name, tag, playbook_path=None, arms=None):
    cell = CELLS[cell_name]
    if arms is None:
        arms = cell["arms"]
    # SEEDS=1 runs seed 0 only (a quick look before committing to the whole run);
    # ROUNDS=4 shortens a pilot. FIRST_SEED=5 SEEDS=5 adds seeds 5-9 to a run that
    # already has seeds 0-4 (same playbook and memory; summarize merges the reports).
    seeds = int(os.environ.get("SEEDS", cell["seeds"]))
    rounds = int(os.environ.get("ROUNDS", cell["rounds"]))
    first_seed = int(os.environ.get("FIRST_SEED", "0"))
    output_path = "results/experiment_{}_{}.json".format(tag, cell_name)
    start_knobs = start_design(cell)
    # The memory a cell reads: the one built with its playbook, unless MEMORY_PATH
    # names a consolidated one (cell w2 reads the memory that includes cell w1).
    memory_path = "results/experiment_{}_memory.json".format(tag)
    if playbook_path is not None:
        memory_path = playbook_path.replace("_playbook.json", "_memory.json")
    memory_path = os.environ.get("MEMORY_PATH", memory_path)
    started = time.time()
    print("cell {} ({}) -> {} | playbook: {} | memory: {} | arms: {} | start: {}".format(
        cell_name, tag, output_path, playbook_path, memory_path, arms,
        {key: start_knobs[key] for key in ["l2_sets", "l2_ways", "l2_prefetcher", "llc_sets", "llc_ways",
                                            "llc_prefetcher", "llc_replacement"]}), flush=True)
    seeds_per_arm = {}
    for arm in arms:
        seeds_per_arm[arm] = seeds
        if arm == "rules" and cell_name != "smoke":
            seeds_per_arm[arm] = min(seeds, RULES_SEEDS)
    experiment.run_experiment(train_socs=cell["memory_socs"], test_soc=cell["test_soc"], traces=cell["test_traces"],
                              rounds=rounds, per_round=PER_ROUND, seeds=seeds_per_arm,
                              output_path=output_path, train_traces=cell["memory_traces"],
                              space_name="C", arms=arms, playbook_path=playbook_path,
                              suite=True, first_seed=first_seed, start_knobs=start_knobs, memory_path=memory_path)
    print("done in {:.1f} h".format((time.time() - started) / 3600.0), flush=True)
    return output_path


if __name__ == "__main__":
    if sys.argv[1] == "learn":
        learn_only(sys.argv[2], sys.argv[3])
        raise SystemExit(0)
    cell_name = sys.argv[1]
    tag = sys.argv[2]
    playbook_path = None
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        playbook_path = sys.argv[3]
    arms = None
    if len(sys.argv) > 4:
        arms = sys.argv[4].split(",")
    run_cell(cell_name, tag, playbook_path, arms)
