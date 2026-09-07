"""The transfer experiment, one command per tier.

  python -m loop.run B <tag> [playbook.json|-] [arm,arm,...]
      Tier B (46k designs, one workload per problem): learn on A+B (mcf, lbm),
      test on unseen C (mcf, lbm, omnetpp). 5 arms, 3 seeds, 24 designs.
  python -m loop.run C <tag> [playbook.json|-] [arm,arm,...]
      Tier C, the hard tier (415k designs, latency coupled to size, MSHRs): learn
      on A+B, test on C, each problem a SUITE of workloads scored by geomean IPC.
      24 designs (= 24 x suite-size simulations) per run.

Output: results/experiment_<tag>_C.json (+ _playbook.json). A playbook path
reuses learned rules (same knowledge, new mechanism); the arm list restricts
which arms run (e.g. "full" after a change that only touches the analyst).
"""

import os
import sys
import time

from loop import experiment

TRAIN_SOCS = ["A_mobile", "B_midrange"]
TEST_SOC = "C_server"
TRACE = {"mcf": "traces/605.mcf_s-665B.champsimtrace.xz",
         "lbm": "traces/619.lbm_s-2676B.champsimtrace.xz",
         "omnetpp": "traces/620.omnetpp_s-874B.champsimtrace.xz",
         "xalancbmk": "traces/623.xalancbmk_s-700B.champsimtrace.xz"}

TIERS = {
    "B": {"space": "B", "suite": False,
          "train_traces": [TRACE["mcf"], TRACE["lbm"]],
          "test_traces": [TRACE["mcf"], TRACE["lbm"], TRACE["omnetpp"]],
          "learn_rounds": 10, "test_rounds": 12, "per_round": 2, "seeds": 3},
    # Proof-of-concept size (user decision Sep 4): 4 workloads, 3 seeds.
    "C": {"space": "C", "suite": True,
          "train_traces": [TRACE["mcf"], TRACE["lbm"], TRACE["omnetpp"], TRACE["xalancbmk"]],
          "test_traces": [TRACE["mcf"], TRACE["lbm"], TRACE["omnetpp"], TRACE["xalancbmk"]],
          "learn_rounds": 10, "test_rounds": 12, "per_round": 2, "seeds": 3},
}
# The paper's core comparison: statistical baselines against rule transfer and the whole loop.
ARMS = ["random", "bo", "bo_pooled", "rules", "full"]

if __name__ == "__main__":
    tier = TIERS[sys.argv[1]]
    tag = sys.argv[2]
    playbook_path = None
    if len(sys.argv) > 3 and sys.argv[3] != "-":
        playbook_path = sys.argv[3]
    arms = ARMS
    if len(sys.argv) > 4:
        arms = sys.argv[4].split(",")
    # SEEDS=1 runs seed 0 only (a quick look before committing to the whole run).
    seeds = int(os.environ.get("SEEDS", tier["seeds"]))
    output_path = "results/experiment_{}_C.json".format(tag)
    started = time.time()
    print("tier {} ({}) -> {} | playbook: {} | arms: {}".format(
        sys.argv[1], tag, output_path, playbook_path, arms), flush=True)
    experiment.run_experiment(train_socs=TRAIN_SOCS, test_soc=TEST_SOC, traces=tier["test_traces"],
                              rounds=tier["learn_rounds"], per_round=tier["per_round"], seeds=seeds,
                              output_path=output_path, train_traces=tier["train_traces"],
                              space_name=tier["space"], arms=arms, test_rounds=tier["test_rounds"],
                              playbook_path=playbook_path, suite=tier["suite"])
    print("done in {:.1f} h".format((time.time() - started) / 3600.0), flush=True)
