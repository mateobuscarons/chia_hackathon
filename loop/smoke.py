"""Smoke test: the WHOLE experiment on a tiny budget. Nothing big launches until this passes.

Runs learn (one SoC, one trace) + all seven arms on another SoC, 2 rounds, 1 seed,
then asserts every arm produced a result and no parallel job failed. ~1 minute, ~$0.05.
"""

import json

from loop import experiment

OUTPUT = "results/smoke.json"

TIER_B_ARMS = ["random", "bo", "bo_pooled", "surrogate", "rules", "full"]

if __name__ == "__main__":
    import sys
    space_name = sys.argv[1] if len(sys.argv) > 1 else "A"
    arms = experiment.ARMS if space_name == "A" else TIER_B_ARMS
    rounds = 2 if space_name == "A" else 1          # Tiers B/C simulate for real: keep it tiny
    traces = ["traces/605.mcf_s-665B.champsimtrace.xz"]
    suite = False
    if space_name == "C":
        # Tier C is a suite problem: two workloads, real simulations, one round.
        traces = ["traces/605.mcf_s-665B.champsimtrace.xz", "traces/619.lbm_s-2676B.champsimtrace.xz"]
        suite = True
    experiment.run_experiment(train_socs=["A_mobile"], test_soc="C_server",
                              traces=traces,
                              rounds=rounds, per_round=2, seeds=1, output_path=OUTPUT,
                              space_name=space_name, arms=arms, suite=suite)
    report = json.load(open(OUTPUT))
    problem_name = list(report["test"].keys())[0]
    arms_done = list(report["test"][problem_name]["arms"].keys())
    missing = [arm for arm in arms if "0" not in report["test"][problem_name]["arms"].get(arm, {})]
    print("arms with results:", len(arms_done) - len(missing), "of", len(arms), "| missing:", missing)
    print("failed jobs:", report["failed_jobs"])
    print("rules distilled:", len(report["rules"]))
    assert len(missing) == 0 and len(report["failed_jobs"]) == 0, "SMOKE TEST FAILED"
    print("SMOKE TEST PASSED")
