"""Smoke test: the WHOLE experiment on a tiny budget. Nothing big launches until this passes.

Runs learn (one SoC, one trace) + all seven arms on another SoC, 2 rounds, 1 seed,
then asserts every arm produced a result and no parallel job failed. ~1 minute, ~$0.05.
"""

import json

from loop import experiment

OUTPUT = "results/smoke.json"

TIER_B_ARMS = ["random", "surrogate", "surrogate_pooled", "rules", "full"]

if __name__ == "__main__":
    import sys
    space_name = sys.argv[1] if len(sys.argv) > 1 else "A"
    arms = experiment.ARMS if space_name == "A" else TIER_B_ARMS
    rounds = 2 if space_name == "A" else 1          # Tier B simulates for real: keep it tiny
    experiment.run_experiment(train_socs=["A_mobile"], test_soc="C_server",
                              traces=["traces/605.mcf_s-665B.champsimtrace.xz"],
                              rounds=rounds, per_round=2, seeds=1, output_path=OUTPUT,
                              space_name=space_name, arms=arms)
    report = json.load(open(OUTPUT))
    arms_done = list(report["test"]["C_server/mcf"]["arms"].keys())
    missing = [arm for arm in arms if "0" not in report["test"]["C_server/mcf"]["arms"].get(arm, {})]
    print("arms with results:", len(arms_done) - len(missing), "of", len(arms), "| missing:", missing)
    print("failed jobs:", report["failed_jobs"])
    print("rules distilled:", len(report["rules"]))
    assert len(missing) == 0 and len(report["failed_jobs"]) == 0, "SMOKE TEST FAILED"
    print("SMOKE TEST PASSED")
