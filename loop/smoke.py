"""Smoke test: the WHOLE experiment on a tiny budget. Nothing big launches until this passes.

Runs learn (one SoC, one trace) + all seven arms on another SoC, 2 rounds, 1 seed,
then asserts every arm produced a result and no parallel job failed. ~1 minute, ~$0.05.
"""

import json

from loop import experiment

OUTPUT = "results/smoke.json"

if __name__ == "__main__":
    experiment.run_experiment(train_socs=["A_mobile"], test_soc="C_server",
                              traces=["traces/605.mcf_s-665B.champsimtrace.xz"],
                              rounds=2, per_round=2, seeds=1, output_path=OUTPUT)
    report = json.load(open(OUTPUT))
    arms_done = list(report["test"]["C_server/mcf"]["arms"].keys())
    missing = [arm for arm in experiment.ARMS if "0" not in report["test"]["C_server/mcf"]["arms"].get(arm, {})]
    print("arms with results:", len(arms_done) - len(missing), "of", len(experiment.ARMS), "| missing:", missing)
    print("failed jobs:", report["failed_jobs"])
    print("rules distilled:", len(report["rules"]))
    assert len(missing) == 0 and len(report["failed_jobs"]) == 0, "SMOKE TEST FAILED"
    print("SMOKE TEST PASSED")
