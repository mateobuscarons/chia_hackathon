"""Repeat ONLY the distillation step of a finished learn phase, from the tables.

The learn phase ends with `experiment.distill`: the analyst reads the training
problem's runs and settled bets, proposes rules, and `verify_claims` measures
each claim before admission. This script re-runs that same step for every
training problem of a tier, with the same evidence (every design measured on
the problem, in the result tables) and the same settled bets (from the learn
playbook), so a prompt or verifier change can be evaluated in minutes instead
of re-running the loop. It replicates the step; it does not vary it.

Usage: python -m loop.distill_tables <tier> <learn_playbook.json> <out_playbook.json>
  tier             B or C (same training SoCs/traces/space as loop.run)
  learn_playbook   the _playbook.json a run wrote (its bets are the ledger shown to the analyst)
  out_playbook     where the fresh playbook goes (rules start from zero, as in the learn phase)
"""

import sys

from loop import analyst, experiment, loop, playbook, run


def distill_problem(learn_store, soc_name, traces, space_name):
    """One training problem, exactly as learn_job does it after its loop: fresh
    rule set, this problem's settled bets as the ledger, every measured design as
    the table and as verification evidence."""
    problem = experiment.problem_for(soc_name, traces, space_name)
    tag = "redistill-" + problem["name"]
    designs = experiment.measured_designs(problem)

    # The learn run's bets for this problem (bet experiments are design names,
    # which start with the SoC name).
    problem_bets = []
    for bet in learn_store["bets"]:
        if bet["experiment"].startswith(soc_name):
            problem_bets.append(bet)
    job_store = {"rules": [], "bets": problem_bets}

    print("[{}] {} designs measured, {} settled bets".format(tag, len(designs), len(problem_bets)), flush=True)
    extra_runs = experiment.distill(job_store, problem, designs, tag)
    print("[{}] {} rules admitted, {} rejected, {} verification sims".format(
        tag, len(job_store["rules"]), len(job_store.get("rejected_rules", [])), len(extra_runs)), flush=True)
    return job_store


if __name__ == "__main__":
    tier = run.TIERS[sys.argv[1]]
    learn_store = playbook.load(sys.argv[2])
    destination = sys.argv[3]

    store = {"rules": [], "bets": []}
    for soc_name in run.TRAIN_SOCS:
        if tier["suite"]:
            job_store = distill_problem(learn_store, soc_name, tier["train_traces"], tier["space"])
            experiment.merge_playbook(store, job_store)
        else:
            for trace_path in tier["train_traces"]:
                job_store = distill_problem(learn_store, soc_name, trace_path, tier["space"])
                experiment.merge_playbook(store, job_store)
    playbook.save(store, destination)
    print("playbook: {} rules, {} rejected -> {}".format(
        len(store["rules"]), len(store.get("rejected_rules", [])), destination), flush=True)
