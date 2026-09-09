"""The transfer experiment: learn a playbook on SoCs A and B, test it on unseen SoC C.

The playbook is distilled from the result tables of the training chips: a
structured set of designs (the baseline, every one-knob change, key pairs, a
spread of mixed designs) simulated on the training suite, so every knob has a
controlled pair and every claim can be measured before a rule is admitted.

Arms on the target SoC (same simulation cap for all):
  random          - shuffle the candidates, run them in order
  bo              - textbook Bayesian optimisation: GP + expected improvement (cold start)
  bo_pooled       - the same BO warm-started with every A/B run (statistical transfer)
  textbook        - GP + rules the LLM wrote BEFORE seeing any result
  rules           - GP + rules distilled from A/B (rule transfer, no LLM in the loop)
  full            - GP + distilled rules + the analyst's reply on a stall (the whole loop)
  llm_direct      - the plain agent: the LLM picks every design from the results table alone
The pooled arm's warm start is frozen once per playbook (a `_pool.json` next to
it) so every cell and seed sees the same prior data. Reports keep every run's history; `summarize` scores designs-to-target against
each cell's fixed reference.
"""

import copy
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor

from loop import forecast, analyst, loop, playbook, surrogate_gp
import os

from loop.champsim_problem import aggregate_suite, enrich_metrics, make_problem, make_suite_problem, trace_short_name

# Parallel arm runs (each simulates per_round designs x suite size at a time).
PARALLEL_RUNS = int(os.environ.get("PARALLEL_RUNS", "5"))


class ProcessJobs:
    """Arm runs and learn jobs as local processes (python -m loop.run)."""

    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=PARALLEL_RUNS)

    def submit(self, function, *arguments):
        return self.pool.submit(function, *arguments)

    def result(self, handle):
        return handle.result()

    def shutdown(self):
        self.pool.shutdown()


class RayJobs:
    """The same jobs as Ray tasks under CHIA (python -m loop.run_chia): every
    simulation they launch is a CHIA task too, so the profiler sees the whole
    loop as one task graph. An arm job itself needs no CPU: it waits on tasks."""

    def __init__(self):
        import ray
        self.ray = ray

    def submit(self, function, *arguments):
        remote_function = self.ray.remote(num_cpus=0)(function)
        return remote_function.remote(*arguments)

    def result(self, handle):
        return self.ray.get(handle)

    def shutdown(self):
        return


def make_jobs():
    if os.environ.get("LOOP_DISPATCH", "local") == "chia":
        return RayJobs()
    return ProcessJobs()


def problem_for(soc_name, traces, space_name, allow_simulation=True):
    """`traces` is one trace path (single-workload problem) or a list (suite)."""
    if isinstance(traces, list):
        return make_suite_problem(soc_name, traces, allow_simulation=allow_simulation, space_name=space_name)
    return make_problem(soc_name, traces, allow_simulation=allow_simulation, space_name=space_name)


def learn(store, soc_names, traces, space_name):
    """Distill rules from every training (SoC, suite) in parallel and merge the
    playbooks into `store`. Returns the pooled history of every measured training
    design (the pooled arm's warm start)."""
    jobs = make_jobs()
    handles = []
    for soc_name in soc_names:
        for trace_path in traces:
            handles.append((soc_name, jobs.submit(learn_job, soc_name, trace_path, space_name)))
    prior_history = []
    for soc_name, handle in handles:
        job_store, history = jobs.result(handle)
        merge_playbook(store, job_store)
        for entry in history:
            prior_history.append(with_soc(entry, soc_name))
    jobs.shutdown()
    return prior_history


# Controlled comparisons the distiller may run to measure a proposed rule's claim.
VERIFY_SIMS_PER_PROBLEM = 5


def learn_job(soc_name, trace_path, space_name):
    """One training problem: every design measured on it is the evidence; the
    analyst proposes rules over the measured one-knob effects and each claim is
    verified by controlled comparison before it is admitted."""
    problem = problem_for(soc_name, trace_path, space_name)
    job_store = {"rules": [], "bets": []}
    tag = "learn-" + problem["name"]
    reference = problem["evaluate"](problem["baseline"])[problem["objective"]]
    history = measured_designs(problem)
    for entry in history:
        entry["reference"] = reference
    extra_runs = distill(job_store, problem, history, tag)
    for entry in extra_runs:
        entry["reference"] = reference
    return job_store, history + extra_runs


def training_runs(soc_names, traces, space_name):
    """Every simulated config of the training SoCs, tagged with its SoC (pooled prior)."""
    runs = []
    for soc_name in soc_names:
        for trace_path in traces:
            problem = problem_for(soc_name, trace_path, space_name, allow_simulation=False)
            designs = measured_designs(problem)
            # Each run carries its own problem's baseline so the surrogate can learn
            # in "speedup over baseline" units across chips.
            reference = problem["evaluate"](problem["baseline"])[problem["objective"]]
            for design in designs:
                design["reference"] = reference
                runs.append(with_soc(design, soc_name))
    return runs


def merge_playbook(store, job_store):
    """Append another playbook's rules, rejected rules and bets, giving rules fresh ids."""
    new_ids = {}
    for rule in job_store["rules"]:
        new_id = playbook.add_rule(store, rule["condition"], rule["claim"], rule["example"], rule["text"],
                                   conditions=rule.get("conditions"), verification=rule.get("verification"))
        new_ids[rule["id"]] = new_id
        merged = store["rules"][-1]
        for field in ["wins", "losses", "claim_wins", "claim_losses", "brier_scores", "origin", "status"]:
            if field in rule:
                merged[field] = rule[field]
    for rejected in job_store.get("rejected_rules", []):
        store.setdefault("rejected_rules", []).append(rejected)
    for bet in job_store["bets"]:
        bet = dict(bet)
        bet["id"] = "BET-{:04d}".format(len(store["bets"]) + 1)
        bet["forecaster"] = new_ids.get(bet["forecaster"], bet["forecaster"])
        store["bets"].append(bet)


def distill(store, problem, history, tag):
    table = loop.format_table(history, problem["table_metrics"])
    ledger_lines = []
    for bet in store["bets"]:
        if bet["outcome"] is not None:
            ledger_lines.append("{} bet '{}' on {} with p={:.2f}: {}".format(
                bet["forecaster"], bet["event"], bet["experiment"], bet["probability"],
                "happened" if bet["outcome"] else "did not happen"))
    # The LLM proposes; the simulator measures every claim before admission, using
    # every design ever measured on this problem as evidence (not just this run).
    # The same evidence, as measured one-knob effects, is shown to the analyst so
    # it spends its rules on the largest effects instead of the first it notices.
    evidence = measured_designs(problem)
    effects_text = forecast.format_effects(forecast.one_knob_effects(evidence, problem["objective"]))
    descriptors_text = "(not available)"
    if problem.get("descriptors") is not None:
        descriptors_text = json.dumps(problem["descriptors"], indent=1)
    proposals = analyst.distill_rules(problem["search_space"], table,
                                      "\n".join(ledger_lines[-60:]),
                                      loop.format_rules(store["rules"]),
                                      problem["condition_metrics"], effects_text=effects_text,
                                      descriptors_text=descriptors_text)
    return loop.verify_claims(store, proposals, problem, history, tag, VERIFY_SIMS_PER_PROBLEM,
                              evidence=evidence)


def measured_designs(problem):
    """Every design ever measured on this problem. For a suite: designs measured
    on EVERY workload of the suite, with suite-aggregated metrics."""
    holders = problem.get("holders", [problem["holder"]])
    first_table = holders[0].sweep_table
    designs = []
    for name in first_table:
        if first_table[name]["metrics"] is None or "soc" in first_table[name]["knobs"]:
            continue
        per_trace = []
        complete = True
        for holder in holders:
            entry = holder.sweep_table.get(name)
            if entry is None or entry["metrics"] is None:
                complete = False
                break
            per_trace.append(enrich_metrics(dict(entry["metrics"])))
        if not complete:
            continue
        if len(holders) == 1:
            metrics = per_trace[0]
        else:
            short_names = []
            for holder in holders:
                short_names.append(trace_short_name(holder.trace_path))
            metrics = aggregate_suite(per_trace, short_names)
        designs.append({"name": name, "knobs": first_table[name]["knobs"], "metrics": metrics})
    return designs


def well_formed(rule, problem):
    """Every rule the LLM writes (distill, textbook, re-scope) passes through here."""
    return loop.valid_rule(rule, problem)


def with_soc(entry, soc_name):
    """Pooled-surrogate transfer needs to know which design a run came from."""
    tagged = copy.deepcopy(entry)
    tagged["knobs"]["soc"] = soc_name
    return tagged


def random_arm(problem, budget, seed):
    names = list(problem["candidates"].keys())
    random.Random(seed).shuffle(names)
    history = [{"name": "baseline", "knobs": problem["baseline"],
                "metrics": problem["evaluate"](problem["baseline"])}]
    knobs_list = [problem["candidates"][name] for name in names[:budget]]
    metrics_list = problem["evaluate_many"](knobs_list)
    for name, knobs, metrics in zip(names[:budget], knobs_list, metrics_list):
        history.append({"name": name, "knobs": knobs, "metrics": metrics})
    return history


def run_arm_job(arm, store, soc_name, trace_path, rounds, per_round, prior_history, seed, space_name):
    """One parallel job: run an arm and return only what the report keeps."""
    random.seed(seed)
    history, arm_store, round_logs = run_arm(arm, store, soc_name, trace_path, rounds, per_round,
                                             prior_history, seed, space_name)
    # Round logs (hypotheses, chosen designs, claim tests, stall scans) stay in the
    # report so a review can see what the analyst proposed and what was picked.
    return {"history": compact(history, "ipc"), "full_history": history,
            "bets": arm_store["bets"], "rules": arm_store["rules"], "rounds": round_logs}


def run_arm(arm, store, soc_name, trace_path, rounds, per_round, prior_history, seed, space_name):
    problem = problem_for(soc_name, trace_path, space_name)
    arm_store = copy.deepcopy(store)          # every arm starts from the same playbook
    arm_store["bets"] = []                    # ...but with its own empty ledger (learn bets live in report["learn_bets"])
    if arm == "random":
        return random_arm(problem, rounds * per_round, seed), arm_store, []
    if arm == "llm_direct":
        arm_store = {"rules": [], "bets": []}
        tag = "llm_direct-{}-s{}".format(problem["name"], seed)
        result = loop.run_llm_direct(problem, rounds, per_round, arm_store, tag, seed=seed)
        return result["history"], arm_store, result["rounds"]
    if arm == "textbook":
        # Textbook rules are admitted UNVERIFIED on purpose: this arm measures what
        # prior knowledge alone is worth; the loop's rules must beat it.
        arm_store = {"rules": [], "bets": []}
        for rule in analyst.textbook_rules(problem["search_space"], problem["objective"], 6,
                                           problem["condition_metrics"]):
            if well_formed(rule, problem):
                clauses = loop.rule_clauses_of(rule)
                playbook.add_rule(arm_store, clauses[0], rule["claim"], rule["example"], rule["text"],
                                  conditions=clauses, verification={"pairs": 0, "llm_gain_pct": rule["claim"]["gain_pct"]})
    prior = []
    if arm == "bo_pooled":
        prior = prior_history
        problem["search_space"] = dict(problem["search_space"], soc=["A_mobile", "B_midrange", "C_server"])
        for name in problem["candidates"]:
            problem["candidates"][name] = dict(problem["candidates"][name], soc=soc_name)
        problem["baseline"] = dict(problem["baseline"], soc=soc_name)
    use_rules = arm in ["textbook", "rules", "full"]
    use_analyst = arm == "full"
    # Every model-driven arm selects by expected improvement; the difference
    # between arms is what warms the surrogate and who bets.
    tag = "{}-{}-s{}".format(arm, problem["name"], seed)
    result = loop.run_loop(problem, rounds, per_round, arm_store, surrogate_gp,
                           use_rules, use_analyst, tag, prior_history=prior, seed=seed)
    return result["history"], arm_store, result["rounds"]


ARMS = ["random", "bo", "bo_pooled", "textbook", "rules", "full", "llm_direct"]


def compact(history, objective):
    """Only what the plots need: config name and objective per run, in order."""
    rows = []
    for entry in history:
        rows.append({"name": entry["name"], objective: entry["metrics"][objective]})
    return rows


def run_experiment(train_socs, test_soc, traces, rounds, per_round, seeds, output_path,
                   train_traces=None, space_name="C", arms=None,
                   playbook_path=None, first_seed=0, suite=False):
    """suite=True: all `traces` form ONE problem per SoC (geomean objective);
    otherwise each trace is its own problem. `rounds` x `per_round` is each arm's
    design budget on the test chip.
    train_traces defaults to `traces` (cross-trace transfer passes another list).
    space_name names the search space; arms defaults to all.
    playbook_path reuses an already-learned playbook (more seeds, same knowledge);
    the pooled arm then gets every measured training design as prior data."""
    if train_traces is None:
        train_traces = traces
    if suite:
        # One problem per SoC, made of every trace.
        traces = [list(traces)]
        train_traces = [list(train_traces)]
    if arms is None:
        arms = ARMS
    started = time.time()
    if playbook_path is None:
        store = {"rules": [], "bets": []}
        prior_history = learn(store, train_socs, train_traces, space_name)
        pool_path = output_path.replace(".json", "_pool.json")
    else:
        store = playbook.load(playbook_path)
        # The frozen pool lives next to the playbook it was made with; it is built
        # once from every measured training design and never changes afterwards.
        if playbook_path.endswith("_playbook.json"):
            pool_path = playbook_path.replace("_playbook.json", "_pool.json")
        else:
            pool_path = playbook_path.replace(".json", "_pool.json")
        if os.path.exists(pool_path):
            with open(pool_path) as pool_file:
                prior_history = json.load(pool_file)
        else:
            prior_history = training_runs(train_socs, train_traces, space_name)
    playbook.save(store, output_path.replace(".json", "_playbook.json"))
    with open(pool_path, "w") as pool_file:
        json.dump(prior_history, pool_file)

    report = {"settings": {"train_socs": train_socs, "test_soc": test_soc, "traces": traces,
                           "train_traces": train_traces, "space": space_name, "arms": arms, "suite": suite,
                           "rounds": rounds, "per_round": per_round, "seeds": seeds},
              "learn_bets": store["bets"], "rules": store["rules"],
              "rejected_rules": store.get("rejected_rules", []), "test": {}}

    # Every (trace, arm, seed) run is independent: run them in parallel jobs.
    # Each one returns a small dict; the report is saved after every result.
    jobs = make_jobs()
    handles = []
    for trace_path in traces:
        problem = problem_for(test_soc, trace_path, space_name)
        report["test"][problem["name"]] = {"arms": {}}
        for arm in arms:
            report["test"][problem["name"]]["arms"][arm] = {}
        # Seed-major order: the first wave of parallel jobs already covers every arm
        # on seed 0, so a running experiment shows an early cross-arm comparison.
        for seed in range(first_seed, first_seed + seeds):
            for arm in arms:
                handle = jobs.submit(run_arm_job, arm, store, test_soc, trace_path,
                                     rounds, per_round, prior_history, seed, space_name)
                handles.append((problem["name"], arm, seed, handle))

    report["failed_jobs"] = []
    for problem_name, arm, seed, handle in handles:
        try:
            run = jobs.result(handle)
        except Exception as error:
            # One bad job must never abort the experiment; it is recorded and visible.
            report["failed_jobs"].append({"problem": problem_name, "arm": arm, "seed": seed,
                                          "error": repr(error)[-400:]})
            print("!! FAILED {} / {} seed {}: {}".format(arm, problem_name, seed, repr(error)[-200:]), flush=True)
            continue
        best_found = max(row["ipc"] for row in run["history"])
        del run["full_history"]
        report["test"][problem_name]["arms"][arm][str(seed)] = run
        print("== {} / {} seed {}: best {:.4f}".format(arm, problem_name, seed, best_found), flush=True)
        with open(output_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
    jobs.shutdown()

    report["wall_seconds"] = time.time() - started
    print("failed jobs:", len(report["failed_jobs"]), flush=True)
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=2)
    return report


if __name__ == "__main__":
    import sys
    stamp = time.strftime("%Y%m%d_%H%M")
    run_experiment(train_socs=["A_mobile", "B_midrange"], test_soc="C_server",
                   traces=sys.argv[1:], rounds=12, per_round=2, seeds=3,
                   output_path="results/experiment_{}.json".format(stamp), suite=True)
