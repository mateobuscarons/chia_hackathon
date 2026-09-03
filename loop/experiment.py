"""The transfer experiment: learn a playbook on SoCs A and B, test it on unseen SoC C.

Arms on the target SoC (same simulation cap for all):
  random          - shuffle the candidates, run them in order
  surrogate       - GP search from scratch (cold start)
  surrogate_pooled- GP warm-started with every A/B run (statistical transfer)
  textbook        - GP + rules the LLM wrote BEFORE seeing any result
  rules           - GP + rules distilled from A/B (rule transfer, no LLM in the loop)
  analyst         - GP + LLM hypotheses, no rules (LLM cold start)
  full            - GP + distilled rules + LLM hypotheses (the whole loop)
The metric is simulations-to-target: how many runs until best-so-far
captures 90% of the gap between baseline and the in-budget optimum (dense sweep).
"""

import copy
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor

from loop import analyst, loop, playbook, surrogate_gp
from loop.champsim_problem import make_problem

TARGET_FRACTION = 0.9
PARALLEL_RUNS = 5          # parallel arm runs (each simulates 2 configs at a time on 10 cores)


def learn(store, soc_names, traces, rounds, per_round, space_name):
    """Run the full loop on every training (SoC, trace) in parallel, distill rules
    from each, and merge the playbooks into `store`. Returns the pooled history."""
    pool = ProcessPoolExecutor(max_workers=PARALLEL_RUNS)
    futures = []
    for soc_name in soc_names:
        for trace_path in traces:
            futures.append((soc_name, pool.submit(learn_job, soc_name, trace_path, rounds, per_round, space_name)))
    prior_history = []
    for soc_name, future in futures:
        job_store, history = future.result()
        merge_playbook(store, job_store)
        for entry in history:
            prior_history.append(with_soc(entry, soc_name))
    pool.shutdown()
    return prior_history


def learn_job(soc_name, trace_path, rounds, per_round, space_name):
    """One training problem: loop with analyst + its own fresh playbook, then distill."""
    problem = make_problem(soc_name, trace_path, space_name=space_name)
    job_store = {"rules": [], "bets": []}
    tag = "learn-" + problem["name"]
    result = loop.run_loop(problem, rounds, per_round, job_store, surrogate_gp,
                           use_rules=True, use_analyst=True, tag=tag)
    distill(job_store, problem, result["history"], tag)
    return job_store, result["history"]


def training_runs(soc_names, traces, space_name):
    """Every simulated config of the training SoCs, tagged with its SoC (pooled prior)."""
    runs = []
    for soc_name in soc_names:
        for trace_path in traces:
            problem = make_problem(soc_name, trace_path, allow_simulation=False, space_name=space_name)
            table = problem["holder"].sweep_table
            for name in table:
                if table[name]["metrics"] is not None:
                    runs.append(with_soc({"name": name, "knobs": table[name]["knobs"],
                                          "metrics": table[name]["metrics"]}, soc_name))
    return runs


def merge_playbook(store, job_store):
    """Append another playbook's rules and bets, giving rules fresh ids."""
    new_ids = {}
    for rule in job_store["rules"]:
        new_id = playbook.add_rule(store, rule["condition"], rule["claim"], rule["example"], rule["text"])
        new_ids[rule["id"]] = new_id
        merged = store["rules"][-1]
        for field in ["wins", "losses", "brier_scores", "origin", "status"]:
            merged[field] = rule[field]
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
    new_rules = analyst.distill_rules(problem["search_space"], table,
                                      "\n".join(ledger_lines[-60:]),
                                      loop.format_rules(store["rules"]))
    for rule in new_rules:
        if not well_formed(rule, problem):
            print("[{}] skipped malformed rule: {}".format(tag, json.dumps(rule)[:160]), flush=True)
            continue
        rule_id = playbook.add_rule(store, rule["condition"], rule["claim"],
                                    rule["example"], rule["text"])
        print("[{}] distilled {}: {}".format(tag, rule_id, rule["text"]), flush=True)


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
    history, arm_store = run_arm(arm, store, soc_name, trace_path, rounds, per_round, prior_history, seed, space_name)
    return {"history": compact(history, "ipc"), "full_history": history,
            "bets": arm_store["bets"], "rules": arm_store["rules"]}


def run_arm(arm, store, soc_name, trace_path, rounds, per_round, prior_history, seed, space_name):
    problem = make_problem(soc_name, trace_path, space_name=space_name)
    arm_store = copy.deepcopy(store)          # every arm starts from the same playbook
    if arm == "random":
        return random_arm(problem, rounds * per_round, seed), arm_store
    if arm == "textbook":
        arm_store = {"rules": [], "bets": []}
        for rule in analyst.textbook_rules(problem["search_space"], problem["objective"], 6):
            if well_formed(rule, problem):
                playbook.add_rule(arm_store, rule["condition"], rule["claim"], rule["example"], rule["text"])
    prior = []
    if arm == "surrogate_pooled":
        prior = prior_history
        problem["search_space"] = dict(problem["search_space"], soc=["A_mobile", "B_midrange", "C_server"])
        for name in problem["candidates"]:
            problem["candidates"][name] = dict(problem["candidates"][name], soc=soc_name)
        problem["baseline"] = dict(problem["baseline"], soc=soc_name)
    use_rules = arm in ["textbook", "rules", "full"]
    use_analyst = arm in ["analyst", "full"]
    tag = "{}-{}-s{}".format(arm, problem["name"], seed)
    result = loop.run_loop(problem, rounds, per_round, arm_store, surrogate_gp,
                           use_rules, use_analyst, tag, prior_history=prior, seed=seed)
    return result["history"], arm_store


def simulations_to_target(history, optimum, objective):
    """Runs until best-so-far captures TARGET_FRACTION of the achievable gain.

    The gap is optimum minus the baseline (history[0]). Scoring against the
    gap, not the raw optimum, keeps workloads with little headroom (omnetpp:
    +2%) from being "solved" at step zero.
    """
    baseline = history[0]["metrics"][objective]
    target = baseline + TARGET_FRACTION * (optimum - baseline)
    best = None
    for index, entry in enumerate(history):
        value = entry["metrics"][objective]
        if best is None or value > best:
            best = value
        if best >= target:
            return index          # index 0 is the free baseline run
    return None


def in_budget_optimum(problem):
    """True optimum from the dense sweep. None when the space was never swept
    (Tier B): the report then scores against the best design any arm found."""
    table = problem["holder"].sweep_table
    best = None
    for name in problem["candidates"]:
        if name not in table:
            return None
        value = table[name]["metrics"]["ipc"]
        if best is None or value > best:
            best = value
    return best


ARMS = ["random", "surrogate", "surrogate_pooled", "textbook", "rules", "analyst", "full"]


def compact(history, objective):
    """Only what the plots need: config name and objective per run, in order."""
    rows = []
    for entry in history:
        rows.append({"name": entry["name"], objective: entry["metrics"][objective]})
    return rows


def run_experiment(train_socs, test_soc, traces, rounds, per_round, seeds, output_path,
                   train_traces=None, space_name="A", arms=None, test_rounds=None,
                   playbook_path=None, first_seed=0):
    """train_traces defaults to `traces` (cross-trace transfer passes another list).
    space_name selects Tier A or B; arms defaults to all; test_rounds defaults to rounds.
    playbook_path reuses an already-learned playbook (more seeds, same knowledge);
    the pooled arm then gets every training run in the Tier tables as prior data."""
    if train_traces is None:
        train_traces = traces
    if arms is None:
        arms = ARMS
    if test_rounds is None:
        test_rounds = rounds
    started = time.time()
    if playbook_path is None:
        store = {"rules": [], "bets": []}
        prior_history = learn(store, train_socs, train_traces, rounds, per_round, space_name)
    else:
        store = playbook.load(playbook_path)
        prior_history = training_runs(train_socs, train_traces, space_name)
    playbook.save(store, output_path.replace(".json", "_playbook.json"))

    report = {"settings": {"train_socs": train_socs, "test_soc": test_soc, "traces": traces,
                           "train_traces": train_traces, "space": space_name, "arms": arms,
                           "test_rounds": test_rounds,
                           "rounds": rounds, "per_round": per_round, "seeds": seeds},
              "learn_bets": store["bets"], "rules": store["rules"], "test": {}}

    # Every (trace, arm, seed) run is independent: run them in parallel processes.
    # Each one returns a small dict; the report is saved after every result.
    pool = ProcessPoolExecutor(max_workers=PARALLEL_RUNS)
    futures = []
    for trace_path in traces:
        problem = make_problem(test_soc, trace_path, space_name=space_name)
        report["test"][problem["name"]] = {"optimum": in_budget_optimum(problem), "arms": {}}
        for arm in arms:
            report["test"][problem["name"]]["arms"][arm] = {}
            for seed in range(first_seed, first_seed + seeds):
                future = pool.submit(run_arm_job, arm, store, test_soc, trace_path,
                                     test_rounds, per_round, prior_history, seed, space_name)
                futures.append((problem["name"], arm, seed, future))

    report["failed_jobs"] = []
    for problem_name, arm, seed, future in futures:
        try:
            run = future.result()
        except Exception as error:
            # One bad job must never abort the experiment; it is recorded and visible.
            report["failed_jobs"].append({"problem": problem_name, "arm": arm, "seed": seed,
                                          "error": repr(error)[-400:]})
            print("!! FAILED {} / {} seed {}: {}".format(arm, problem_name, seed, repr(error)[-200:]), flush=True)
            continue
        optimum = report["test"][problem_name]["optimum"]
        run["sims_to_target"] = None
        if optimum is not None:
            run["sims_to_target"] = simulations_to_target(run["full_history"], optimum, "ipc")
        del run["full_history"]
        report["test"][problem_name]["arms"][arm][str(seed)] = run
        print("== {} / {} seed {}: sims_to_target={}".format(
            arm, problem_name, seed, run["sims_to_target"]), flush=True)
        with open(output_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
    pool.shutdown()

    report["wall_seconds"] = time.time() - started
    print("failed jobs:", len(report["failed_jobs"]), flush=True)
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=2)
    return report


if __name__ == "__main__":
    import sys
    stamp = time.strftime("%Y%m%d_%H%M")
    run_experiment(train_socs=["A_mobile", "B_midrange"], test_soc="C_server",
                   traces=sys.argv[1:], rounds=10, per_round=2, seeds=3,
                   output_path="results/experiment_{}.json".format(stamp))
