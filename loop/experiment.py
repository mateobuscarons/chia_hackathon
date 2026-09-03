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

from loop import analyst, loop, playbook, surrogate_gp
from loop.champsim_problem import make_problem

TARGET_FRACTION = 0.9


def learn(store, soc_names, traces, rounds, per_round):
    """Run the full loop on the training SoCs, then distill rules after each."""
    prior_history = []
    for soc_name in soc_names:
        for trace_path in traces:
            problem = make_problem(soc_name, trace_path)
            tag = "learn-" + problem["name"]
            result = loop.run_loop(problem, rounds, per_round, store, surrogate_gp,
                                   use_rules=True, use_analyst=True, tag=tag)
            distill(store, problem, result["history"], tag)
            for entry in result["history"]:
                prior_history.append(with_soc(entry, soc_name))
    return prior_history


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
    """The analyst must name a real metric, a real knob and one of its allowed values."""
    try:
        metric_ok = rule["condition"]["metric"] in problem["table_metrics"]
        float(rule["condition"]["value"])
        knob = rule["claim"]["knob"]
        allowed = [str(value) for value in problem["search_space"][knob]]
        value_ok = str(rule["claim"]["value"]) in allowed
        float(rule["claim"]["gain_pct"])
        return metric_ok and value_ok and isinstance(rule.get("text"), str)
    except (KeyError, TypeError, ValueError):
        return False


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


def run_arm(arm, store, soc_name, trace_path, rounds, per_round, prior_history, seed):
    problem = make_problem(soc_name, trace_path)
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
                           use_rules, use_analyst, tag, prior_history=prior)
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
    best = None
    for name in problem["candidates"]:
        value = problem["evaluate"](problem["candidates"][name])["ipc"]
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
                   train_traces=None):
    """train_traces defaults to `traces`; pass a different list for cross-trace transfer."""
    if train_traces is None:
        train_traces = traces
    started = time.time()
    store = {"rules": [], "bets": []}
    prior_history = learn(store, train_socs, train_traces, rounds, per_round)
    playbook.save(store, output_path.replace(".json", "_playbook.json"))

    report = {"settings": {"train_socs": train_socs, "test_soc": test_soc, "traces": traces,
                           "train_traces": train_traces,
                           "rounds": rounds, "per_round": per_round, "seeds": seeds},
              "learn_bets": store["bets"], "rules": store["rules"], "test": {}}

    for trace_path in traces:
        problem = make_problem(test_soc, trace_path)
        optimum = in_budget_optimum(problem)
        trace_report = {"optimum": optimum, "arms": {}}
        for arm in ARMS:
            trace_report["arms"][arm] = {}
            for seed in range(seeds):
                random.seed(seed)
                history, arm_store = run_arm(arm, store, test_soc, trace_path,
                                             rounds, per_round, prior_history, seed)
                trace_report["arms"][arm][str(seed)] = {
                    "history": compact(history, problem["objective"]),
                    "sims_to_target": simulations_to_target(history, optimum, problem["objective"]),
                    "bets": arm_store["bets"],
                    "rules": arm_store["rules"],
                }
                print("== {} / {} seed {}: sims_to_target={}".format(
                    arm, problem["name"], seed,
                    trace_report["arms"][arm][str(seed)]["sims_to_target"]), flush=True)
                # Save after every arm so a crash or timeout loses nothing.
                report["test"][problem["name"]] = trace_report
                with open(output_path, "w") as report_file:
                    json.dump(report, report_file, indent=2)

    report["wall_seconds"] = time.time() - started
    with open(output_path, "w") as report_file:
        json.dump(report, report_file, indent=2)
    return report


if __name__ == "__main__":
    import sys
    stamp = time.strftime("%Y%m%d_%H%M")
    run_experiment(train_socs=["A_mobile", "B_midrange"], test_soc="C_server",
                   traces=sys.argv[1:], rounds=10, per_round=2, seeds=3,
                   output_path="results/experiment_{}.json".format(stamp))
