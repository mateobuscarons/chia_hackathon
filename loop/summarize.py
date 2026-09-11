"""Tables from an experiment report, finished or still running.

  python -m loop.summarize results/experiment_X.json [more reports] [--reference 0.7491 | --uniform 300]
  python -m loop.summarize progress <run.log> [--start 0.1656]   # a cell still running, read from its log

Per test problem: designs each arm needed to reach 90 / 95 / 99% of the
reference (per seed, censored median), how many seeds got there, final best,
area under the best-so-far curve; then the calibration scoreboard per forecaster
class on the shared dispute bets. The reference is each cell's fixed best known
design (pass it with --reference), or the best of the cell's uniform random
sample of N designs (--uniform N, read from the cell's tables); without either the
best design any arm found in the report is used and the table says so. Arms and seeds still running are
simply the ones not yet in the report.
"""

import json
import sys

TARGET_FRACTIONS = [0.9, 0.95, 0.99]


AGENT_ARMS = ["llm_direct", "memory"]


def forecaster_class(forecaster_id):
    if forecaster_id == "SURROGATE":
        return "surrogate"
    if forecaster_id.startswith("RULE"):
        return "rules"
    if forecaster_id.startswith("FACT"):
        return "facts"
    if "-REPLY-" in forecaster_id:
        return "analyst"
    if "-PICK-" in forecaster_id:
        return forecaster_id.split("-")[0]      # the agent arm's name opens its pick ids
    return "other"


def best_so_far(history, objective="ipc"):
    curve = []
    best = None
    for row in history:
        if best is None or row[objective] > best:
            best = row[objective]
        curve.append(best)
    return curve


def best_found(trace_report):
    """The best IPC any arm found in this report (fallback reference)."""
    best = None
    for arm in trace_report["arms"]:
        for run in trace_report["arms"][arm].values():
            for row in run["history"]:
                if best is None or row["ipc"] > best:
                    best = row["ipc"]
    return best


def designs_to_target(history, reference, fraction):
    """Designs until best-so-far captures `fraction` of the gap between the
    baseline (history[0]) and the reference; None if never. Index 0 is the
    free baseline run, so the count is the number of designs bought."""
    baseline = history[0]["ipc"]
    if reference <= baseline:
        # No headroom: the start already beats the reference, so "designs to
        # target" means nothing for this cell.
        return None
    target = baseline + fraction * (reference - baseline)
    curve = best_so_far(history)
    for index, value in enumerate(curve):
        if value >= target:
            return index
    return None


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def area_under_curve(history, objective="ipc"):
    """Mean best-so-far gain over the baseline across the run (reference-free;
    rewards getting good early)."""
    baseline = history[0][objective]
    curve = best_so_far(history, objective)
    if len(curve) <= 1:
        return 0.0
    total = 0.0
    for value in curve[1:]:
        total += (value / baseline - 1.0)
    return 100.0 * total / (len(curve) - 1)


def all_bets(report):
    """Every settled bet in the report. Kinds: sign / half / full (the three fixed
    questions about a claim, answered by rules, surrogate and replies alike),
    forecast (an agent's own within-5% forecast). Levels: suite / workload."""
    bets = list(report.get("learn_bets", []))
    for problem_name in report["test"]:
        for arm in report["test"][problem_name]["arms"]:
            for seed in report["test"][problem_name]["arms"][arm]:
                bets += report["test"][problem_name]["arms"][arm][seed]["bets"]
    settled = []
    for bet in bets:
        if bet["outcome"] is not None:
            settled.append(bet)
    return settled


def brier_line(label, class_bets):
    brier = 0.0
    wins = 0
    for bet in class_bets:
        outcome = 1.0 if bet["outcome"] else 0.0
        brier += (bet["probability"] - outcome) ** 2
        if (bet["probability"] >= 0.5) == bet["outcome"]:
            wins += 1
    return "   {:<28} bets={:>4} brier={:.3f} win-rate={:.2f}".format(
        label, len(class_bets), brier / len(class_bets), wins / len(class_bets))


def summarize(report, reference=None):
    for problem_name in report["test"]:
        trace_report = report["test"][problem_name]
        cell_reference = reference
        reference_note = "fixed reference"
        if cell_reference is None:
            cell_reference = best_found(trace_report)
            reference_note = "best design any arm found in this report (pass --reference for the fixed one)"
        if cell_reference is None:
            print("== {} | no finished run yet".format(problem_name))
            continue
        print("== {} | reference {:.4f} ({})".format(problem_name, cell_reference, reference_note))
        print("   {:<17} {:>5} {:>22} {:>22} {:>22} {:>10} {:>7}".format(
            "arm", "seeds", "to 90% (median, hit)", "to 95% (median, hit)", "to 99% (median, hit)", "final best", "auc %"))
        for arm in trace_report["arms"]:
            runs = trace_report["arms"][arm]
            if len(runs) == 0:
                print("   {:<11} {:>5}  (not finished yet)".format(arm, 0))
                continue
            columns = []
            for fraction in TARGET_FRACTIONS:
                counts = []
                hits = 0
                for seed in runs:
                    history = runs[seed]["history"]
                    count = designs_to_target(history, cell_reference, fraction)
                    if count is None:
                        count = len(history)      # censored at the budget
                    else:
                        hits += 1
                    counts.append(count)
                shown = "{:g}".format(median(counts))
                if hits < len(counts):
                    shown = ">" + shown if hits == 0 else shown
                columns.append("{:>14} {:>3}/{}".format(shown, hits, len(counts)))
            bests = []
            aucs = []
            for seed in runs:
                history = runs[seed]["history"]
                bests.append(best_so_far(history)[-1])
                aucs.append(area_under_curve(history))
            print("   {:<17} {:>5} {} {} {} {:>10.4f} {:>7.2f}".format(
                arm, len(runs), columns[0], columns[1], columns[2], sum(bests) / len(bests), sum(aucs) / len(aucs)))

    print("== calibration: the same three questions per claim (sign / half / full), suite level, per forecaster")
    bets = all_bets(report)
    for klass in ["surrogate", "rules", "analyst"]:
        for kind in ["sign", "half", "full"]:
            class_bets = []
            for bet in bets:
                if forecaster_class(bet["forecaster"]) == klass and bet["kind"] == kind and bet.get("level", "suite") == "suite":
                    class_bets.append(bet)
            if len(class_bets) > 0:
                print(brier_line("{} / {}".format(klass, kind), class_bets))
    print("== calibration: rules and remembered facts per workload (sign of the measured pair)")
    for klass in ["rules", "facts"]:
        for kind in ["sign", "half", "full"]:
            class_bets = []
            for bet in bets:
                if forecaster_class(bet["forecaster"]) == klass and bet["kind"] == kind and bet.get("level") == "workload":
                    class_bets.append(bet)
            if len(class_bets) > 0:
                print(brier_line("{} / {} / workload".format(klass, kind), class_bets))
    print("== calibration: agents on their own forecasts (measured within 5% of the prediction or above)")
    for klass in ["analyst"] + AGENT_ARMS:
        class_bets = []
        for bet in bets:
            if forecaster_class(bet["forecaster"]) == klass and bet["kind"] == "forecast":
                class_bets.append(bet)
        if len(class_bets) > 0:
            print(brier_line(klass, class_bets))

    print("== playbook: {} rules, {} rejected, {} failed jobs".format(
        len(report.get("rules", [])), len(report.get("rejected_rules", [])), len(report.get("failed_jobs", []))))
    for rule in report.get("rules", []):
        verification = rule.get("verification") or {}
        print("   {} {}={} measured {:+.1f}% (LLM said {:+.1f}%, {} pair(s)) | {}".format(
            rule["id"], rule["claim"]["knob"], rule["claim"]["value"], rule["claim"]["gain_pct"],
            verification.get("llm_gain_pct", 0.0), verification.get("pairs", 0), rule["text"][:90]))


def uniform_reference(report, how_many):
    """The best suite geomean among the cell's uniform random sample of designs
    (loop.collect uniform, same seed), over those measured on every workload."""
    import math
    from loop import champsim_problem, collect
    settings = report["settings"]
    traces = settings["traces"][0]
    if not isinstance(traces, list):
        traces = settings["traces"]
    soc_name = settings["test_soc"]
    tables = []
    for trace_path in traces:
        tables.append(champsim_problem.load_table(champsim_problem.table_path(soc_name, trace_path)))
    best = None
    counted = 0
    for knobs in collect.uniform_designs(soc_name, how_many)[1:]:
        name = champsim_problem.config_name(knobs, soc_name)
        log_sum = 0.0
        complete = True
        for table in tables:
            entry = table.get(name)
            if entry is None or entry["metrics"] is None:
                complete = False
                break
            log_sum += math.log(max(entry["metrics"]["ipc"], 1e-9))
        if not complete:
            continue
        counted += 1
        value = math.exp(log_sum / len(tables))
        if best is None or value > best:
            best = value
    print("== uniform reference: best of {} measured designs out of {} sampled = {}".format(
        counted, how_many, "{:.4f}".format(best) if best is not None else "none yet"))
    return best


def load_merged(paths):
    """Several reports of the same experiment (different seeds) read as one."""
    merged = None
    for path in paths:
        with open(path) as report_file:
            report = json.load(report_file)
        if merged is None:
            merged = report
            continue
        for problem_name in report["test"]:
            merged["test"].setdefault(problem_name, {"arms": {}})
            for arm in report["test"][problem_name]["arms"]:
                merged["test"][problem_name]["arms"].setdefault(arm, {}).update(
                    report["test"][problem_name]["arms"][arm])
    return merged


def progress(log_path, start=None):
    """Best-so-far per arm and seed while a cell is still running. Every arm prints
    one line per design ("[arm-problem-sN] round R | design | ipc=V | label"), so the
    run log is a complete record long before the report is written."""
    runs = {}
    with open(log_path) as log_file:
        for line in log_file:
            # Ray prefixes each line with its worker ("(run_arm_job pid=123) [tag] round ..."),
            # so the tag is the last bracket before "] round ", not the start of the line.
            if "] round " not in line or "ipc=" not in line:
                continue
            head = line[:line.index("] round ")]
            if "[" not in head:
                continue
            tag = head[head.rindex("[") + 1:]
            try:
                round_number = int(line.split("] round ")[1].split(" |")[0])
                value = float(line.split("ipc=")[1].split(" |")[0].strip())
            except (IndexError, ValueError):
                continue
            label = line.rstrip("\n").split("| ")[-1]
            run = runs.setdefault(tag, {"designs": 0, "best": None, "round": 0, "opening": None})
            run["designs"] += 1
            run["round"] = max(run["round"], round_number)
            if run["best"] is None or value > run["best"]:
                run["best"] = value
                run["best_at"] = run["designs"]
            if run["opening"] is None:
                run["opening"] = label
    if len(runs) == 0:
        print("no round lines in " + log_path)
        return
    # The start design is design 0 and never prints a round line, so without --start
    # the "best" column is the best design an arm BOUGHT, which is what matters anyway.
    if start is not None:
        print("start design: %.4f\n" % start)
    print("%-12s %-5s %7s %7s %9s %9s %9s  %s" % (
        "arm", "seed", "round", "designs", "best", "vs start", "found at", "first pick"))
    by_arm = {}
    for tag in sorted(runs):
        arm = tag.split("-")[0]
        seed = tag.split("-s")[-1]
        run = runs[tag]
        by_arm.setdefault(arm, []).append(run["best"])
        if start is None:
            gain = "        -"
        else:
            gain = "%8.1f%%" % (100.0 * (run["best"] / start - 1.0))
        print("%-12s %-5s %7d %7d %9.4f %9s %9d  %s" % (
            arm, seed, run["round"], run["designs"], run["best"], gain,
            run.get("best_at", 0), run["opening"][:40]))
    print()
    for arm in sorted(by_arm, key=lambda a: -sum(by_arm[a]) / len(by_arm[a])):
        values = by_arm[arm]
        mean = sum(values) / len(values)
        if start is None:
            print("  %-12s mean best %.4f over %d run(s)" % (arm, mean, len(values)))
        else:
            print("  %-12s mean best %.4f (%+.1f%% on the start) over %d run(s)" % (
                arm, mean, 100.0 * (mean / start - 1.0), len(values)))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) > 0 and arguments[0] == "progress":
        start_value = None
        if "--start" in arguments:
            start_value = float(arguments[arguments.index("--start") + 1])
        progress(arguments[1], start_value)
        raise SystemExit(0)
    reference = None
    uniform_count = None
    if "--reference" in arguments:
        position = arguments.index("--reference")
        reference = float(arguments[position + 1])
        arguments = arguments[:position] + arguments[position + 2:]
    if "--uniform" in arguments:
        position = arguments.index("--uniform")
        uniform_count = int(arguments[position + 1])
        arguments = arguments[:position] + arguments[position + 2:]
    report = load_merged(arguments)
    if uniform_count is not None:
        reference = uniform_reference(report, uniform_count)
    summarize(report, reference)
