"""The loop itself: seed runs -> hypotheses bet -> simulate -> settle -> repeat.

This file only wires the other modules together; it owns no cleverness.
Part A (this chunk): the three helpers. Part B: the round loop.
"""

from loop.configs import make_config, config_name
from loop.simulate import build_binary, run_simulation
from loop.analyst import propose_hypotheses
from loop import playbook

CHAMPSIM_ROOT = "champsim"
BASE_CONFIG = "champsim/champsim_config.json"
GENERATED_DIR = "configs/generated"
PLAYBOOK_PATH = "loop/playbook.json"

# Short runs for the MVP (~1.5 min each); scale up for the real study.
WARMUP_INSTRUCTIONS = 5_000_000
SIMULATION_INSTRUCTIONS = 10_000_000


def run_experiment(knobs, trace_path):
    """Config -> binary -> simulation. Returns the metrics dict."""
    config_path = make_config(knobs, BASE_CONFIG, GENERATED_DIR)
    binary_path = build_binary(config_path, CHAMPSIM_ROOT)
    metrics = run_simulation(
        binary_path, trace_path, WARMUP_INSTRUCTIONS, SIMULATION_INSTRUCTIONS,
    )
    return metrics


def format_table(history):
    """Render experiment history as the compact table the analyst reads."""
    lines = []
    for entry in history:
        knobs = entry["knobs"]
        metrics = entry["metrics"]
        line = "config: {} | ipc={:.4f} | L2_mpki={:.1f} | LLC_mpki={:.1f}".format(
            config_name(knobs), metrics["ipc"],
            metrics["L2C_mpki"], metrics["LLC_mpki"],
        )
        lines.append(line)
    return "\n".join(lines)


def find_previous(history, name):
    """Reuse a result if this exact config was already simulated."""
    for entry in history:
        if config_name(entry["knobs"]) == name:
            return entry
    return None


def run_loop(rounds, trace_path, hypotheses_per_round=3):
    """Run the full cycle for N rounds. Every bet lands in the playbook file."""
    store = playbook.load(PLAYBOOK_PATH)
    history = []

    # Seed: the untouched SoC config grounds the first hypotheses.
    baseline = {"l2_sets": 1024, "llc_sets": 2048,
                "l2_prefetcher": "no", "llc_replacement": "lru"}
    history.append({"knobs": baseline, "metrics": run_experiment(baseline, trace_path)})
    print("seed:", format_table(history))

    for round_number in range(1, rounds + 1):
        hypotheses = propose_hypotheses(format_table(history), hypotheses_per_round)
        for index, hypothesis in enumerate(hypotheses):
            experiment = config_name(hypothesis["knobs"])
            forecaster = "HYP-r{}-{}".format(round_number, index + 1)

            # The bet is registered BEFORE the simulation runs.
            bet_id = playbook.place_bet(store, forecaster, experiment,
                                        hypothesis["event"], hypothesis["probability"])

            previous = find_previous(history, experiment)
            if previous is None:
                metrics = run_experiment(hypothesis["knobs"], trace_path)
                history.append({"knobs": hypothesis["knobs"], "metrics": metrics})
            else:
                metrics = previous["metrics"]

            happened = check_event(hypothesis["event"], metrics)
            playbook.settle_bet(store, bet_id, happened)
            print("round {} | {} | bet '{}' p={} -> {} | ipc={:.4f}".format(
                round_number, hypothesis["hypothesis"][:60], hypothesis["event"],
                hypothesis["probability"], "WON" if happened == (hypothesis["probability"] >= 0.5) else "LOST",
                metrics["ipc"]))
        playbook.save(store, PLAYBOOK_PATH)
    return history


def check_event(event, metrics):
    """Settle a bet objectively. Events look like 'ipc >= 0.35' or 'ipc < 0.3'."""
    parts = event.split()
    metric_name = parts[0]
    operator = parts[1]
    threshold = float(parts[2])
    value = metrics[metric_name]
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    raise ValueError("unsupported event format: " + event)
