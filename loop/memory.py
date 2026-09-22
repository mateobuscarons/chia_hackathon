"""The loop's memory: what past searches measured, as situations and moves.

A record is one measured single-knob change on some chip - the state that chip was in
before the move, the knob that moved, and what happened to speed, silicon and power.
Never a whole design: the loop recognises a situation and recalls a move, it does not
copy an answer.

A situation is written in fractions only - shares of the memory time, how full the
channels ran, how much of the binding budget was spent. An amount ("2 MB at the last
level") means nothing on another chip; a share means the same thing on every chip, and
it names no chip, so the prompts' standing rule is untouched.

The corpus is built once per chip, offline: the wait shares need that chip's own cache
latencies and `loop.chip` is bound to one chip per process.
"""

import glob
import json
import math
import os

from loop import chip
from loop import council
from loop import space
from loop import suite

MEMORY_DIR = "results/memory"
KNOBS = list(space.SEARCH_SPACE)


def situation(knobs, metrics, workloads):
    """The state a chip was in, as fractions: where the memory time sat, how full the
    channels ran, and how much of the binding budget the design spent. None when the
    counters this needs are missing from the row."""
    entry = {"knobs": knobs, "metrics": metrics}
    shares = {"L1D": 0.0, "L2C": 0.0, "LLC": 0.0, "DRAM": 0.0}
    fill = 0.0
    for workload in workloads:
        tiers = council.memory_time(entry, workload)
        if tiers is None:
            return None
        bytes_per_cycle = council.off_chip_bytes_per_cycle(entry, workload)
        if bytes_per_cycle is None:
            return None
        for tier in shares:
            shares[tier] += tiers[tier] / len(workloads)
        peak = chip.card()["dram_peak_bytes_per_cycle"]
        fill += bytes_per_cycle / peak / len(workloads)
    rounded = {}
    for tier in shares:
        rounded[tier] = round(shares[tier], 3)
    return {"wait": rounded, "fill": round(fill, 3),
            "mm2": round(space.silicon_mm2(knobs), 3),
            "watts": round(space.watts(knobs, metrics, workloads), 3)}


def measured(workloads):
    """Every design this chip has measured on all of `workloads`, keyed by name: its
    knobs, its situation, and what it scored - the suite's geometric-mean IPC, the
    silicon it takes and the watts it drew. A row the counters cannot describe, or one
    the simulator never completed, is dropped rather than guessed at."""
    tables = {}
    for workload in workloads:
        tables[workload] = suite.load_table(suite.table_path(workload))
    designs = {}
    for name in sorted(tables[workloads[0]]):
        if not name.startswith(chip.NAME + "_"):
            continue
        metrics = {}
        log_sum = 0.0
        for workload in workloads:
            row = tables[workload].get(name)
            if row is None or row["metrics"] is None:
                metrics = None
                break
            for key in row["metrics"]:
                metrics["{}:{}".format(workload, key)] = row["metrics"][key]
            log_sum += math.log(max(row["metrics"]["ipc"], 1e-9))
        if metrics is None:
            continue
        knobs = tables[workloads[0]][name]["knobs"]
        state = situation(knobs, metrics, workloads)
        if state is None:
            continue
        designs[name] = {"knobs": knobs, "situation": state,
                         "ipc": math.exp(log_sum / len(workloads)),
                         "mm2": space.silicon_mm2(knobs),
                         "watts": space.watts(knobs, metrics, workloads)}
    return designs


def cases(report_path, workloads):
    """Every experiment a council run actually ran, read from its report: what the chip
    looked like before the move (the observation), which knobs the specialist moved (the
    experiment), what that measured (the result) and the specialist's own words for why
    (the reasoning). Every sketch carries the concern that proposed it, so the reasoning
    is the one written for that move and not a summary.

    Cost is kept ABSOLUTE - the silicon and the watts, before and after. A share of the
    budget is wrong the moment the budget moves: the same design sits at 77 % of a 4.0
    mm2 cap and 123 % of a 2.5 mm2 one. The share is computed when a record is read,
    against whatever cap is in force then."""
    report = json.load(open(report_path))
    designs = measured(workloads)
    found = []
    for arm in report["runs"]:
        for seed in report["runs"][arm]:
            run = report["runs"][arm][seed]
            by_index = {}
            for design in run["designs"]:
                by_index[design["index"]] = design
            for record in run["rounds"]:
                parent = by_index.get(record.get("against"))
                if parent is None or parent["name"] not in designs:
                    continue
                state = designs[parent["name"]]["situation"]
                proposals = record.get("proposals") or {}
                # A sweep entry's "reasoning" is a label, not a rationale, and a round
                # that recorded one text for two concerns cannot say which move it
                # explains. Both keep the experiment and lose the sentence: a measured
                # move with no reason is honest, a move with someone else's is not.
                shared = set()
                seen = set()
                for name in proposals:
                    text = proposals[name]["reasoning"]
                    if text in seen:
                        shared.add(text)
                    seen.add(text)
                for design in run["designs"]:
                    if design["round"] != record["round"]:
                        continue
                    if not design["source"].startswith("sketch:"):
                        continue
                    concern = design["source"].split(":")[1]
                    if concern not in proposals:
                        continue
                    moved = (record.get("moves") or {}).get(concern) or {}
                    if not moved:
                        continue
                    reasoning = proposals[concern]["reasoning"]
                    if reasoning in shared or reasoning.startswith("one-knob sweep"):
                        reasoning = None
                    found.append({"wait": state["wait"], "fill": state["fill"],
                                  "mm2": round(parent["mm2"], 3), "watts": round(parent["watts"], 3),
                                  "bottleneck": (record.get("sheet") or {}).get("bottleneck"),
                                  "experiment": moved,
                                  "gain": round((design["ipc"] / parent["ipc"] - 1.0) * 100.0, 2),
                                  "to_mm2": round(design["mm2"], 3),
                                  "to_watts": round(design["watts"], 3),
                                  "refused": design["violations"],
                                  "reasoning": reasoning})
    return found


def build(workloads, report_paths):
    """This chip's corpus, written to results/memory/<chip>.json for the loop to read."""
    found = []
    under = None
    for path in report_paths:
        found += cases(path, workloads)
        card = json.load(open(path))["card"]
        under = [card["area_budget_mm2"], card["power_budget_w"]]
    os.makedirs(MEMORY_DIR, exist_ok=True)
    out = os.path.join(MEMORY_DIR, chip.NAME + ".json")
    with open(out, "w") as handle:
        json.dump({"chip": chip.NAME, "measured_under": under, "records": found}, handle)
    return out, len(found)


def caps():
    """The caps in force right now, from the chip the run is tuning."""
    card = chip.card()
    return card["area_budget_mm2"], card["power_budget_w"]


def spent(mm2, watts, area_cap, power_cap):
    """How much of the NEAREST cap a design spends. Both caps in one fraction: what binds
    is whichever is closer, and on this chip that is usually the power cap, not the
    silicon. Computed when a record is read, never stored - a design at 77 % of a 4.0 mm2
    cap is at 123 % of a 2.5 mm2 one, and the record must read correctly under both."""
    worst = 0.0
    if area_cap:
        worst = max(worst, mm2 / area_cap)
    if power_cap:
        worst = max(worst, watts / power_cap)
    return worst


def box(state, area_cap, power_cap):
    """The two questions that file a situation: how much of the memory time is spent off
    chip, and whether the design has room left under the caps in force. Channel fill is
    NOT a question here - it tracks which chip a record came from, so filing by it would
    file by chip. It ranks inside the box instead."""
    offchip = state["wait"]["DRAM"]
    if offchip < 0.25:
        where = "on-chip"
    elif offchip < 0.45:
        where = "mixed"
    else:
        where = "off-chip"
    room = spent(state["mm2"], state["watts"], area_cap, power_cap)
    return (where, "tight" if room >= 0.75 else "room")


def distance(state, record):
    """How far one situation is from another inside the same box: the four wait shares
    and how full the channels ran, all fractions, so no term needs a weight."""
    total = 0.0
    for tier in state["wait"]:
        total += (state["wait"][tier] - record["wait"][tier]) ** 2
    total += (state["fill"] - record["fill"]) ** 2
    return total ** 0.5


CORPUS = {}


def corpus(sources=None):
    """The records to read from: every chip's file under results/memory, or only the
    chips named. Which past runs a loop learns from is the loop's policy, not this
    file's."""
    key = tuple(sources) if sources else "all"
    if key not in CORPUS:
        loaded = []
        for path in sorted(glob.glob(os.path.join(MEMORY_DIR, "*.json"))):
            blob = json.load(open(path))
            if sources and blob["chip"] not in sources:
                continue
            for record in blob["records"]:
                record["chip"] = blob["chip"]
                loaded.append(record)
        CORPUS[key] = loaded
    return CORPUS[key]


MOVED_AT_ALL = float(os.environ.get("MEMORY_MIN_GAIN", "1.0"))   # percent of speed


def fetch(state, knobs, knob_names, gains=3, losses=2, sources=None):
    """Up to `how_many` past experiments run in a situation like this one: same box under
    the caps in force NOW, nearest inside it, and still worth offering - the move is
    buildable from `knobs`, what it produced fits today's caps, and it changed the speed
    at all. One per set of knobs moved, so the list is that many distinct suggestions.

    Split between moves that gained and moves that lost, because ranking by closeness
    alone fills every slot with losses - most records share the stock as their parent and
    most experiments fail. A move that cost 12 % is worth a slot; three of them are not.
    Empty when nothing past fits: silence is the honest answer."""
    area_cap, power_cap = caps()
    here = box(state, area_cap, power_cap)
    scored = []
    for record in corpus(sources):
        moved = record["experiment"]
        theirs = True
        for knob in moved:
            if knob not in knob_names:
                theirs = False
        if not theirs:
            continue
        if abs(record["gain"]) < MOVED_AT_ALL:
            continue
        if box(record, area_cap, power_cap) != here:
            continue
        if spent(record["to_mm2"], record["to_watts"], area_cap, power_cap) > 1.0:
            continue
        candidate = dict(knobs)
        for knob in moved:
            candidate[knob] = moved[knob][1]
        if not space.in_space(candidate):
            continue
        if not space.within_budget(space.typed_knobs(candidate)):
            continue
        scored.append((distance(state, record), record))
    scored.sort(key=lambda pair: pair[0])
    chosen = []
    taken = set()
    for wanted, limit in [(True, gains), (False, losses)]:
        picked = 0
        for _, record in scored:
            if (record["gain"] > 0) != wanted:
                continue
            key = tuple(sorted(record["experiment"]))
            if key in taken:
                continue
            taken.add(key)
            chosen.append(record)
            picked += 1
            if picked == limit:
                break
    return chosen


def lines(found):
    """The block a prompt carries: per experiment, the situation it was run in, the knobs
    it moved, what it measured, and - only where the rationale recorded belongs to that
    move - the sentence written for it at the time."""
    out = []
    heading = None
    for record in found:
        here = "  with the wait at L1D {L1D:.0%} / L2 {L2C:.0%} / LLC {LLC:.0%} / off chip {DRAM:.0%}".format(
            **record["wait"]) + ", channels {:.0%} full, {:.2f} mm2 and {:.2f} W spent:".format(
            record["fill"], record["mm2"], record["watts"])
        if here != heading:
            out.append(here)
            heading = here
        moved = []
        for knob in sorted(record["experiment"]):
            pair = record["experiment"][knob]
            moved.append("{} {} -> {}".format(knob, pair[0], pair[1]))
        out.append("    {}  ->  {:+.1f}% speed, {:.2f} mm2, {:.2f} W".format(
            "; ".join(moved), record["gain"], record["to_mm2"], record["to_watts"]))
        if record["reasoning"]:
            out.append("    the reason given at the time: " + record["reasoning"])
    return out
