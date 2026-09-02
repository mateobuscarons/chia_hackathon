"""CHIA nodes: the loop's simulator and LLM calls as schedulable @ChiaFunction tasks.

  build_from_config  - compile ChampSim for an arbitrary config JSON. Fills the
                       gap in chia.simulators.champsim, whose build node only
                       accepts a prefetcher module (upstream PR candidate).
  simulate           - CHIA's own ChampSimNode.run_champsim, flattened to the
                       metrics dict the loop uses.
  AnalystNode.ask    - Gemini through chia.models.vertex, JSON in / JSON out.

Resources: "champsim_build" is 1 per checkout (builds share the tree and
cannot overlap); "champsim" is one slot per simulation core.
"""

import json
import os
import tempfile

from chia.base.ChiaFunction import ChiaFunction
from chia.models.vertex import VertexGeminiLLM
from chia.simulators.champsim import ChampSimNode

from loop.simulate import build_binary

TUNED_CACHES = ["L1D", "L2C", "LLC"]


@ChiaFunction(resources={"champsim_build": 1})
def build_from_config(config, champsim_root):
    """Config dict -> compiled ChampSim binary as bytes (travels through Ray's object store)."""
    config_dir = os.path.join(tempfile.gettempdir(), "chia_champsim_configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, config["executable_name"] + ".json")
    with open(config_path, "w") as config_file:
        json.dump(config, config_file)
    binary_path = build_binary(config_path, champsim_root)
    with open(binary_path, "rb") as binary_file:
        return binary_file.read()


@ChiaFunction(resources={"champsim": 1})
def simulate(binary, trace_path, warmup_instructions, simulation_instructions):
    """Run CHIA's ChampSim node and flatten its result to {ipc, <cache>_mpki, ...}."""
    result = ChampSimNode.run_champsim(
        binary, trace_path, warmup_instructions=warmup_instructions,
        simulation_instructions=simulation_instructions, timeout_s=3600)
    if not result.success:
        raise RuntimeError("ChampSim run failed: " + result.stdout_tail[-500:])
    metrics = {"ipc": result.ipc}
    for cache_name in TUNED_CACHES:
        stats = result.cache_stats[cache_name]
        # Demand traffic only, same definition as loop/simulate.py.
        misses = sum(stats.load_miss) + sum(stats.rfo_miss) + sum(stats.write_miss)
        hits = sum(stats.load_hit) + sum(stats.rfo_hit) + sum(stats.write_hit)
        metrics[cache_name + "_hits"] = hits
        metrics[cache_name + "_misses"] = misses
        metrics[cache_name + "_mpki"] = misses * 1000.0 / result.instructions
    return metrics


class AnalystNode:
    """Gemini on Vertex through CHIA's model layer. One instance per loop run."""

    def __init__(self, model, project, location="us-central1"):
        self.llm = VertexGeminiLLM(model=model, project=project, location=location,
                                   system_message="Answer with valid JSON only, no prose.")

    @ChiaFunction(resources={"vertex_creds": 0.01})
    def ask(self, prompt):
        """Prompt -> parsed JSON. Token counts are recorded by CHIA's profiler."""
        answer = self.llm.prompt(prompt)
        if not answer.success:
            raise RuntimeError("Gemini call failed: " + answer.stderr[-500:])
        text = answer.result.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
