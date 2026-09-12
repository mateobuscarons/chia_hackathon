"""CHIA nodes: the loop's simulator and LLM calls as schedulable @ChiaFunction tasks.

  build_from_config  - compile ChampSim for an arbitrary config JSON. Fills the
                       gap in chia.simulators.champsim, whose build node only
                       accepts a prefetcher module (upstream PR candidate).
  simulate           - run a built binary on one trace per core and flatten the
                       stats to the metrics dict the loop uses. CHIA's own
                       ChampSimNode.run_champsim takes a single trace, so the
                       multi-core chip needs this one (second upstream candidate).
  AnalystNode.ask    - Gemini through chia.models.vertex, JSON in / JSON out.

Resources: "champsim_build" is 1 per checkout (builds share the tree and
cannot overlap); "champsim" is one slot per simulation core.
"""

import hashlib
import json
import os
import stat
import tempfile

from chia.base.ChiaFunction import ChiaFunction
from chia.models.vertex import VertexGeminiLLM
from chia.trace.profiler import _ProfiledResult

from loop.simulate import build_binary, run_simulation


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
def simulate(binary, trace_paths, warmup_instructions, simulation_instructions):
    """Write the binary to a content-addressed path on this worker (once), run it
    on the traces (one per core) and return {ipc, <cache>_mpki, ...}."""
    # With the profiler on, a task's result travels wrapped in CHIA's profiling
    # record; only get() unwraps it, and here the build future arrives directly.
    if isinstance(binary, _ProfiledResult):
        binary = binary.value
    content_hash = hashlib.sha256(binary).hexdigest()[:16]
    binary_path = os.path.join(tempfile.gettempdir(), "champsim_" + content_hash)
    if not os.path.isfile(binary_path):
        temporary_path = "{}.{}.tmp".format(binary_path, os.getpid())
        with open(temporary_path, "wb") as binary_file:
            binary_file.write(binary)
        os.chmod(temporary_path, os.stat(temporary_path).st_mode | stat.S_IXUSR)
        os.replace(temporary_path, binary_path)
    return run_simulation(binary_path, trace_paths, warmup_instructions, simulation_instructions)


class AnalystNode:
    """Gemini on Vertex through CHIA's model layer. One instance per worker process."""

    def __init__(self, model, project, location="us-central1"):
        # 65k output tokens: the model's thinking counts against this cap, and an
        # answer of eight full designs was truncated at the layer's 16k default.
        self.llm = VertexGeminiLLM(model=model, project=project, location=location, max_tokens=65536,
                                   system_message="Answer with valid JSON only, no prose.")

    @ChiaFunction(resources={"vertex_creds": 0.01})
    def ask(self, prompt):
        """Prompt -> parsed JSON. Token counts are recorded by CHIA's profiler."""
        answer = self.llm.prompt(prompt)
        if not answer.success:
            raise RuntimeError("Gemini call failed: " + answer.stderr[-500:])
        text = answer.result.strip()
        if text == "":
            raise RuntimeError("Gemini returned an empty answer")
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
