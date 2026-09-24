"""Proof of the configuration-build block: build one design through it and run it through CHIA's own
run node; the IPC must equal the table's value for the same design (ChampSim is deterministic on one
machine). Usage, in a venv with chialoops and the block installed:
    python check_champsim_config.py <spec.json> <champsim root> <trace>
where spec.json holds {"config": <full ChampSim configuration>, "table_ipc": <the value to match>}.
"""
import json, os, sys, time
from chia.simulators.champsim_config import build_champsim_config, check_config
from chia.simulators.champsim import ChampSimNode
spec = json.load(open(sys.argv[1]))
config = spec["config"]
print("check_config:", check_config(config) or "no problems")
root = sys.argv[2]
started = time.time()
built = build_champsim_config(root, config, timeout_s=1800)      # a plain call: the node runs in-process
print("build: success={} executable={} binary={:.1f} MB in {:.0f}s".format(built.success, built.module_name[:40] + "...", len(built.binary) / 1e6, time.time() - started))
if not built.success:
    print("stderr tail:", (built.stderr or "")[-800:]); sys.exit(1)
started = time.time()
run = ChampSimNode.run_champsim(built.binary, sys.argv[3],
                                warmup_instructions=1_000_000, simulation_instructions=2_000_000, timeout_s=1800)
print("run: success={} ipc={:.4f} in {:.0f}s | table ipc for this design: {:.4f} | equal: {}".format(
    run.success, run.ipc, time.time() - started, spec["table_ipc"], abs(run.ipc - spec["table_ipc"]) < 1e-4))
