# Cache hierarchy design under hard caps with a council of agents, on CHIA

A council of Gemini agents searches 13 knobs of a three-level cache hierarchy (6.6 million designs)
under a silicon cap and a power cap, with one ChampSim simulation per design to judge it. It is a
CHIA loop: every model call, build, simulation, candidate and gate decision is a CHIA task, and
one profiler log per run holds all of them. On the llama2 inference trace the council reaches the
quality random search has after 1000 designs in 39 to 44 designs, and with two decider gates
applied it reaches the same final IPC for 22 to 30% less model spend.

The blocks the loop needed, and CHIA did not have, ship here in CHIA's layout: a spend view and
cap over the profiler log, an evaluation ledger, a calibrated decider behind a call gate and a
context gate, a Gemini node that bills thinking tokens, and a ChampSim node that builds a whole
hierarchy from one configuration. The call gate, run on CHIA's own CIRCT issue loop, cuts its
assess stage from 16 to 13 agent turns with the same issues sent forward.

## One command, no key

```bash
python3 council_loop/demo.py
```

Prints, in under a second and from the shipped records alone: the gate on CHIA's CIRCT stage, the
gate's counterfactual on the council and what applying it saved, the council against random
search, and the loop's own profiler logs summed by the spend view.

## What is here

```
council_loop/            the loop: analyst.py (the model calls), council.py (the recipe), search.py
                         (the arms and the driver), simulate.py (builds and runs through CHIA's
                         ChampSim nodes), space.py, socs.py, suite.py, chip.py; ledgers.py and
                         report.py score and tabulate runs; audit.py, fidelity.py, circt_assess.py
council_loop/chia_blocks the blocks, tests and docs in CHIA's tree layout; install.sh puts them
                         into the released package; check.sh proves them in a clean environment
council_loop/demo.py     the readout above
results/                 every record the paper cites (see results/README.md)
paper/                   the paper (bash paper/build.sh)
upstream/                the fixes and findings that go to CHIA and ChampSim
```

## The blocks, in one line each

| block | module | what it does |
|---|---|---|
| spend view and cap | `chia/trace/spend.py` | sums tokens and USD per node and model from the profiler log; `chia viz-profile --format spend`; `LLMSpend` stops a loop before the call that would pass a cap |
| evaluation ledger | `chia/trace/ledger.py` | one row per candidate; best-so-far, evaluations to a target, plateau, speedups, rank agreement |
| decider and call gate | `chia/models/decider.py`, `call_gate.py` | one yes-or-no question per optional agent call, answered by a calibrated decider; shadow records, apply skips |
| context gate | `chia/models/context_gate.py` | per briefing section, is it needed; a difficulty rating sets the thinking budget |
| Gemini JSON node | `chia/models/vertex_json.py` | one JSON call at a set temperature and thinking budget, every token kind and the USD in the event |
| ChampSim configuration build | `chia/simulators/champsim_config.py` | builds ChampSim from a whole configuration; a 7-line patch returns the simulator's raw record from CHIA's run node |

## Checks

```bash
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash council_loop/chia_blocks/install.sh .venv          # the blocks and two patches into the installed CHIA
cd council_loop/chia_blocks && ../../.venv/bin/python -m pytest -q chia && bash check.sh
```

`check.sh` builds a clean environment with the released CHIA, installs the blocks, runs the 33
tests inside the package and prints the spend view over the shipped CIRCT logs.

## Running the loop

Needs a ChampSim checkout at `champsim/` (with `upstream/0002-champsim-spp-dev-ghr-victim.patch`
applied), the traces under `traces/`, credentials for Gemini on Vertex (`GCP_PROJECT`, application
default credentials) and a TypeSafe key for the decider (`TYPESAFE_API_KEY`, or `jev_api=` in `.env`).
The CACTI characterisation is shipped in `results/cacti_22nm.json`, so no CACTI build is needed.

```bash
# the frozen recipe, gates in shadow, caps on (SOC=nocap for caps off)
SOC=next OPENINGS=20 SEEDS=4 BUDGET=60 WAVE=5 PLATEAU=20 CONTEXT_GATE=shadow CALL_GATE=shadow \
  LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=5 PARALLEL_RUNS=4 \
  .venv/bin/python -m council_loop.search llama2 <tag> council
# gates applied
CONTEXT_GATE=apply CALL_GATE=apply CALL_GATE_THRESHOLD=0.4 GATE_THRESHOLD=0.5 ANALYST_GATE_THRESHOLD=0.3 ...
# random search, 1000 designs
SOC=next SEEDS=3 BUDGET=1000 LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=10 PARALLEL_RUNS=3 \
  .venv/bin/python -m council_loop.search llama2 <tag> random
# every prompt of an opening round, no model call
SOC=next LOOP_WARMUP=1000000 LOOP_SIM=2000000 .venv/bin/python -m council_loop.search preview llama2
```

A run writes its report to `results/runs/`, its ledger to `results/ledgers/<tag>.jsonl` and its
profiler log to `results/profiles/<tag>/`. `python -m council_loop.ledgers` rebuilds ledgers from
reports and from random's seeds; `python -m council_loop.report` writes the gating tables.

## License

BSD-3-Clause, see `LICENSE`.
