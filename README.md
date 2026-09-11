# Few-shot cache tuning from a memory of earlier searches (a CHIA loop)

An LLM agent tunes a ChampSim cache hierarchy for a workload suite it has never
seen, with a budget of 8 simulated designs. It carries a **memory** written by the
code from earlier searches on other workloads: one **case** per workload, holding
its descriptors, its best design and every measured single-knob effect. The agent
reads a **digest** of the nearest cases: the moves that paid everywhere, the traps,
where the best designs disagree, and the closest best design to copy. A hybrid arm
lets the LLM propose eight designs a round and a Gaussian process pick the two to
simulate. Baselines under the identical budget: textbook Bayesian optimisation, the
same agent without memory, and a one-shot replay of the nearest case's best design. The headline cell remembers SPEC and graph searches and
is tested on Google datacenter traces.

## Layout

| file | role |
|---|---|
| `loop/agent.py` | the LLM agent (four arms from two switches: memory digest on/off, GP selection on/off): the Gemini call (Vertex; under CHIA through `chia.models.vertex`) with its cost log, prompt from data sections, proposals, retry, deterministic fallback |
| `loop/memory.py` | cases built from result tables, descriptor-distance retrieval, the digest the agent reads, replay, the offline leave-one-out check |
| `loop/bo.py` | the GP surrogate and the expected-improvement baseline over a seeded sample plus the incumbent's neighbourhood |
| `loop/champsim_problem.py` | ChampSim glue: traces -> the `problem` dict (suite objective = geomean IPC), descriptors, result-table cache |
| `loop/configs.py` | the chip profile, the 13-knob space, area budget, latency-from-size, config generation |
| `loop/simulate.py`, `loop/chia_nodes.py` | build and run ChampSim; the same as CHIA tasks (build from any config, simulate; the Vertex node) |
| `loop/trace_profile.py` | workload profile from the trace alone (footprint theory miss-ratio curve) |
| `loop/workloads.py` | the admission gate and the headroom screen (free); the 11-design probe and uniform samples (simulate); trace fetching (HTTP prefix, GAP zip member); table merge |
| `loop/run.py`, `loop/summarize.py` | cells and arms, the report, the CHIA entry (`LOOP_DISPATCH=chia`); the few-shot score table and the verification list |
| `results/table_<trace>.json` | the shared simulation cache, one per workload (the paper's dataset) |
| `results/profile_<trace>.json` | the profile of every workload in use |
| `results/memory_<cell>.json`, `results/run_<cell>_<tag>.json` | a cell's memory and report |
| `cluster/` | the one-VM GCP recipe, the detached launcher, CHIA cluster configs |
| `upstream/` | patches and notes for PRs back to CHIA / ChampSim |

## Run

```bash
python -m loop.run memory dc                              # cases from the cached tables
python -m loop.memory leave_one_out <memory traces> -- <test traces>   # free: do effects and best designs transfer?
python -m loop.workloads admit                               # which workloads are worth a search
python -m loop.workloads headroom <trace> <trace> <trace>    # headroom and the share no single knob reaches
SEEDS=1 LOOP_DISPATCH=chia python -m loop.run smoke s1   # every arm, one round, through CHIA
SEEDS=2 LOOP_DISPATCH=chia python -m loop.run dc d1               # the headline cell
python -m loop.summarize results/run_dc_d1.json
```
Setup, cells, evidence and rules are in `CLAUDE.md`; the VM recipe in `cluster/README.md`.
