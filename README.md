# LLM + memory + Bayesian optimisation: few-shot cache tuning for workloads never seen before (a CHIA loop)

The direction of this project is one search agent made of three parts, each covering
what the others cannot: an **LLM** that reads the workloads and proposes designs, a
**memory** of earlier searches on other workloads that tells it what paid, what hurt
and where the open questions are, and **Bayesian optimisation** that picks among the
LLM's proposals with a surrogate fit on the current run. The setting is few-shot: a
ChampSim cache hierarchy, a workload suite the agent has never seen, and a budget of
16 simulated designs, of which only the first few matter for the claim.

The memory is written by the code from result tables: one **case** per workload,
holding its descriptors, its best design and every measured single-knob effect. The
agent reads a **digest** of the closest remembered workloads: the moves that paid
everywhere, the traps, where the best designs disagree, and one design to copy and
adapt, the one that did best across every remembered workload. The arms are an
ablation of the three parts under the identical budget: Bayesian optimisation alone,
the LLM alone, and the LLM with the memory digest and a Gaussian process choosing
among its proposals. The headline cell remembers SPEC and graph searches and is
tested on Google datacenter traces.

## Results so far

Cell `dc`: memory from SPEC17 (mcf, omnetpp, lbm) and GAP graph searches (bfs.urand, pr.urand, bfs.kron),
tested on three Google datacenter traces the loop has never seen (sierra.a.4, merced, tahoe), Gemini 2.5
Flash, 16 designs per run. The score is how much of the distance from the stock chip (0.3837 suite IPC) to
the best design known on this suite (0.4933, +28.6%) a method has covered after N simulated designs.

| arm | D1 | D2 | D4 | D6 | D8 | D12 | D16 | best design |
|---|---|---|---|---|---|---|---|---|
| Bayesian optimisation alone | 43% | 43% | 50% | 58% | 76% | 84% | 86% | 0.4778 |
| LLM alone | 54% | 76% | 80% | 81% | 84% | 87% | 88% | 0.4802 |
| LLM + memory | 90% | 91% | 94% | 95% | 95% | 97% | 98% | 0.4910 |

Mean over seeds; the last column is the mean of each run's best design.

- The memory arm's **first** simulated design is already at 90% of the gap, above where
  either baseline ends after sixteen. That is the few-shot claim: the value is in the
  opening, and neither baseline closes it inside the budget.
- It is the only arm to pass 90%, and it passes 95% by design 6.
- It also removes the variance. Across seeds the sixteenth design lands at 51-98% of the
  gap for Bayesian optimisation, 84-95% for the plain LLM, and 96-99% with the memory:
  a fixed simulation budget becomes predictable, which is what a design team buys.
- The reference is the best design measured on all three traces across every run and
  sample cached so far, not a proven optimum; a longer reference search
  can only move it up, which would lower every share in the table.

The same loop and the same memory on the three admitted datacenter traces the first suite did not
take (whiskey, bravo, delta): nothing about them shaped the memory, the design it hands over, or the
choice of the first suite, and that design had never been simulated on them. Stock 0.4389, best
design known 0.5463 (+24.5%).

| arm | D1 | D2 | D4 | D6 | D8 | D12 | D16 | best design |
|---|---|---|---|---|---|---|---|---|
| Bayesian optimisation alone | 43% | 43% | 50% | 57% | 58% | 69% | 83% | 0.5276 |
| LLM alone | 26% | 54% | 72% | 74% | 76% | 79% | 85% | 0.5305 |
| LLM + memory | 81% | 82% | 88% | 93% | 94% | 96% | 97% | 0.5428 |

Mean over seeds. Across seeds the sixteenth design lands at 63-94% of the gap for Bayesian
optimisation, 78-91% for the plain LLM, and 92-100% with the memory — one memory run found the best
design known on this suite. Every design and prompt of this run is in `results/run_dc2_e1.json`.

Every design, every prompt and every model answer of the 16-design runs are in
`results/run_dc_d3.json` and `run_dc_d3b.json`; the earlier 8-design runs are in
`run_dc_d1.json` and `run_dc_d2.json`.

## Layout

| file | role |
|---|---|
| `loop/agent.py` | the LLM agent, one code path for both LLM arms (memory digest on/off, GP selection on/off): the Gemini call (Vertex; under CHIA through `chia.models.vertex`) with its cost log, prompt from data sections, proposals, retry, deterministic fallback |
| `loop/memory.py` | cases built from result tables, descriptor-distance retrieval, the digest the agent reads, the design it hands over, the offline sign-survival check |
| `loop/bo.py` | the GP surrogate and the expected-improvement baseline over a seeded sample plus the incumbent's neighbourhood |
| `loop/champsim_problem.py` | ChampSim glue: traces -> the `problem` dict (suite objective = geomean IPC), descriptors, result-table cache |
| `loop/configs.py` | the chip profile, the 13-knob space, area budget, latency-from-size, config generation |
| `loop/simulate.py`, `loop/chia_nodes.py` | build and run ChampSim; the same as CHIA tasks (build from any config, simulate; the Vertex node) |
| `loop/trace_profile.py` | workload profile from the trace alone (footprint theory miss-ratio curve) |
| `loop/workloads.py` | the admission gate and the headroom screen (free); the 11-design probe and the reference search (simulate); trace fetching (HTTP prefix, GAP zip member); table merge |
| `loop/run.py`, `loop/summarize.py` | cells and arms, the report, the CHIA entry (`LOOP_DISPATCH=chia`); the few-shot score table and the progress view of a running cell |
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
SEEDS=5 BUDGET=16 LOOP_DISPATCH=chia python -m loop.run dc d3     # the headline cell
python -m loop.summarize results/run_dc_d3.json results/run_dc_d3b.json
```
Setup, cells, evidence and rules are in `CLAUDE.md`; the VM recipe in `cluster/README.md`.
