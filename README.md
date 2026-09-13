# LLM + memory + optimisation: few-shot cache tuning for workloads never seen before (a CHIA loop)

One search agent made of three parts, each covering what the others cannot: an **LLM**
that reads the workloads and proposes designs, a **memory** of earlier searches on other
workloads that tells it what paid, what hurt and where the open questions are, and a
**surrogate** that picks among the LLM's proposals from the current run. The setting is
few-shot: a ChampSim cache hierarchy, a workload suite the agent has never seen, and a
budget of 16 simulated designs, of which only the first few matter for the claim.

The memory is written by the code from result tables: one **case** per workload, holding
its descriptors, its best design and every measured single-knob effect. The agent reads a
**digest** of the closest remembered workloads: the moves that paid, the traps, where the
best designs disagree, and one design to copy and adapt. The arms are an ablation of the
parts under an identical budget. The headline cell remembers SPEC and graph searches and
is tested on Google datacenter traces.

## Results

Cell `dc`: memory from SPEC17 (mcf, omnetpp, lbm) and GAP graph searches (bfs.urand,
pr.urand, bfs.kron), tested on three Google datacenter traces the loop has never seen
(sierra.a.4, merced, tahoe), Gemini 2.5 Flash, 16 designs per run, 5 seeds. The score is
how much of the distance from the stock chip (0.3837 suite IPC) to the best design known
(0.4994, +30.1%) a method has covered after N simulated designs.

| arm | D1 | D2 | D4 | D8 | D12 | D16 | best design |
|---|---|---|---|---|---|---|---|
| optimisation alone | 22% | 22% | 42% | 64% | 82% | 89% | 0.4861 |
| LLM alone | 49% | 56% | 72% | 80% | 82% | 84% | 0.4810 |
| LLM + memory | 90% | 91% | 91% | 92% | 93% | 93% | 0.4915 |

Mean over seeds; the last column is the mean of each run's best design.

- The memory arm's **first** simulated design is already at 90% of the gap, above where
  either baseline ends after sixteen. That is the few-shot claim.
- **The durable form of it is speed, not level.** Against an independent 100-design search
  on the same suite: the memory arm's first design is what that search needs **42** designs
  to match, and its 16-design result is what it needs **77** to match — **4.8× fewer
  simulations for the same design quality**. Quality-at-N moves whenever the reference
  moves; designs-to-quality does not.
- **What transfers is a good region, not the optimum.** The best design known sits six knob
  changes from the one the memory hands over, and everything within five of it is capped at
  93–95%. The memory buys the opening almost for free and cannot by itself buy the last ten
  percent.
- The ceiling is set by an **independent** search (`loop/forest.py`), not by any arm.
  Scoring against a design an arm found bounds the metric at the best arm — doing that
  inflated an earlier version of this table by about five points.

The same loop and the same memory on the three admitted datacenter traces the first suite
did not take (whiskey, bravo, delta): nothing about them shaped the memory or the design it
hands over, and that design had never been simulated on them. Stock 0.4389.

| arm | D1 | D2 | D4 | D8 | D12 | D16 | best design |
|---|---|---|---|---|---|---|---|
| optimisation alone | 22% | 26% | 46% | 72% | 78% | 85% | 0.5315 |
| LLM alone | 24% | 47% | 72% | 77% | 79% | 80% | 0.5263 |
| LLM + memory | 93% | 93% | 93% | 93% | 93% | 95% | 0.5433 |

This suite's ceiling is not yet independent, so these shares will fall by roughly five
points when its reference search lands. Every design, prompt and model answer is in
`results/run_dc_f1.json` and `results/run_dc2_f1.json`.

The findings and how they were measured are in `REPORT.md`.

## Layout

| file | role |
|---|---|
| `loop/agent.py` | the LLM agent, one code path for both LLM arms (memory digest on/off, surrogate selection on/off): the Gemini call (Vertex; under CHIA through `chia.models.vertex`) with its cost log, prompt from data sections, proposals, retry, deterministic fallback |
| `loop/memory.py` | cases built from result tables, descriptor-distance retrieval, the digest the agent reads, the design it hands over, the offline sign-survival check |
| `loop/memory_build.py` | the declared procedure that fills a memory: search, one-knob anchor sweeps, confirm — with every design's stage and proposer recorded |
| `loop/forest.py` | the random-forest surrogate and the search that uses it: the optimisation baseline, the memory-started arm, and the independent reference |
| `loop/bo.py` | the Gaussian process, kept for the memory arm's candidate filter and the optimiser-searched memory build |
| `loop/champsim_problem.py` | ChampSim glue: traces -> the `problem` dict (suite objective = geomean IPC), descriptors, result-table cache |
| `loop/configs.py` | the chip profile, the 13-knob space, area budget, latency-from-size, config generation |
| `loop/simulate.py`, `loop/chia_nodes.py` | build and run ChampSim; the same as CHIA tasks (build from any config, simulate; the Vertex node) |
| `loop/trace_profile.py` | workload profile from the trace alone (footprint theory miss-ratio curve) |
| `loop/workloads.py` | the admission gate and the headroom screen (free); the 11-design probe (simulates); trace fetching; table merge |
| `loop/run.py`, `loop/summarize.py` | cells and arms, the report, the CHIA entry (`LOOP_DISPATCH=chia`); the few-shot score table and the progress view |
| `results/table_<trace>.json` | the shared simulation cache, one per workload (the paper's dataset) |
| `results/profile_<trace>.json` | the profile of every workload in use |
| `results/memory_llm.json`, `results/run_<cell>_<tag>.json` | the memory and a cell's report |
| `cluster/` | the one-VM GCP recipe, the detached launcher, CHIA cluster configs |
| `upstream/` | patches for PRs back to CHIA / ChampSim (described in `REPORT.md`) |

## Run

```bash
python -m loop.run memory dc                                          # build the memory (simulates)
python -m loop.memory leave_one_out <memory traces> -- <test traces>  # free: do effects transfer?
python -m loop.workloads admit                                        # which workloads are worth a search
python -m loop.workloads headroom <trace> <trace> <trace>             # headroom and the share no single knob reaches
SEEDS=1 LOOP_DISPATCH=chia python -m loop.run smoke s1                # every arm, one round, through CHIA
SEEDS=5 BUDGET=16 LOOP_DISPATCH=chia python -m loop.run dc f1         # the headline cell
python -m loop.forest 100 10 <trace> <trace> <trace>                  # the suite's independent ceiling
python -m loop.summarize results/run_dc_f1.json
```
Setup, cells, design and rules are in `CLAUDE.md`; the findings in `REPORT.md`; the VM
recipe in `cluster/README.md`.
