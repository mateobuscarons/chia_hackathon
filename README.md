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

Two workload suites the loop has never seen, both Google datacenter traces, memory built
from SPEC17 (mcf, omnetpp, lbm) and GAP graph searches (bfs.urand, pr.urand, bfs.kron).
Gemini 2.5 Flash, 16 designs per run, 5 seeds per arm. The score is the share of the
distance from the stock chip to the best design known, and that ceiling comes from an
**independent** 100-design search carrying no memory and no LLM — scoring against a design
one of the arms found would bound the metric at the best arm.

**`dc`** (sierra.a.4, merced, tahoe) — stock 0.3837, best known 0.4994, +30.1 %:

| arm | D1 | D2 | D4 | D8 | D12 | D16 | best design |
|---|---|---|---|---|---|---|---|
| optimisation from the stock chip | 22 % | 22 % | 42 % | 64 % | 82 % | 89 % | 0.4861 |
| LLM alone | 49 % | 56 % | 72 % | 80 % | 82 % | 84 % | 0.4810 |
| LLM + memory | **90 %** | 91 % | 91 % | 92 % | 93 % | 93 % | 0.4915 |
| optimisation from the memory's design | **90 %** | 90 % | 90 % | 91 % | 91 % | 92 % | 0.4899 |

**`dc2`** (whiskey, bravo, delta — the admitted traces the first suite did not take, so
nothing about them shaped the memory or the design it hands over) — stock 0.4389, best
known 0.5485, +25.0 %:

| arm | D1 | D2 | D4 | D8 | D12 | D16 | best design |
|---|---|---|---|---|---|---|---|
| optimisation from the stock chip | 22 % | 26 % | 46 % | 72 % | 78 % | 84 % | 0.5315 |
| LLM alone | 24 % | 47 % | 72 % | 77 % | 79 % | 80 % | 0.5263 |
| LLM + memory | **93 %** | 93 % | 93 % | 93 % | 93 % | 95 % | 0.5433 |
| optimisation from the memory's design | **93 %** | 93 % | 93 % | 95 % | 95 % | **96 %** | 0.5443 |

The optimisation row is the **stronger of the two surrogates we ran** on each suite. Both a
tuned Gaussian process and a random forest were run from the stock chip; they tie on `dc`
(89 % against 88 %) and differ by 16 points on `dc2` (84 % against 68 %), which is inside the
noise at these seed counts — individual cold-start runs land anywhere from 48 % to 96 %.
Reporting the weaker one would inflate the margin through a seed draw, so the table takes the
better one and both are in the reports.

- **One design from memory beats sixteen designs of search.** The memory-started arms open
  at 90 % and 93 %; the best memoryless optimizer reaches 89 % and 84 % after sixteen, and
  never reaches the opening on either suite. Nothing differs between those rows but where the
  search begins.
- **The memory is the effect, not the agent reading it.** Handing the memory's design to a
  plain optimizer matches the LLM agent on `dc` (92 % against 93 %) and beats it on `dc2`
  (96 % against 95 %) — with no digest, no prompt and no model call at test time. What the
  LLM demonstrably contributes is at *build* time: at an equal budget its search leaves 36
  single-knob effects supported by controlled pairs where an optimizer's search leaves 0.
- **What transfers is a good region, not the optimum.** The best design known sits six knob
  changes from the one the memory hands over, and everything within five of it is capped at
  93–95 %. The memory buys the opening almost for free and cannot by itself buy the last ten
  percent.

Every design, prompt and model answer is in `results/run_dc_f1.json`, `run_dc_g1.json`,
`run_dc2_f1.json` and `run_dc2_g1.json`. The findings and how they were measured are in
`REPORT.md`.

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
