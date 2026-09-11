# Rules that bet: a calibrated, hypothesis-driven CHIA loop for cache-hierarchy design

An LLM agent tunes ChampSim cache hierarchies from the results table and the
workloads' trace-profiled descriptors, and carries a **memory** from one search to
the next, written entirely by the system: **cases** and **facts** (what was measured,
where, with provenance, retrieved by descriptor similarity and gated by their own
record) and **strategies** (procedural notes written from wins and failures: which
knob to probe first, which interaction to test after which change, which habit
cost designs). Every retrieved item shows its record; a cited item is scored on
whether the pick beat the incumbent; a fact whose direction fails is retrieved
only for closer workloads afterwards. The agent pre-registers a point forecast for
every design and is Brier-scored; so are the playbook rules and the GP baselines.
The experiment asks whether knowledge survives a **workload-class boundary on one
chip** (chip C's own SPEC results -> graph workloads on chip C -> other graph
workloads on chip C), with a chip-boundary control on the side.

The ledger, forecasters, selector and memory are simulator-agnostic (`loop/loop.py`
and `loop/memory.py` read knob and descriptor names from a `problem` dict);
ChampSim is just the first problem.

## Layout

| file | role |
|---|---|
| `loop/memory.py` | the agent and its memory: prompt assembled from data sections (problem, workloads, distance to memory, nearest cases, similar facts, strategies, own calibration, results, task); the plain agent is the same code with no shelf open; mechanical write path (cases, facts, forecasts) plus one reflection call for strategies; linter (no chip, workload or family names, no numbers in strategies); scoreboard; consolidation; `build` from result tables |
| `loop/loop.py` | the GP round for the baseline arms: fit surrogate, pick by expected improvement with the speaking rules' bounded mean shifts, on a stall one owed claim test or a scan of the incumbent, bet, run, settle, re-scope; `verify_claims` measures every proposed rule before admission |
| `loop/forecast.py` | rules as delta claims (value or direction), controlled pairs (found whatever the search started from), credibility, mean shifts, EI selection, stall scan, measured one-knob effects |
| `loop/playbook.py` | rules + bet ledger, one auditable JSON; Brier scoring; rejected rules |
| `loop/surrogate_gp.py` | GP in log-speedup units (ordinal + one-hot encoding, ARD Matern), vectorised prediction |
| `loop/analyst.py` | every LLM call (agent pick, strategy writing and merging, reflection, distill, re-scope, budget fit); model by `ANALYST_MODEL`; cost log |
| `loop/trace_profile.py` | workload profile from the trace alone (working set, stride regularity, locality, miss-ratio curve from footprint theory); the chip-independent half of every descriptor, read against the chip's geometry and budget in `champsim_problem.chip_descriptors` (footprint ratios, predicted miss ratios, movable MPKI) |
| `loop/champsim_problem.py` | ChampSim glue: SoC + trace(s) -> `problem` dict; single-workload or suite (geomean); descriptors; result-table cache |
| `loop/configs.py`, `loop/socs.py` | search space, SoC profiles, area budget, latency-from-size (placeholders, team-reviewed) |
| `loop/simulate.py`, `loop/collect.py` | ChampSim build (parallel trees, per-binary lock) and run; simulate a fixed design list per chip (`uniform` = a cell's reference sample and null curve, `top` = the fidelity check and the final validation via `LOOP_WARMUP` / `LOOP_SIM`, `structured` = a learn set), fetch one GAP trace by byte range, merge tables simulated on another machine |
| `loop/experiment.py` | distill+verify the playbook from result tables, run the arms (random, bo, bo_pooled, rules, llm_direct, memory) in parallel jobs, write the report |
| `loop/run.py` | the cells (`w1`, `w2`, `k`, `smoke`) and the `learn` step (playbook + pool + memory from a cell's training tables); env `SEEDS`, `FIRST_SEED`, `ROUNDS`, `MEMORY_PATH`, `ANALYST_MODEL` |
| `loop/summarize.py`, `loop/plots.py` | tables (designs to 90/95/99% of the cell's uniform reference, hit fraction, final best, AUC, calibration per forecaster; works on a running report), `progress` reads a cell that is still running straight from its log, figures |
| `loop/offline.py`, `loop/cpi_stack.py`, `loop/surrogates.py` | the zero-cost offline harness: workload admission gate, headroom screen for any workload set from any start design, descriptor coverage, gate G-language, the CPI-stack findings |
| `loop/chia_nodes.py`, `loop/run_chia.py`, `cluster/` | CHIA tasks (build from any config, simulate one trace per core, Vertex analyst), the CHIA entry point that runs a cell with arms as Ray tasks and the profiler on, GCP recipes and the unattended chain |
| `PREREGISTRATION.md` | the predictions written before each cell ran, and how they settled |
| `upstream/` | patches and notes for PRs back to CHIA / ChampSim |

## Run

```bash
python -m loop.memory check                              # offline acceptance checks (empty memory == plain agent, linter, distances)
python -m loop.run_chia local smoke s1                   # the gate before any launch: every arm, one round, real simulations, through CHIA
python -m loop.run learn w1 v1                           # w1's playbook, pool and memory from chip C's SPEC tables
python -m loop.run_chia local w1 v1 results/experiment_v1_playbook.json          # cell w1: GAP set 1 on chip C
python -m loop.memory consolidate results/experiment_v1_memory.json results/experiment_v1_memory_after_w1.json w1
MEMORY_PATH=results/experiment_v1_memory_after_w1.json python -m loop.run_chia local w2 v1 results/experiment_v1_playbook.json   # cell w2: GAP set 2, the headline
python -m loop.run learn k v1_k && python -m loop.run_chia local k v1 results/experiment_v1_k_playbook.json   # cell k: the chip-boundary control
COLLECT_SOCS=C_server python -m loop.collect uniform 300 traces/<gap set 2 traces>   # a cell's reference sample (before the cell runs)
python -m loop.summarize results/experiment_v1_w1.json --uniform 300              # when the runs have finished
python -m loop.summarize progress results/w2a_w2_s0.log --start 0.1656           # a cell still running, from its log
python -m loop.trace_profile traces/605.mcf_s-665B.champsimtrace.xz              # one workload's profile (.xz or .gz; cached under results/)
python -m loop.offline admit | coverage | language <playbook.json>               # free gates
python -m loop.offline headroom [--start spec] <trace> <trace> <trace>           # is there anything for a search to find?
bash cluster/chain.sh v1                                                          # the whole session unattended on the VM (see the script header)
```
Setup (ChampSim, CHIA, traces, GCP auth) is in `CLAUDE.md`; the one-VM cloud recipe in `cluster/README.md`.
