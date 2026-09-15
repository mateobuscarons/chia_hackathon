# Few-shot cache tuning from a memory of earlier searches

Cache-hierarchy design-space exploration on ChampSim. The question: **given a stock chip and a
workload suite the loop has never seen, how many simulations does it take to get within 5 % of the
best design known, with and without a memory of earlier searches on other workloads?**
Designs-to-level, not level-at-N — the second moves whenever the ceiling moves, and it has.

Prior work it is measured against: ArchAgent (arXiv 2602.22425), which has an agent write cache
policies inside ChampSim and states that it starts every search from scratch and that
"extrapolation from representative workloads" is open; ArchGym ("all optimizers tie under tuned
hyperparameters"); DOSA; AgentDSE (MLArchSys @ ISCA 2026, arXiv 2606.21836), the closest prior
work — a coding agent tunes a ChampSim hierarchy in ~50 simulations, no transfer.

**Findings live in `REPORT.md`. Contributions back to the frameworks live in `CHIA_BLOCKS.md`.**
This file is state, design and operations.

## The claim

`dc2` (whiskey, bravo, delta), stock 0.4389, best known 0.5485 from an independent 100-design
search. Same optimizer both rows, same settings; only the starting point differs.

```
share of the gap        D1    D4    D8   D16     designs a from-scratch search needs
bo        (stock)      19%   46%   57%   68%     90% -> D15   93% -> D26   95% -> D37
pooled_bo (memory)     93%   93%   95%   96%
```

`dc` (sierra.a.4, merced, tahoe), stock 0.3837, best known 0.4994:

```
bo        (stock)      22%   61%   72%   88%     90% -> D27   92% -> D30   93% -> D30
pooled_bo (memory)     90%   90%   91%   92%
```

The head start is the same on both suites. It converts differently, and §2 of `REPORT.md` now
gives the mechanism: the handover's neighbourhood is far richer on `dc2` (a two-knob move reaches
99 % of the gap) than on `dc` (nothing within five knobs beats 95 %).

## The design

**Arms** (`loop/search.py`), all from the stock chip, same budget, seeds and fidelity:

- `bo` — random forest + expected improvement, one design per round, so every pick is made
  against measured outcomes.
- `pooled_bo` — the identical search, started from the design the memory hands over. One variable
  differs from `bo`, and it is the memory. **This is the arm to quote.**
- `llm_alone` — the Gemini agent, two designs a round from the results table (`loop/analyst.py`).

**BUDGET counts designs, not rounds.** The unit of cost is the design; the unit of adaptation is
the round. The two designs of an LLM round are chosen together from the same information, so D1 is
the design it ranked first, not a step of learning. `bo` and `pooled_bo` buy one design per round,
so every column is a fresh decision.

**The memory** (`results/memory.json`, built by `loop/memory.py`): one design plus the receipt for
it. Built in two stages over groups of memory workloads — **search**, one LLM search per group
(40 designs, 8 a round, every proposal measured); **confirm**, each group's best 20 designs
measured on every memory workload. A design may be handed over only if it was measured on all but
one of them, which is what the confirm stage exists for and is worth four points of gap share.
`handover()` is the whole interface a search sees: 13 knob values fitted to the area budget.

**Search space** (`loop/space.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from
size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC
sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost
nothing (known simplification). Chip C: wide core, two memory channels, placeholder profile.
Fidelity 5M warmup / 10M simulated (Spearman 0.919 against 50M/50M over 26 designs).

**Scoring** (`search score`): per arm, the share of the stock-to-best-known gap after N designs,
mean over seeds with min..max below. Best known = the best design measured on every workload of the
suite in the cached tables, so it moves as searches land.

**The ceiling must stay an independent mechanism.** `search ceiling` is the same forest run long
from the stock chip. Scoring against a design one of the arms found bounds the metric at the best
arm, which happened once on `dc` and cost every arm five points. **And never measure a ratio
against the ceiling run**: it uses warm-up 10 and batch 6, which flattens it exactly where a
comparison lives, and that produced a retracted 4.8x. Measure against the arm's own configuration
run long — `results/baseline/<cell>_s<seed>.json`, 75 designs, 3 seeds a cell, verified to
reproduce the `bo` arm design-for-design.

**The prompt** (`analyst.prompt`) is the chip and its knobs, the results table, the task. Nothing
else — the workload descriptors were removed (see `REPORT.md` §3), so the agent knows only what it
has measured. A pick must be a real, in-budget, unmeasured design; else one retry with the rejected
designs listed, then a deterministic one-knob perturbation of the incumbent. Never a random design.

## Cells and workloads

| cell | test | role |
|---|---|---|
| `dc` | sierra.a.4, merced, tahoe (Google datacenter, DPC4 `gtrace_v2`) | the headline |
| `dc2` | whiskey, bravo, delta | the confirmation, on traces that shaped nothing |
| `smoke` | mcf, lbm, one round | the gate before any launch |

The memory remembers mcf, omnetpp, lbm, bfs.urand, pr.urand, bfs.kron and nothing else.

**All six datacenter traces are held out from the memory in the same way** — no simulation during
the build, no influence on the handover. The difference is who else saw them: twelve families were
screened, six admitted, all six probed with 11 designs, and then we *chose* which three to headline
on their numbers. `dc2` is the three we did not pick, so it is the control for **our** selection,
not the memory's.

**Why the datacenter set.** Their access streams are irregular, so a stride prefetcher does not hand
over the gain. On every datacenter trace a replacement policy alone hurts or does nothing (srrip
−13 % on sierra.a.4) while the best designs contain one: the policy pays only after a prefetcher and
a bigger LLC, the published SPEC-to-datacenter inversion, measured here.

**Do not merge the six into one suite.** Only 81 designs have ever been measured on all six, against
1294 on `dc` and 948 on `dc2`, so a merged ceiling would rest on 81 designs and every share would
inflate. The compute is identical either way, so merging costs the replication and buys nothing.

## What is in `results/`

```
memory.json          the handover and its receipt
tables/<trace>.json  the simulation cache — the dataset, and the denominator `score` uses
runs/<cell>_<tag>.json   a cell's report: every design, and round 1's prompt for LLM arms
baseline/<cell>_s<seed>.json   the `bo` arm run to 75 designs, behind the designs-to-level table
llm_usage.json       running token cost (gitignored: per machine)
```

Nothing else is written. `results_archive/` holds tables moved aside before a memory rebuild; what
lands there is superseded by definition and must never be merged back.

**The simulation cache.** One table per workload, keyed by `space.config_name(knobs)` — the chip
plus a sorted knob listing, which carries the whole design, so nothing has to store it twice.
Every batch reads the table before simulating and appends to it under a file lock with an atomic
rename, so six parallel runs share one cache and `workloads merge` folds in another machine's.
2305 designs and 7980 measurements so far; within a single report about 20 % of requested designs
are already there, and `pooled_bo`'s five seeds all open on the same handover, so five requests
cost one set of simulations. `champsim_bin/` is the same idea one level down: 939 compiled
binaries under the same names, worth ~31 hours of compiling.

A row is `{"knobs": {...}, "metrics": {ipc, LLC_hits, LLC_misses, LLC_mpki}}`. **A design the
simulator cannot measure is cached too**, as `"metrics": null` — the lookup raises on it and
`is_candidate` excludes it, so a bad design costs one failed attempt ever rather than one per run.
To retry one deliberately, delete its row.

## Next, in this order

1. **A continuation search that exploits a strong start** — the priority. The handover opens at
   90-93 % and the arms gain only 2-6 points across the whole budget. On `dc` the optimum sits six
   knob changes away and everything nearer is capped at 95 %, so no local refinement reaches it.
   Candidates, none costed: a trust region that restarts globally when it stalls; letting the search
   spend part of its budget on a deliberate multi-knob departure rather than a step from the
   incumbent. Measure as designs-to-level, mean over seeds.
2. **Re-run `llm_alone`.** Its published numbers (D1 24 % on `dc2`, 49 % on `dc`) came from a prompt
   that carried workload descriptors, which are gone. The arm now has no information the optimizer
   lacks, which makes it a cleaner baseline — and an unmeasured one.
3. **Rebuild the memory under the two-stage procedure.** `results/memory.json` was written from the
   last build, which also ran an anchor stage (hence `"designs": 163`); the current procedure asks
   for ~76. The handover is identical either way, but the artifact and the code should agree.
4. **Push the test workloads further out.** What is measured today is transfer inside the memory's
   own neighbourhood, and with the descriptors gone there is no distance metric left to say how far
   out a candidate is. Defining "further out" by measurement rather than by trace statistics is the
   first half of this item; the second is a family the old admission gate rejected for being too
   stride-regular.
5. **An unseen chip**, then **a stronger model** (`ANALYST_MODEL=gemini-2.5-pro`).

Small, known: the LLM re-proposes already-measured designs, 4 to 9 of 8 per round; listing measured
designs in the task text would remove it.

## How to run

```bash
python -m loop.memory build results/memory.json <spec traces> -- <gap traces>   # simulates
SEEDS=5 BUDGET=16 python -m loop.search dc2 h1 [arm,arm]                        # a cell
python -m loop.search score results/runs/dc2_h1.json                            # the score table
python -m loop.search ceiling 100 10 <trace> <trace> <trace>                    # the independent ceiling
BUDGET=75 SEEDS=3 python -m loop.search dc2 long bo                             # a from-scratch curve
python -m loop.workloads probe <trace> ...                                      # the 11-design probe
python -m loop.workloads headroom <trace> ...                                   # what the probe revealed
python -m loop.workloads merge <fetched results dir>                            # VM tables into results/
SEEDS=1 python -m loop.search smoke s1                                          # the gate before any launch
```

Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (designs, default 16), `PARALLEL_RUNS`, `SIM_THREADS`
(concurrency = 3 x this), `CHAMPSIM_BUILD_SHARE` (set to 8 on the VM), `MEMORY_PATH`,
`ANALYST_MODEL` (default `gemini-2.5-flash`; `gemini-2.5-pro` is the only Pro enabled),
`MB_SEARCH` / `MB_CONFIRM` for the memory build.

**Check `gcloud compute instances list` before anything else: the VM must be stopped whenever
nothing runs on it.**

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch   # or spp_dev designs crash
./config.sh champsim_config.json && make -j8 && cd ..
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python numpy scikit-learn google-genai
```

Traces (`traces/`, gitignored): SPEC17 from `https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/`;
GAP from Zenodo record 20043527 with `loop.workloads fetch_gap` (cite *Characterizing the impact of
last-level cache replacement policies on big-data workloads*, IISWC 2020); datacenter traces from the
DPC4 bucket `https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt`
(`gtrace_v2/<family>/<trace>.champsim.gz`) as 100 MB prefixes with `loop.workloads fetch`.

GCP: project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1` (c2d-standard-32, europe-west4-a,
~1.5 USD/h), recipe in `cluster/README.md`. **The VM has no git checkout: code goes there with
`gcloud compute scp loop/*.py`, never `git pull`.**

Budget: **~80-90 EUR of credits left** — the GCP console is the source of truth, this line goes
stale and must not be extrapolated from. Cost of a run, measured over every VM run: **94-154 designs
simulated per hour, ~110 typical**. Arms are not the cost, designs are: estimate a launch as
(cells x arms x seeds x budget) / 110 hours. The binding limit is the global `CPUS_ALL_REGIONS`
quota of 32 while europe-west4 allows 200; raising that one quota would cut wall time
proportionally at the same total cost, and is refused while the billing account is on trial credits.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only** — they describe the
mechanism and the measurements, do not speculate about how the work will be received, and name no
one outside the technical references. Every launch needs the user's explicit OK with cost and time.
The Mac runs reference searches and overnight work under `caffeinate` (which stops idle sleep,
**not** lid-close sleep). The VM is stopped when idle. Kill by process group or exact pid, never a
broad `pkill`. Code stays lean, one mechanism, no version archaeology; domain decisions are surfaced
with options and left to the user. Deleted material is in git history, never in the tree.
