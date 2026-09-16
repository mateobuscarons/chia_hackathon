# Few-shot cache tuning from a memory of earlier searches

Cache-hierarchy design-space exploration on ChampSim. The question: **given a stock chip and a
workload suite the loop has never seen, how many simulations does it take to get within x % of the
best design known, with and without a memory of earlier searches on other workloads?**
Designs-to-level, not level-at-N.

Prior work it is measured against: ArchAgent (arXiv 2602.22425), an agent writing cache policies
inside ChampSim, every search from scratch; ArchGym ("all optimizers tie under tuned
hyperparameters"); DOSA; AgentDSE (MLArchSys @ ISCA 2026, arXiv 2606.21836), a coding agent tuning
a ChampSim hierarchy in ~50 simulations, no transfer.

**Findings live in `REPORT.md`. Contributions back to the frameworks live in `CHIA_BLOCKS.md`.**
This file is state, design and operations.

## The claim

`dc2` (whiskey, bravo, delta), stock 0.4389, best known 0.5485 from an independent 100-design
search. Share of the stock-to-best-known gap after N designs, mean over seeds, and the design at
which the mean curve reaches a level:

```
                            D1    D4    D8   D12   D16      95%    98%     seeds
bo            (stock)      19%   46%   57%   61%   68%      D37    D73     4 (levels from 3 runs of 75)
council_stock (stock)      21%   52%   63%   87%   91%       -      -      3
pooled_bo     (memory)     93%   93%   95%   95%   96%      D11     -      5
council       (memory)     93%   95%   96%   97%   97%       D4     -      3 (95..99 at D12)
```

`bo` and `pooled_bo` are the same optimizer; only the starting point differs, and the same holds
for `council_stock` and `council`. Without a memory the council and the forest tie. The council seed
that reached 99 % shrank the L2 from 512x16 to 256x4, an eighth of the size, in two consecutive
geometry rounds; that move alone is 98.6 % of the gap and the forest never took it. The other two
seeds stopped at 256x16 after a shrink to 256x8 scored 0.0001 lower and ended the streak. Reports:
`results/runs/dc2_g1.json` (the forests), `results/runs/dc2_c5b.json` (the councils); transcripts
`results/council*-c5b-s*.log`.

`dc` (sierra.a.4, merced, tahoe), stock 0.3837, best known 0.4994, at D1/4/8/16: `bo`
22/61/72/88 %, `council_stock` 25/66/81/87 %, `pooled_bo` 90/90/91/92 %, `council` 90/92/93/93 %.
Both memory arms open at 90 % and neither adds more than 3 points in 16 designs. Report
`results/runs/dc_c5.json`.

## The design

**Arms** (`loop/search.py`), all from the stock chip, same budget, seeds and fidelity:

- `bo` — random forest + expected improvement, one design per round.
- `pooled_bo` — the identical search, started from the design the memory hands over.
- `council` (`loop/council.py`) — four Gemini specialists, one per concern: prefetch (3 knobs),
  geometry (6), replacement (2), concurrency (2). One concern moves per round; the measured design
  is the incumbent with only that concern's knobs changed, so every measurement is attributable.
  The concerns rotate, offset by the seed; a concern whose design became the incumbent leads again;
  a hold costs no design and passes the turn; when all four hold on one design the search stops.
- `council_stock` — the identical council, started from the stock chip. One variable differs from
  `council`, and it is the memory, exactly as `bo` differs from `pooled_bo`.

**BUDGET counts designs, not rounds.** The unit of cost is the design; the unit of adaptation is
the round. Every arm measures one design per round. A design already in the tables is read from
them rather than simulated, which changes the wall clock and nothing else: it is still one design,
one index and one round. A design the simulator cannot measure is dropped and does not consume the
budget, so the round runs again.

**What a specialist sees** (`council.specialist`): its principles (textbook cache architecture,
what a knob does, never which value); the chip and objective; its knobs and values, for prefetch a
placement table with one factual line per mechanism, for geometry the capacity-to-latency ladder;
the incumbent's per-level report for the levels it owns (shape, hit latency, per workload mpki,
hit ratio, prefetch coverage and accuracy, miss latency); the last design if it lost, with what
moved; a ledger of what each assignment of its knobs has scored, with how many other knobs
differed; the recent designs; the task. Nothing names a workload, a trace or a chip. A proposal must
be its own knobs with allowed values, else one retry, else it counts as a hold.

**What the council runs taught.** Geometry finds the L2 shrink in 2x steps, so momentum is what
makes it pay, and the streak dies on a tie: on `dc2` two of three seeds stopped at 256x16 when 256x8
scored 0.0001 lower. Prefetch made 39 moves over the six c5 seeds and won two, both worth under
0.001 IPC: it places spp_dev or ip_stride in nearly every proposal, changes two or three levels at
once, and never proposed the transpose (next_line at L1D, va_ampm_lite at L2), which is the last
1.2 points on `dc2` and, together with the shrink, most of the remaining 7 on `dc`. Two recurring
losing geometry moves: L1D to 32x12 and LLC halved, one design each in nearly every seed. A
diagnosis variant that chose the leader by reading the whole report was worse (97 % mean), because
it named prefetch every other round; it was removed.

**The memory** (`results/memory.json`, built by `loop/memory.py`): one design plus the receipt.
Built in two stages over groups of memory workloads — **search**, one LLM search per group;
**confirm**, each group's best 20 designs measured on every memory workload. A design may be handed
over only if measured on all but one of them. `handover()` is the whole interface: 13 knob values
fitted to the area budget.

**Search space** (`loop/space.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from
size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC
sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost
nothing (known simplification). Chip C: wide core, two memory channels. Fidelity 5M warmup / 10M
simulated (Spearman 0.919 against 50M/50M over 26 designs).

**Scoring** (`search score`): per arm, the share of the stock-to-best-known gap after N designs,
mean over seeds with min..max below. Best known = the best design measured on every workload of the
suite in the cached tables, so it moves as searches land.

**The ceiling must stay an independent mechanism.** `search ceiling` is the same forest run long
from the stock chip. Never measure a ratio against the ceiling run (warm-up 10, batch 6); measure
against the arm's own configuration run long — `results/baseline/<cell>_s<seed>.json`, 75 designs,
3 seeds a cell.

## Cells and workloads

| cell | test | role |
|---|---|---|
| `dc` | sierra.a.4, merced, tahoe (Google datacenter, DPC4 `gtrace_v2`) | the headline |
| `dc2` | whiskey, bravo, delta | the confirmation, on traces that shaped nothing |
| `smoke` | mcf, lbm, one round | the gate before any launch |

The memory remembers mcf, omnetpp, lbm, bfs.urand, pr.urand, bfs.kron and nothing else. All six
datacenter traces are held out from the memory in the same way; `dc2` is the three we did not pick
to headline, so it is the control for **our** selection. On every datacenter trace a replacement
policy alone hurts or does nothing while the best designs contain one: the policy pays only after a
prefetcher and a bigger LLC. **Do not merge the six into one suite**: a merged ceiling would rest on
the few designs measured on all six and every share would inflate.

## What is in `results/`

```
memory.json                    the handover and its receipt
tables/<trace>.json            the simulation cache — the dataset, and the denominator `score` uses
runs/<cell>_<tag>.json         a cell's report: every design; dc2_g1, dc_g1 (the forests), dc2_c5b, dc_c5 (the councils)
baseline/<cell>_s<seed>.json   the `bo` arm run to 75 designs, behind the designs-to-level numbers
*.log                          transcripts, gitignored; the council's are one file per (arm, seed)
llm_usage.json                 running token cost (gitignored: per machine)
```

**The simulation cache.** One table per workload, keyed by `space.config_name(knobs)`. Every batch
reads the table before simulating and appends under a file lock with an atomic rename, so parallel
runs share one cache and `workloads merge` folds in another machine's. 2464 designs and 8585
measurements. A row carries `metrics_version`; a row from an older simulator is a cache miss and is
re-simulated, so no report has holes. A design the simulator cannot measure is cached as
`"metrics": null` and excluded; to retry one, delete its row. `champsim_bin/` is the same idea one
level down: compiled binaries under the same names.

## Next, in this order

1. **The prefetch specialist.** The one place points remain: it reaches for the most aggressive
   mechanism and never proposes moving a mechanism down a level. **Principle wording is not the
   lever.** Asked 100 times about frozen designs from the tables (the handover and the shrunk
   design, on both suites) under five wordings - coverage-versus-accuracy reading, an attribution
   line against multi-level moves, neutral mechanism descriptions - every answer put ip_stride at
   L1D and spp_dev at L2 or LLC, changed two or three levels at once, and every measured one lost
   by 0.012-0.028 IPC. The transpose loses in both halves alone (next_line at L1D 0.5262, va_ampm_lite
   at L2 0.5324, against 0.5403) and pays only as the pair, so a single-level discipline cannot find
   it either. The geometry specialist is the same: given the stalled L2, five of five asks halve the
   LLC (a 4-point loss) under the current principles and under two more (a tie-is-not-a-verdict line,
   the access-time weighing). What is left is mechanical, not verbal: temperature, several candidates
   a round, or accepting the plateau. The probe builds the specialist's prompt from
   `suite.measured_designs` entries and scores each proposal against the tables; no simulation.
2. **Rebuild the memory under the two-stage procedure.** `results/memory.json` came from a build that
   also ran an anchor stage (`"designs": 163`); the handover is identical, but artifact and code
   should agree.
3. **Push the test workloads further out**, then **an unseen chip**.

## How to run

```bash
SEEDS=1 python -m loop.search smoke s1                          # the gate before any launch, every arm
SEEDS=2 BUDGET=12 python -m loop.search dc2 c5 council          # the council on a cell
SEEDS=5 BUDGET=16 python -m loop.search dc2 h1 [arm,arm]        # a cell
python -m loop.search score results/runs/dc2_c5b.json           # the score table
python -m loop.search ceiling 100 10 <trace> <trace> <trace>    # the independent ceiling
BUDGET=75 SEEDS=3 python -m loop.search dc2 long bo             # a from-scratch curve
python -m loop.memory build results/memory.json <spec traces> -- <gap traces>
python -m loop.workloads merge <fetched results dir>            # another machine's tables into results/
```

Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (designs, default 16), `PARALLEL_RUNS`, `SIM_THREADS`
(concurrency = 3 x this), `CHAMPSIM_BUILD_SHARE` (8 on the VM), `MEMORY_PATH`, `ANALYST_MODEL`
(default `gemini-3.1-pro-preview`, served from `ANALYST_LOCATION=global`; the 2.5 models serve from
`us-central1`), `MB_SEARCH` / `MB_CONFIRM` for the memory build.

Cost of a run: **~110 designs simulated per hour on the VM**, 58 on the Mac. Estimate a launch as
(cells x arms x seeds x budget) / rate, minus the ~20 % of designs already in the cache. c5 and c5b
(2 cells side by side, 2 council arms, 3 seeds, 16 designs) took 2.7 h on the VM and 238 Gemini
calls, 3.44 USD.

**Check `gcloud compute instances list` before anything else: the VM must be stopped whenever
nothing runs on it.** A long run goes on the VM as a detached chain that ends in `shutdown -h now`.

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch   # or spp_dev designs crash
./config.sh champsim_config.json && make -j8 && cd ..
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python numpy scikit-learn google-genai
```

Traces (`traces/`, gitignored): SPEC17 from `https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/`;
GAP from Zenodo record 20043527 with `loop.workloads fetch_gap`; datacenter traces from the DPC4
bucket `https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt` as 100 MB prefixes with
`loop.workloads fetch`.

GCP: project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1` (c2d-standard-32, europe-west4-a,
~1.5 USD/h). Its service account has Vertex access. **The VM has no git checkout**: the live tree is
`/home/mateobuscarons/hackathon` (ssh lands as another user; use `sudo -u mateobuscarons`), code
goes there with `gcloud compute scp loop/*.py`, never `git pull`. On the Mac, `gcloud` and the
Python client use `CLOUDSDK_CONFIG=$HOME/.config/gcloud-chia`.

Budget: trial credits — the GCP console is the source of truth. The binding limit is the global
`CPUS_ALL_REGIONS` quota of 32.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only** — they describe the
mechanism and the measurements, do not speculate about how the work will be received, and name no
one outside the technical references. Every launch needs the user's explicit OK with cost and time.
The Mac runs under `caffeinate` (which stops idle sleep, **not** lid-close sleep). The VM is stopped
when idle. Kill by process group or exact pid, never a broad `pkill`. Code stays lean, one
mechanism, no version archaeology; domain decisions are surfaced with options and left to the user.
Deleted material is in git history, never in the tree.
