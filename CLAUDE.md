# Few-shot cache tuning by a council of LLM specialists

Cache-hierarchy design-space exploration on ChampSim. The question: **given a stock chip and a
workload suite the loop has never seen, how many simulations does it take to get within x % of the
best design known?** Designs-to-level, not level-at-N.

Prior work it is measured against: ArchAgent (arXiv 2602.22425), an agent writing cache policies
inside ChampSim, every search from scratch; ArchGym ("all optimizers tie under tuned
hyperparameters"); DOSA; AgentDSE (MLArchSys @ ISCA 2026, arXiv 2606.21836), a coding agent tuning
a ChampSim hierarchy in ~50 simulations, no transfer.

**Contributions back to the frameworks live in `CHIA_BLOCKS.md`.** This file is state, design and
operations. The README is written once the results are in.

## The hackathon

Agentic Architecture hackathon (agentic-arch.org), sponsored by Google, NVIDIA and IEEE TCMM,
built on CHIA (SLICE lab, UC Berkeley; docs.chialoops.ai, arXiv 2606.27350). Its mission, quoted:
"Invent and share creative agentic architecture loops" made "accessible to the community as
composable building blocks integrated into the open-source CHIA framework." Deliverables: a 4-page
two-column paper and the loop open-sourced with reproducible results. Judged by a human program
committee on "author-identified highlights, paper submission, artifact, and their judgment"; AI
reviewers with personas comment but do not rank. Tracks this project answers, quoted:

- "Cross-SoC optimization of cache hierarchies" (the named track; not yet done here)
- "Reusable CHIA blocks for agentic design workflows that run on NVIDIA-accelerated infrastructure
  and can be reused after the hackathon"
- "Agentic flows for detailed block-level microarchitecture breakdowns/analyses"

What the CHIA paper itself names as open: "We need principled methods to evaluate these loops.
Metrics should be built into agentic platforms"; reward hacking of simulators; agents as "a piece
in a larger agentic flow"; the "largely untested design space of agentic discovery flows".
CHIA's own ChampSim node (`chia/simulators/champsim.py`) builds from a prefetcher source file and
runs one trace; it has no configuration-space build, no multi-trace run, no design space, no
measurement store and no memory across loops.

**What the contribution is.** The council: an analyst and four LLM specialists that read the
simulator's per-level reports and move in team rounds, every move sketched cheaply before one
design a round is committed, so every counted measurement is attributable to a decision taken on
measured evidence. Cross-SoC means the same loop, unchanged, on several SoCs, each from its stock
chip, against a tuned optimizer at the same budget. What the council would carry between
searches is a case memory of its own moves and their outcomes (see "The redesign"); the first
version of this repo handed over one design instead, and its code and record are in git history.

**What the submission is built from, in priority order.**

1. **The loop on three SoCs**, below, against the forest at the same budget. No ablation arms;
   those come last, if time remains.
2. **CHIA blocks as code, submitted** (`CHIA_BLOCKS.md`): the ChampSim configuration-space node
   with one trace per core, CACTI in cache mode, the `chia/dse/` package (space, store, cases,
   report tool, harness), the Vertex generation config, the `gs://` resolver, the spp_dev fix.
3. **The paper**: the per-SoC tables, the council's mechanism, the blocks.

## The loop as it runs

Two arms (`loop/search.py`), both from the stock chip, same budget, seeds and fidelity:

- `bo` — random forest + expected improvement, one design per round.
- `council` (`loop/council.py`) — the analyst reads every design measured and writes the sheet:
  a verdict per level with its number, the levels ranked by misses per kilo-instruction so the
  level that misses most is never "fine", the bottleneck, each finding with its delta, design ids
  and the counter behind it, what several knobs moving together make unattributable, which knobs
  fail by their own counters, a prediction checked against the next measurement. All four
  specialists (prefetch, geometry, replacement, concurrency; own knobs, allowed values, hold on
  no evidence) read it and propose at once, one move on one level each. Every proposal is
  sketched at the probe rung (`PROBE_WARMUP` / `PROBE_SIM`, else the run's fidelity): the
  incumbent, each proposal alone on it, all together, measured at once and counted for nothing;
  the analyst composes the round's design from the proposals with the sketches in front of it,
  choosing among them and altering no value; a composed design the wave did not sketch is
  sketched before it is committed; a design the sketches predict to lose is vetoed and costs
  nothing; the composed design is measured at the run's fidelity and is the round's one counted
  design. Round 1 is the same with the analyst's opening design from the stock report, split by
  concern. When every specialist holds, the analyst takes the turn with one design of its own,
  split by concern like the opening. The incumbent is the best design measured. The search stops
  when the analyst too has nothing (or two of its turns in a row commit nothing), when the budget
  is spent, or after three rounds per design of budget. When the probe rung is the run's fidelity
  the composed design's measurement is already in the table from its sketch, so a round costs
  one probe wave. Transcript: `results/<arm>-<tag>-s<seed>.log`, every prompt and answer.

**BUDGET counts designs, not rounds.** The unit of cost is the design; the unit of adaptation is
the round. A design already in the tables is read from them rather than simulated, which changes
the wall clock and nothing else. A design the simulator cannot measure is dropped and does not
consume the budget. Sketches are extra simulations that count for nothing; a run's report keeps
every sketch, so simulations per counted design can be read off it.

**What a specialist sees** (`council.specialist`): its principles (textbook cache architecture,
what a knob does, never which value); the chip and objective; its knobs and values, for prefetch a
placement table with one factual line per mechanism, for geometry the capacity-to-latency ladder;
the sheet; the incumbent's per-level report for the levels it owns (shape, hit latency, per
workload mpki, hit ratio, prefetch coverage and accuracy, miss latency) and the view derived from
the counters; its concern's moves with their deltas and sketches; what each value of its knobs has
done; the recent designs; the task. Nothing names a workload, a trace or a chip. A proposal must
be its own knobs with allowed values on one level, else one retry, else it counts as a hold.

**The model** (`loop/analyst.py`): Gemini through `google-genai`, one model per process
(`ANALYST_MODEL`, default 3.1 Pro): JSON mode, a thinking budget, 429 backoff. Cost per call is
logged to `results/llm_usage.json` from the price table in the module. gpt-oss-120b, DeepSeek V3.2
and Kimi K2 Thinking were run through Vertex's OpenAI-compatible endpoint on the same prompts:
gpt-oss opened poorly and committed little, Kimi was throttled and slow, DeepSeek climbed slower
than Gemini; that backend is in git history.

**Search space** (`loop/space.py`, 13 knobs, 6.6 M raw designs, area-coupled, latency derived from
size): L1D sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC
sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity <= 4608 KB; L1D and MSHRs cost
nothing (known simplification). Chip C: wide core, two memory channels. Fidelity 5M warmup / 10M
simulated by default (Spearman 0.919 against 50M/50M over 26 designs).

**Scoring** (`search score`): per arm, the share of the stock-to-best-known gap after N designs,
mean over seeds with min..max below. Best known = the best design measured on every workload of the
suite in the cached tables at the run's fidelity, so it moves as searches land.

**The ceiling must stay an independent mechanism.** `search ceiling` is the same forest run long
from the stock chip (warm-up 10, batch 10). Scoring against a design an arm found bounds the
metric at that arm.

## State

Development runs on the `llama2` cell at 1M/2M (stock 0.4731, best known 1.4697): before the
analyst's turn and the verdict rule, two seeds x 20 designs reached 96.2 and 91.7 % of the gap and
stopped at D8 and D7 on the old all-hold rule with the budget unspent; the level every sheet
called "fine" was the L1D, which misses 81 times per thousand instructions. On the same prompt
six models proposed the same wrong level, so the fix is the sheet and the stop rule, not the
model. With the fixes, Gemini reaches 96.2 % at D5 in both seeds and then stops: the analyst's
turns propose one-knob moves the sketches refuse. The best known design differs from the 96 %
design in four knobs (L1D 12 ways, DRRIP at L2, SRRIP at the LLC, 32 LLC MSHRs); every design
above 1.45 in the table has 32 LLC MSHRs, and the concurrency specialist has never proposed a
move.

**Open item.** The council only climbs: the incumbent is the best design measured, a losing
design is never accepted as the next base, and the veto refuses a design the sketches predict to
lose. It cannot cross a valley to a better basin, so it can sit in a local optimum with budget
left. Not addressed.

## Cells and workloads

| cell | test | role |
|---|---|---|
| `aiml` | llama2_7b, stable-diffusion, clip (ML inference, DPC4 `ai-ml`, 200 MB prefixes) | the suite |
| `llama2` | llama2_7b alone | development: rho 0.96 with the suite objective over 217 designs |
| `smoke` | mcf, lbm, one round | the gate before any launch |

## The redesign

Everything is a CHIA node or a CHIA tool; the council is the only piece that is ours.

```
soc profile ─► cache_ppa (CACTI) ─► render config ─► build_champsim_config ─► run_champsim × traces
                    │ ladder, energy, area   ▲ knobs                                     │
                    ▼                        │                                           ▼
             analyst ─► report tool (ChiaTool) ◄─ specialist (LLM) ─► store (SQLiteNode)
                                     ▲                          ▲
                                     └────── case memory ───────┘
```

- **SoC profile**: core width and ROB, frequency, memory channels and data rate, core count,
  process node, area budget. One JSON each. Three SoCs: **mobile** (A_mobile from git history:
  narrow, ROB 128, DDR-1600, one channel; 2 GHz, 45 nm, 2048 KB), **server** (chip C as today;
  4 GHz, 32 nm, 4608 KB), **quad** (D_quad: four stock cores with a deeper ROB, private L1/L2,
  one shared LLC, two channels; 32 nm, 8192 KB; a design is measured on mixes of four traces).
- **Cache PPA**: CHIA's `run_cacti` in cache mode (associativity, 64-byte blocks, tag array) per
  level: access time, read energy, leakage, area. Cycles = ceil(access time x frequency), written
  into ChampSim's config in place of the derived formula, so a level's latency depends on the SoC.
  L1D in CACTI's parallel ("normal") mode, L2 and LLC in sequential mode (tag first, then one
  way), where latency follows capacity and ways cost little. CACTI takes only power-of-two
  associativity, so the 12-way L1D is interpolated on log capacity between 8 and 16 ways at the
  same sets. CACTI models the array alone; nothing is added for the fabric. Cached by (size, ways,
  block, node). Area is reported in mm² beside the KB budget. Stock shapes at 32 nm, 4 GHz:
  L1D 64x12 7 cycles, L2 1024x8 6, LLC 2048x16 11 (the formula: 4, 9, 15); at 45 nm, 2 GHz:
  4, 4, 7.
- **Build and run**: the configuration-space node; one trace per core on the quad.
- **Store**: CHIA's `SQLiteNode` on the head: measurements keyed by SoC and design name, and the
  cases table. Its read-only query tool comes with it.
- **Analyst and specialists**: as they run today.
- **Case memory**: written after every measured design: SoC descriptor, bottleneck sheet,
  concern, the move (from, to), delta IPC, delta area and energy. Retrieval is k nearest
  neighbours on a fixed feature vector (per-level hit ratios, coverage, miss latency, budget use,
  core width class); no embeddings. The specialist sees the nearest cases and what each move did.
  A run reads its own cases and those of the SoCs run before it, in the order server, mobile,
  quad; the server starts empty. It never reads the other seed of its own SoC, nor the development
  runs on chip C. Every retrieved case is logged with its source SoC and whether the taken move's
  sign matched it, so the transfer between SoCs is measured per direction without an ablation arm.
  Beacon's (arXiv 2608.30932) memory is scoped to one run; with 16 designs the ledger already
  covers that, so the cases from other SoCs are the new information here.

**Arms**: `council` and `bo`, both from the stock chip, same budget and seeds, per SoC.

**Suites**: `aiml` on every SoC. On the quad a design is measured on mixes of four traces drawn
from the three; two mixes proposed, the third if the budget allows.

**Grid and cost.** CACTI latency changes the chip, so every table is rebuilt; nothing in
`results/tables/` serves the new loop. Per SoC: a ceiling of 100 designs and two arms x 2 seeds x
16 designs = 164 designs. Measured rates: ~21 aiml designs an hour on the VM for a single-core
SoC (the 100-design ceiling took 4.8 h); a quad design is ~2.7x with two mixes, ~4x with three.

```
SoC        ceiling   designs   VM hours
server        100       164       ~8
mobile        100       164       ~8
quad          100       164      ~21 (two mixes)   ~33 (three)
total                   492      ~37               ~49
```

Three VMs in parallel, if the quota allows, bring the wall clock to the quad's. Gemini: about
seven calls a round (sheet, four specialists, follow-ups, synthesis) at 0.015 USD a call. The quad's ceiling may be cut to 60 if the
schedule needs it; the docs then say so where the number is used.

**What changes in `loop/`**: `simulate.py` keeps the config rendering and reads latencies from
CACTI; `suite.py` becomes the store and the harness's problem; `analyst.py` becomes
`VertexGeminiLLM` with a generation config; `space.py` becomes the `chia.dse` space with this loop's knobs;
`council.py` gains the tool and the cases; `search.py` keeps the arms and calls the harness.
`socs.py` returns from git history as the SoC profiles.

**Risks.** CHIA's Vertex backend calls itself experimental and its MCP tool loop is its least
exercised path; the gate runs it first, with the precomputed-views fallback ready.

## What is in `results/`

```
tables/<trace>[_w{W}M_s{S}M].json   the simulation cache, one table per workload and fidelity — the
                                    dataset, and the denominator `score` uses
runs/<cell>_<tag>.json              a cell's report: every design, every round with its sketches
*.log                               transcripts of the current runs, gitignored, one per (arm, seed)
llm_usage.json                      running token cost (gitignored: per machine)
```

**The simulation cache.** One table per workload and fidelity, keyed by `space.config_name(knobs)`.
Every batch reads the table before simulating and appends under a file lock with an atomic
rename, so parallel runs share one cache. A row carries `metrics_version`; a row from an older
simulator is a cache miss and is re-simulated, so no report has holes. A design the simulator
cannot measure is cached as `"metrics": null` and excluded; to retry one, delete its row.
`champsim_bin/` is the same idea one level down: compiled binaries under the same names. The
tables of the first version's suites (datacenter traces, SPEC and GAP) stay as the dataset.

## Next, in this order

Two phases. **On the Mac**, with the Gemini budget in hand, the mechanism is validated where it is
cheapest: chip C, the `llama2` cell at 1M/2M, many designs cache hits. **On the cluster**, the
grid runs once. The gate for the second phase is that the first ends with one command.

On the Mac:

1. **The council on chip C**: the model comparison in flight, then the `aiml` cell at 1M/2M,
   2 seeds x 16 designs, against `bo` at the same budget. Iterate on the analyst, the team round
   and the models until it separates.
2. **Blocks as code** while simulations run (`CHIA_BLOCKS.md`): the ChampSim configuration-space
   node with one trace per core, CACTI cache mode, `chia/dse/`, the Vertex generation config and
   the opt-in 429 backoff, each with its Tier 0 tests. No compute.
3. **New-chip plumbing**: SoC profiles, CACTI latency into the config, one trace per core; the
   smoke cell on all three SoCs. mcf and lbm are ten times cheaper than the ML traces.
4. **The gate** through the local path and through `chia_remote` on a local Ray cluster, then the
   grid launcher dry-run: one design per SoC as a detached chain.

No ceiling on a new chip runs on the Mac; the three 100-design runs are the cluster's.

On the cluster:

5. **The grid**: server, mobile, quad. Three VMs in parallel if the quota allows, the quad's
   ~21 h setting the wall clock; else one VM in that order, ~37 h.
6. **The paper.**
7. Afterwards, if time remains: the `gs://` and spp_dev pull requests, ablation arms (without
   the analyst, without sketches, without cases, single agent, all knobs at once), a GP optimizer
   as a second tuned baseline.

## How to run

```bash
SEEDS=1 python -m loop.search smoke s1                                   # the gate before any launch
SEEDS=2 BUDGET=20 LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=5 python -m loop.search llama2 f1 council
SEEDS=2 BUDGET=16 python -m loop.search aiml a1                          # both arms on the suite
LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m loop.search score results/runs/llama2_e1.json
python -m loop.search ceiling 100 10 <trace> <trace> <trace>             # the independent ceiling
python -m loop.workloads fetch <url> <out> [prefix_mb]                   # a trace, or its prefix
```

A run goes detached under `caffeinate -i` with `nohup`, its stdout to `results/run_<tag>.log`.

Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (designs, default 16), `PARALLEL_RUNS`, `LOOP_WARMUP` /
`LOOP_SIM` (the run's fidelity, own tables; on llama2, 2M/5M and 1M/2M rank designs as 5M/10M
does: rho 0.994 and 0.995 over 38 designs, the same top ten, the same best design, IPC within
0.4 %), `PROBE_WARMUP` / `PROBE_SIM` (the council's sketch rung, else the run's), `SIM_THREADS`
(concurrency = 3 x this per process), `CHAMPSIM_BUILD_SHARE` (8 on the VM), `ANALYST_MODEL`
(default `gemini-3.1-pro-preview`, served from `ANALYST_LOCATION=global`; the 2.5 models serve
from `us-central1`).

**What a design costs, measured on the Mac (11 cores).** A ChampSim build is ~6 s: only the
per-config main object and the link are rebuilt. Simulation time is set by the design's
prefetcher traffic, not by the trace: at 5M/10M, clip takes 1.2 min on the stock binary, llama2
2.7, stable-diffusion 0.6, and a binary with spp_dev at two levels takes four times longer. At
1M/2M a llama2 council round is ~2 to 4 min with two seeds sharing the machine. Gemini 3.1 Pro
is ~10 s a call. The VM does ~110 designs an hour at 5M/10M, the Mac 58.

**Check `gcloud compute instances list` before anything else: the VM must be stopped whenever
nothing runs on it.**

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch   # or spp_dev designs crash
./config.sh champsim_config.json && make -j8 && cd ..
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -r requirements.txt
# CACTI 7, the repo CHIA's cacti image builds. clang needs portable flags and one fix: the Nuca
# constructor's default argument moves from the definition to the declaration.
git clone --depth 1 https://github.com/ucb-bar/cacti.git cacti && cd cacti
sed -i.orig '79s|DeviceType \*dt);|DeviceType *dt = \&(g_tp.peri_global));|' nuca.h
sed -i.orig '46s| = &(g_tp.peri_global)||' nuca.cc && rm nuca.h.orig nuca.cc.orig
make opt -j8 OPT="-O2 -DNTHREADS=8 -Wno-reserved-user-defined-literal" && cd ..   # ./cacti/cacti
```

Traces (`traces/`, gitignored): SPEC17 from `https://dpc3.compas.cs.stonybrook.edu/champsim-traces/speccpu/`;
GAP from Zenodo record 20043527 with `loop.workloads fetch_gap`; the ML-inference and datacenter
traces from the DPC4 bucket `https://pub-c31f67d79d1b4cd28ff320612b1a9f84.r2.dev/manifest.txt`
as 100 or 200 MB prefixes with `loop.workloads fetch`.

GCP: project `project-c23a6080-f5d0-4871-9cb`, VM `champsim-1` (c2d-standard-32, europe-west4-a,
~1.5 USD/h). Its service account has Vertex access. **The VM has no git checkout**: the live tree is
`/home/mateobuscarons/hackathon` (ssh lands as another user; use `sudo -u mateobuscarons`), code
goes there with `gcloud compute scp loop/*.py`, never `git pull`. On the Mac, `gcloud` and the
Python client use `CLOUDSDK_CONFIG=$HOME/.config/gcloud-chia`; REST calls to Vertex with a user
account need the `x-goog-user-project` header. The binding limit is the global `CPUS_ALL_REGIONS`
quota of 32. Budget: trial credits — the GCP console is the source of truth.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only** — they describe the
mechanism and the measurements, do not speculate about how the work will be received, and name no
one outside the technical references. Every launch needs the user's explicit OK with cost and time.
The Mac runs under `caffeinate` (which stops idle sleep, **not** lid-close sleep). The VM is stopped
when idle. Kill by process group or exact pid, never a broad `pkill`. Code stays lean, one
mechanism, no version archaeology; domain decisions are surfaced with options and left to the user.
Deleted material is in git history, never in the tree.
