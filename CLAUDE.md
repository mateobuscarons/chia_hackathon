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
   with one trace per core, CACTI in cache mode, the `chia/dse/` package (space, store, report
   tool, harness), the Vertex generation config, the `gs://` resolver, the spp_dev fix; the loop
   profiled end to end with CHIA's profiler, and one cell run with and without CHIA.
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
  no evidence) read it and propose at once, one move each on their own knobs, on one level (a
  proposal across levels is asked once more for one level, then taken as proposed). Every proposal
  is sketched at the probe rung (`PROBE_WARMUP` / `PROBE_SIM`, else the run's fidelity): the
  incumbent, each proposal alone on it, all together, one wave of parallel simulations; the
  analyst composes the round's design from the proposals with the sketches in front of it,
  choosing among them and altering no value; a composed design the wave did not sketch is
  measured then; a design the sketches predict to lose is not measured; a part the simulator
  cannot measure is out of the round and on the refused list, never a zero in the ledger. Every
  design evaluated is a design of the run and counts against the budget; the council reads and
  stands on the composed designs, and the incumbent is the best of them; the
  sketches are in the sketch ledgers and, in a jump round, in the neighbourhood list. Round 1 is
  the same with the analyst's
  opening design from the stock report, split by concern. When every specialist holds, the
  analyst takes the turn with one design of its own, split by concern like the opening. Two
  rounds in a row that do not move the incumbent make the next round a jump round (below). The
  search stops when the budget is spent or after three rounds per design of budget; a stalled
  search keeps jumping, its refused list growing. Transcript: `results/<arm>-<tag>-s<seed>.log`,
  every prompt and answer.
- **The jump round** (`STALL_ROUNDS`, `FLAT_SHARE` in `council.py`): the council only climbs, so
  it stalls in a local optimum with budget left. After two rounds that do not move the incumbent
  the prompts change with the state: everyone reads the moves the sketches refused against the
  incumbent (a move refused once is a hold when re-proposed, and costs no sketch) and, per knob,
  the values never measured as a one-knob change from it; each specialist proposes the untried
  one-knob move it expects most, alone, so the sketch is attributable, the blamed levels first
  but any of its levels; only a concern whose neighbourhood is exhausted proposes the coupled
  move that puts its levels in a different regime; they hold only by naming the measurement that
  would make them move; the analyst's sheet adds the counterfactual bottleneck, the level named
  least whose evidence would motivate a different design, and it composes from the parts that
  gained, never carrying a part the sketches refused unless all the parts together are not a
  loss (`FLAT_SHARE` of the incumbent is the tolerance). A jump the sketches do not predict to
  gain is not measured; the next round jumps again with a smaller neighbourhood and sweeps the
  level the jump rounds against this design have blamed and sketched least (`sweep_level`), so a
  stall does not spend its budget on the levels the sheet keeps naming. The stall counter is
  TuRBO's; the two-round trigger is the criterion of AutoTuring's critic (arXiv 2609.19387).

**BUDGET counts every design the run evaluates, sketches included.** The unit of cost is the
design; the unit of time is the round, one wave of up to five sketches in parallel plus the
composed design when the wave did not measure it. A design already in the tables is read from them
rather than simulated, which changes the wall clock and nothing else; a design evaluated twice in
a run counts once. A design the simulator cannot measure is dropped and does not consume the
budget. The forest arm measures `BO_BATCH` designs a round, five for the same parallelism, so
the two arms are compared at equal designs.

**What a specialist sees** (`council.specialist`): its principles (textbook cache architecture,
what a knob does, never which value); the chip and objective; its knobs and values, for prefetch a
placement table with one factual line per mechanism, for geometry the capacity-to-latency ladder;
the sheet; the incumbent's per-level report for the levels it owns (shape, hit latency, per
workload mpki, hit ratio, prefetch coverage and accuracy, miss latency) and the view derived from
the counters; its concern's moves with their deltas and sketches; what each value of its knobs has
done; the recent designs; the task. In a jump round it also reads the caveat that knobs moved
together are not separated in that ledger, and the values of its knobs never measured as a
one-knob change from the current design, within the area budget: the neighbourhood the run has
not looked at. Nothing names a workload, a trace or a chip. A proposal must be its own knobs with
allowed values, else one retry, else it counts as a hold. A climb round reads nothing more; only
a jump round adds the stall section and the neighbourhood list.

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

**The `llama2` cell at 1M/2M** (stock 0.4731, best known 1.4710 over about 800 designs in the
table): the council, 2 seeds x 60 designs, reaches 1.4706 in both seeds, 99.96 % of the gap, on
the same design by different routes: va_ampm_lite at the L1D and L2, spp_dev and SHiP at the LLC,
32 LLC MSHRs, stock geometry, LRU at L2. One seed has 96 % at round 5 and 99.96 % at round 13,
the other 96 % at round 9 and 99.96 % at round 15; 21 and 18 rounds, 63 and 61 designs, 71 minutes
on the Mac. Of the run's thirteen jump rounds four won, every one a single knob the climb had
refused or never reached (32 LLC MSHRs alone, a 12-way L1D with LRU back at L2); the empty ones
came after 1.4706 or on one-notch MSHR values. Report `results/runs/llama2_j8.json`, transcripts
`results/council-j8-s{0,1}.log`. The same climb without the jump round stops at 96 % in both
seeds with the budget unspent.

**What the last 4 % was, and why the climb alone does not find it.** Between the 96 % design and
the best known, 32 LLC MSHRs alone is +0.0389; the other knobs are within -0.005 alone and in
every combination without it. The two MSHR knobs are coupled: 16 L2 MSHRs is +0.0025 on the 96 %
design and -0.026 once the LLC has 32, so a specialist that moves its two knobs as a pair lands
at 1.4373 and reads 16 as the better L2 value. Moves pair across concerns too: va_ampm_lite at L2
is -0.006 on an LRU hierarchy and +0.047 behind SHiP at the LLC, so whichever specialist speaks
first decides whether the pair is found, and a sketch refused once is read as settled. Hence the
rules the mechanism rests on: one knob on one level per proposal, so every sketch is attributable;
the incumbent and the ledgers stand on the composed designs, the sketches count and feed the
neighbourhood list; a stall sweeps the values never measured as a one-knob change from the
incumbent, and rotates to the level the sheet blames least when a sweep finds nothing, because
the sheet blames the levels that miss most while the gains sit at the LLC.

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
- **Case memory (postponed to after the submission; the design stays here)**: written after every measured design: SoC descriptor, bottleneck sheet,
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
60 designs = 340 designs. Measured rate at 5M/10M: ~21 aiml designs an hour on the VM for a
single-core SoC (the 100-design ceiling took 4.8 h); a quad design is ~2.7x with two mixes, ~4x
with three. At 1M/2M, the fidelity of record on chip C, the rate is measured on the first suite
chain (item 3 below) before the grid's fidelity is fixed.

```
SoC        ceiling   designs   VM hours at 5M/10M
server        100       340       ~16
mobile        100       340       ~16
quad          100       340       ~44 (two mixes)   ~65 (three)
total                  1020       ~76               ~97
```

Three VMs in parallel, if the quota allows, bring the wall clock to the quad's. Gemini: about
seven calls a round (sheet, four specialists, follow-ups, synthesis) at 0.015 USD a call, about
0.1 USD a round. The quad's ceiling may be cut to 60 if the schedule needs it; the docs then say
so where the number is used.

**What changes in `loop/`**: `simulate.py` keeps the config rendering and reads latencies from
CACTI; `suite.py` becomes the store and the harness's problem; `analyst.py` becomes
`VertexGeminiLLM` with a generation config; `space.py` becomes the `chia.dse` space with this loop's knobs;
`council.py` gains the tool (the cases are postponed); `search.py` keeps the arms and calls the
harness.
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

The mechanism is fixed (the j8 loop, above). What follows is the baseline, the suite, the SoCs,
the CHIA wiring with the loop profiled and compared with and without CHIA, the grid and the
paper. Simulation hours are the critical path: every step that needs compute is one detached
chain, and the desk work runs while it does. Times are from the measured rates above.

1. **Close the `llama2` cell with its baseline** (Mac, no new compute if run `q5` is used).
   - The forest at 1M/2M, 2 seeds, `BO_BATCH=5`, from the stock chip. The other session's run
     `q5` (2 seeds x 500 designs, batch 5, the same fidelity) is in flight; its first 60 designs
     per seed are this baseline and its later designs grow the table the best known is read
     from. If `q5` is stopped early, run `bo` at 2 x 60 (about 40 minutes).
   - `search score` gains a by-round view: for the council a round is a wave plus the composed
     design (the report has the round of every design), for the forest a round is one batch of
     `BO_BATCH`; the table reads share of the gap after N rounds, mean over seeds, min..max, and
     rounds-to-level (90, 96, 99 %). Designs stay as the second axis.
   - `search cell <cell> <tag>`: one command that reads the tables, runs the ceiling when no
     100-design forest exists at the run's fidelity on every workload of the cell, then both arms
     at 2 x 60 with the seeds given; it is the chain every later cell uses, on the Mac and on the
     VM.
   - Output: the llama2 table by rounds (council, forest, best known) and the two seeds' final
     designs, both for the paper.
2. **Implement feedback** on the loop and the `llama2` result before the suite runs.
3. **Validate on the suite: the `aiml` cell at 1M/2M on chip C** (Mac, one overnight chain of
   about 8 hours, or 3 on the VM).
   - The rank correlation of 1M/2M against 5M/10M on the suite, from the designs present in both
     tables (about 200 at 5M/10M, about 280 at 1M/2M per workload), the justification for 1M/2M
     as the fidelity of record on chip C beside llama2's rho 0.995.
   - The chain: a council-only smoke (about 2 minutes), the ceiling if the denominator is missing
     (100 forest designs, about 1.5 hours on the Mac), council 2 x 60 (about 5 hours on the Mac:
     a wave waits for its slowest simulation, clip at 6 minutes on a prefetcher-heavy design),
     forest 2 x 60 with `BO_BATCH=5` (about 1.5 hours).
   - Output: the aiml table by rounds, the two final designs, and the measured 1M/2M rate on the
     suite (designs an hour, CPU-minutes per design by design class) that sizes the grid and
     fixes its fidelity.
4. **The SoCs: profiles, CACTI, multi-core, and what the council reads on them** (desk work; the
   only compute is the smoke cell on each SoC, minutes).
   - `loop/socs.py` from git history: one profile per SoC with core width and ROB, frequency,
     memory channels and data rate, core count, process node, area budget. **server** is chip C
     (4 GHz, 32 nm, 4608 KB); **mobile** is A_mobile (narrow core, ROB 128, DDR-1600, one channel,
     2 GHz, 45 nm, 2048 KB); **quad** is D_quad (four stock cores with a deeper ROB, private
     L1/L2, one shared LLC, two channels, 32 nm, 8192 KB).
   - CACTI in cache mode (`CHIA_BLOCKS.md` §7): per level and SoC, (sets, ways, 64-byte block,
     node) to access time, read energy, leakage and area; cycles = ceil(access time x frequency)
     written into the config in place of the derived formula; L1D in parallel mode, L2 and LLC in
     sequential mode; the 12-way L1D interpolated on log capacity between 8 and 16 ways; results
     cached by key; area in mm² reported beside the KB budget. CACTI 7 is built locally and the
     stock ladders are read (The redesign). CACTI latency changes chip C's stock shape (7/6/11
     cycles against the formula's 4/9/15), so the server's grid tables are rebuilt; the 1M/2M
     tables in `results/tables/` remain the record of the mechanism's development.
   - Multi-core in `simulate.py` and the ChampSim node: one trace per core; the quad's two
     mixes fixed and written down (which of the three traces sits on which core); a quad design
     is the two mixes measured, each a four-core simulation. The objective on the quad is a
     decision to take before the smoke: the sum or the geometric mean of the four cores' IPC; the
     denominator follows it.
   - The report per core in the prompts: a private level's line carries the mean over cores and
     the spread, the shared LLC one line; one principle each for geometry and concurrency on a
     shared level, that its misses are the sum over cores, its capacity per core is the size
     divided by the cores, and prefetch traffic from every core contends for the same channels.
     Textbook, valid on any SoC, and the only prompt change for the quad.
   - The smoke cell (mcf, lbm, one round) on all three SoCs, the quad with a four-core mix.
5. **The CHIA wiring: the blocks as code, the loop in CHIA layout, the novelty stated** (desk
   work, no compute; `CHIA_BLOCKS.md` holds the per-block design, tests and docs).
   - Block 1, `chia/simulators/champsim.py`: the configuration-space build from a config dict and
     a run that takes one trace per core, the tree pool as the node's resource, the binary cache
     as CHIA's cache. Tier 0 tests on the config written and the command line built.
   - Block 2, `chia/dse/`: `space.py` (this loop's knobs and budget, sampling deterministic under
     a seed), `store.py` (the measurement store on `SQLiteNode`, keyed by SoC and design name,
     the file tables' read-before-simulate and append semantics), `report_tool.py` (the per-level
     report as a `ChiaTool`), `harness.py` (`run_cell`, `score`, `ceiling`, designs-to-level and
     rounds-to-level as the harness's own metrics). The cases module is postponed.
   - Block 3, `chia/models/vertex.py`: the generation config that reaches Gemini, thinking
     tokens, the opt-in 429 backoff, with the mocked-loop tests.
   - Block 7, CACTI cache mode, from item 4.
   - The loop in `chia/examples/*` layout: one loop file, `cluster.yaml`, `README.md`; one switch
     between local calls and `chia_remote`, so the same loop runs on the Mac without Ray and on
     the cluster with it. `cluster.yaml`: a `champsim_build` worker, N `champsim` workers on the
     pinned ChampSim image with the spp_dev patch, a `cacti` worker, a `vertex_creds` worker.
   - The novelty, stated as what the code is: the council loop's blocks reusable by any CHIA DSE
     loop, the sketch wave (parts alone and together in one parallel call, deltas and interaction
     back) and the stall-and-sweep jump round (stall counter, tabu, neighbourhood list, level
     rotation, guard) as two nodes with the design space as their only interface; the store as
     the cache every arm and loop shares; designs-to-level built into the harness, which the CHIA
     paper names as open ("metrics should be built into agentic platforms").
6. **Profile the whole loop with CHIA** (desk work plus one llama2 run, about 70 minutes on the
   Mac with Ray local).
   - `chia.trace.profiler.start_collector` on the driver; every `ChiaFunction` is instrumented
     with no code change: the builds, each simulation, each CACTI call, each model call, each
     store read and write, with worker, execution time and the dependency edges between them.
   - `add_info` on the loop's nodes: tokens and cost on the model call (CHIA's Vertex node
     records them), simulated instructions and the design's name on a run, hit or miss on a
     store read; `log_event` once per round with the round number, the mode (climb or jump, the
     swept level), the stall count, the designs so far and the composed design's delta, so the
     timeline is readable by round.
   - `chia viz-profile` renders the dependency graph (the wave's fan-out into the composition),
     the interactive timeline (per round: model time, simulation time, the slowest sketch as the
     critical path, idle cores), and the CSV table (per run and per arm: simulation seconds, model
     seconds, cache hits, cost per round).
   - Output: one profile log per cell in `results/`, one timeline figure and one table of where
     the wall clock goes for the paper; the same numbers say whether a cheaper sketch rung
     (`PROBE_WARMUP` / `PROBE_SIM`) would shorten the round.
7. **The same cell with CHIA and without** (one `llama2` council run per path, about 70 minutes
   each on the Mac; the forest arm the same).
   - Without: today's path, ChampSim as a subprocess, a thread pool, JSON tables under a file
     lock, no Ray. With: the nodes on Ray workers, the `SQLiteNode` store, `cache`/`bypass`, the
     profiler on. Same seeds, same budget, same fidelity.
   - Measured and tabled: the designs measured and their IPC agree (the tables are the check);
     wall clock per round and per run; CPU utilisation over the run from the timeline; lines of
     plumbing in each path (build, run, store, parallelism); a round replayed from the log with
     `bypass`, without model calls; what one path has and the other does not (the profile, the
     replay, the grid across machines from `cluster.yaml`).
   - Output: the with-and-without table and the replayed round, for the paper's blocks section.
8. **The gate**: the smoke cell through the local path and through `chia_remote` on a local Ray
   cluster with the profiler on, the Gemini tool loop exercised and the precomputed-views fallback
   ready; then the grid launcher dry-run, one design per SoC as a detached chain ending in the
   VM's shutdown. No ceiling on a new chip runs on the Mac.
9. **The grid** on the cluster: server, mobile, quad; per SoC the `search cell` chain, the ceiling
   and both arms at 2 x 60, at the fidelity fixed in item 3, the profiler on. At 1M/2M, from the
   Mac's measured rates and the VM's throughput of about twice the Mac's: about 3 hours for
   server, 3 for mobile, 10 for the quad with two mixes (a four-core simulation is one process,
   so a council round waits about 24 minutes for its slowest mix; the forest and the ceiling run
   beside it). The global `CPUS_ALL_REGIONS` quota is 32 vCPUs: one 32-vCPU VM in sequence, about
   16 hours, or three VMs sharing the 32, the quad's council wave of 20 simulations sizing its VM.
   Each chain ends in the VM's shutdown.
10. **The paper and the README**: per SoC the table by rounds (council, forest, best known) and
    the final designs; the mechanism (the sheet, the team round with sketches, the jump round) with
    the llama2 record as the worked example; the with-and-without-CHIA table and the profile
    figure; the blocks; the limits stated as measured (two seeds, 1M/2M with its rank correlation,
    one suite). The README is written once, from these tables.

**Postponed to after the submission**: the case memory and the cross-SoC transfer it measures,
the ablation arms (without the analyst, without sketches, single agent, all knobs at once), a GP
optimizer as a second tuned baseline, the `gs://` and spp_dev pull requests, any further model
comparison.

## How to run

```bash
SEEDS=1 python -m loop.search smoke s1                                   # the gate before any launch
SEEDS=2 BUDGET=60 LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=5 python -m loop.search llama2 j8 council
SEEDS=2 BUDGET=60 python -m loop.search aiml a1                          # both arms on the suite
LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m loop.search score results/runs/llama2_j8.json
python -m loop.search ceiling 100 10 <trace> <trace> <trace>             # the independent ceiling
python -m loop.workloads fetch <url> <out> [prefix_mb]                   # a trace, or its prefix
```

A run goes detached under `caffeinate -i` with `nohup`, its stdout to `results/run_<tag>.log`.

Env: `SEEDS`, `FIRST_SEED`, `BUDGET` (designs, default 60), `PARALLEL_RUNS`, `LOOP_WARMUP` /
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
1M/2M a llama2 council round is ~2 to 4 min with two seeds sharing the machine. At 1M/2M, one
simulation per trace on the idle Mac: the stock design takes 0.7 min on llama2, 0.3 on clip and
0.2 on stable-diffusion, 1.2 CPU-minutes per aiml design; the design the council ends on
(prefetchers at every level, spp_dev at the LLC) takes 4.5, 6.1 and 1.4, 12 CPU-minutes. Gemini
3.1 Pro is ~10 s a call. The VM does ~110 designs an hour at 5M/10M, the Mac 58.

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
