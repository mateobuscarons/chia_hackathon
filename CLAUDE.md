# Few-shot cache tuning by a council of LLM specialists

Cache-hierarchy design-space exploration on ChampSim. The question: **given a chip and a
workload suite the loop has never seen, how many rounds does it take to get within x % of the
best design known?** Rounds-to-level, not level-at-N: a round is one wave of parallel
simulations, so it is the unit of time; a design is the unit of cost. The cell the paper is built
on adds a cost cap and reads as a product brief (below, "The generation step").

Prior work it is measured against: ArchAgent (arXiv 2602.22425), an agent writing cache policies
inside ChampSim, every search from scratch; ArchGym ("all optimizers tie under tuned
hyperparameters"); DOSA; **AgentDSE** (arXiv 2606.21836), a coding agent tuning a ChampSim
hierarchy in ~50 calls against 1000 for random, genetic and Bayesian - the same knobs as this
loop's old cell, under a 4.5 MiB SRAM cap, three SPEC traces, IPC only, no cost model and no
transfer; **Beacon** (arXiv 2608.30932), multi-agent chiplet DSE with bottleneck diagnosis and a
memory of past runs, so agents-with-memory is not novel on its own and the claim has to be the
brief changing; **ArchEval** (arXiv 2607.03601), the emerging benchmark, which scores constraint
handling.

**Contributions back to the frameworks live in `CHIA_BLOCKS.md`; the design of the generation-step
scenario in `CROSS_SOC_LOOP.md`.** This file is state, design and operations. The README is
written once the results are in.

## The hackathon

Agentic Architecture hackathon (agentic-arch.org), sponsored by Google, NVIDIA and IEEE TCMM,
built on CHIA (SLICE lab, UC Berkeley; docs.chialoops.ai, arXiv 2606.27350). Mission, quoted:
"Invent and share creative agentic architecture loops" made "accessible to the community as
composable building blocks integrated into the open-source CHIA framework." Deliverables: a 4-page
two-column paper and the loop open-sourced with reproducible results. Judged by a human program
committee on "author-identified highlights, paper submission, artifact, and their judgment".
Tracks this project answers: "Cross-SoC optimization of cache hierarchies", "Power optimization
across the stack", "Reusable CHIA blocks for agentic design workflows".

What the CHIA paper names as open: "metrics should be built into agentic platforms"; reward
hacking of simulators; agents as "a piece in a larger agentic flow". CHIA's own ChampSim node
(`chia/simulators/champsim.py`) builds from a prefetcher source file and runs one trace; it has no
configuration-space build, no multi-trace run, no design space, no cost model, no measurement
store and no memory across loops.

**What the contribution is.** The council: an analyst and four LLM specialists that read the
simulator's per-level reports and move in team rounds, every move sketched cheaply before one
design a round is committed, so every counted measurement is attributable to a decision taken on
measured evidence. Cross-SoC means the same loop, unchanged, on several SoCs, each from its stock
chip, against a baseline at the same budget - and **the chip enters the prompts only as derived
numbers, never as a name or a branch**.

## The rule the cross-SoC work rests on

Three sources of chip fact, each derived once, nothing hardcoded per SoC:

| source | kind | what it yields |
|---|---|---|
| the SoC profile (`loop/socs.py`) | declarative | the chip card (`loop/chip.py`), the caps |
| the simulator's counters | measured | the constraint view and the power split in the prompts |
| the design space ∩ the caps | derived | the shapes this chip can actually build |

**The design space is derived too** (`space.shape_ladder`): sizes and associativity are a ladder
from the chip's own stock shape, /4 to x4, cut to what CACTI will build and what the caps leave
standing. Which prefetcher, which replacement policy and how deep the miss queue stay written
down, because they are the simulator's menu, not the chip's.

`loop/socs.py` is the only file in `loop/` that names a chip. Adding an SoC is one entry there and
no value list.
Verify with: `grep -rn "server\|mobile\|quad\|prev\|next" loop/*.py | grep -v socs.py` - only
generic English uses of the words.

**The chip card** (`loop/chip.py`, `loop/council.py`): core width, ROB, queues, clock, cores,
node; channels x rate = B per core-cycle and the DRAM service cost in cycles; the levels and
whether private or shared; the budget line (KB, or mm² and W on a capped chip). Per workload, from
the counters: memory time by tier (AMAT-style, each level's own hits, net of merges), off-chip
traffic against what the channels deliver, the measured miss cost against an idle DRAM, row-buffer
hit rate, and on a capped chip the power split by level and off chip. **The bottleneck is ranked
by share of memory time, not by mpki**; the mpki rule blamed the L1D in ~60 % of sheets on every
chip.

**Cache latency, area and energy come from CACTI** in cache mode at the SoC's node and clock:
`cycles = ceil(access_time_ns x GHz)`, L1D parallel, L2/LLC sequential, 64 B block, non-power-of-two
ways interpolated on log ways. Five fields per shape: access ns, mm², read nJ, write nJ, leakage
mW, cached in `results/cacti_<node>nm.json` (a two-field entry from before energy was read is
characterised again on demand). A machine without a CACTI build only needs that file. On a 2 GHz
part capacity is nearly free in cycles and twice the silicon; associativity costs cycles and
energy per access; capacity costs leakage.

## The loop as it runs

Arms (`loop/search.py`), all from the stock chip, same budget, seeds and fidelity:

- `council` (`loop/council.py`) - the analyst reads every design measured and writes the sheet
  (verdict per level with its number, levels ranked by memory time, bottleneck, findings with
  deltas and design ids, unattributable moves, failing knobs, a prediction; on a capped chip
  `over_budget`: the cost nearest its cap and its driver). Four specialists (prefetch, geometry,
  replacement, concurrency) propose at once, one move each on their own knobs on one level. Every
  proposal is sketched at the probe rung (`PROBE_WARMUP`/`PROBE_SIM`, else the run's fidelity):
  incumbent, each alone, every pair, all together, one wave; the analyst composes the round's
  design from the sketches, altering no value. Every design evaluated counts against the budget;
  the incumbent is the best **feasible** composed design. A move refused against the incumbent is
  a hold; a knob that lost twice is `blocked`. One round that does not move the incumbent makes
  the next a **jump round**: the wave sketches every unmeasured one-knob move at one level (the
  level swept longest ago first, a never-swept level before any, `SWEEP_CAP` = 8), beside the
  analyst's **restructure** written from its counterfactual bottleneck. `SWEEP=0` is the older
  guess-based jump. The stall counter is TuRBO's.
- `random` - uniform draws from the feasible set, `BO_BATCH` a round, no model. The baseline on
  the generation-step cell.
- `bo` - random forest + expected improvement, `BO_BATCH` a round; the tuned baseline of the
  older cells, not run on the generation step.

**A round costs** one simulation step plus ~3 serial model calls (~40 s): 1/3/7/11 new designs for
1/2/3/4 concerns proposing; a jump round up to 9. `ROUNDS` caps the council, `BO_ROUNDS` the
random and forest arms, `BUDGET` caps designs; all three in the report's `flags`. Size in rounds
when comparing arms.

**What a specialist sees**: its principles (textbook, never a value; one power principle each
for prefetch and geometry on a capped chip); the chip card and objective; its knobs and the values
this chip can build (for geometry the ladder in cycles, mm², nJ and leakage W); the sheet; the
incumbent's per-level report and cost line; the constraint view; what each workload contributes;
its concern's moves with deltas and sketches; what each value of its knobs has done (mean W on a
capped chip); the recent designs; the task. Nothing names a workload, a trace or a chip.

**The model** (`loop/analyst.py`): Gemini through `google-genai`, `ANALYST_MODEL` (default 3.1
Pro), JSON mode, thinking budget, 429 backoff, cost logged to `results/llm_usage.json`.
`GCP_PROJECT` picks the project billed (default the first account's).

**Search space** (`loop/space.py`, 13 knobs; 73.7 M raw designs on the `next` chip, and per chip
since the shapes are derived): L1D sets/ways/prefetcher; L2 and LLC
sets/ways/prefetcher/replacement/MSHR. Feasibility is `within_budget`; cost is `silicon_mm2`,
`leakage_w`, `power_split`/`watts`; after-measurement rules are `violations` - which on a
multi-core chip put the floor under **each core**, not under the mix.

**Scoring** (`search score`): per arm the share of the stock-to-best-known gap after N rounds and
N designs, mean over seeds with min..max, rounds-to-level. Best known = the best **feasible** design
measured on every workload in the tables at the run's fidelity. **The ceiling must stay
independent**: `search ceiling` is the forest run long. Scoring against a design an arm found
bounds the metric at that arm - the standing caveat on every number below.

## The generation step (the cell the paper is built on)

Design in `CROSS_SOC_LOOP.md`. A design team's hierarchy shipped on `E_prev`; the node shrinks,
the core widens, the memory changes, a workload class arrives and the cache silicon is capped
because the freed area went to more cores. Last generation's design (9.46 mm² at 22 nm) cannot
be built. The loop answers what replaces it, against **the obvious engineering answer**: the same
design with the last level shrunk until it fits, 3.07 mm², 0.98 W, IPC 0.7539 - the `next` chip's
stock, and the reference for every number.

| `SOC` | chip | clock / node | core | memory | caps |
|---|---|---|---|---|---|
| `prev` | `E_prev` | 3.2 GHz, 32 nm | 4-wide, ROB 256 | 2 ch x 3200 | 8704 KB (last generation's own) |
| `next` | `F_next-af8a` | 3.8 GHz, 22 nm | 6-wide, ROB 384 | 2 ch x 4800 | **4.0 mm², 1.0 W, no workload slower than the stock** |

Cell `gen` = llama2 (the class that arrived) + whiskey_0000 (the datacenter workload the old chip
was built for). Objective unchanged: geometric-mean IPC.

**Three caps, three keys in the SoC entry** (`area_budget_mm2`, `power_budget_w`,
`regression_tolerance`; a chip without them keeps the KB budget and no power rule):

- silicon over the three levels from CACTI, and the leakage the shape alone decides, are refused
  **before** simulation (`space.within_budget`), so they cost no design and never appear in a knob
  list or a random draw;
- power (CACTI read/write energy x counters + leakage x time + 15 pJ/bit off chip, an assumption
  named in `space.power_split`) and the per-workload floor are checked **after** measurement
  (`space.violations`). A design that breaks one is measured, recorded with the reason, shown to
  the council in every ledger, and never stood on (`incumbent_of`, `best_so_far`, the best known).

**What the council reads about cost**, all derived: a budget line in the chip card; a cost line
`3.07 of 4.0 mm2 | 0.98 of 1.0 W: L1D 0.08, L2C 0.15, LLC 0.58, off-chip 0.17`; the ladder with
nJ per access and leakage per shape; a `power L1D 10% L2C 15% LLC 53% off-chip 22%` line beside
memory time; mean watts per knob value in the value ledger; one power principle for prefetch and
one for geometry; the sheet's `over_budget` key. About 9250 tokens a round, from 8935. Smoke
finding: the analyst reads the power and names the last level's leakage, but still claimed halving
it would close a 0.43 W gap the ladder prices at 0.29 W - it reads, it does not yet compute.

**The gate** (`results/gate/`, 79 designs, no model): best feasible design +27.2 % over the
baseline on 15 % less silicon and 13 % less power, 12 of 13 knobs moved, the best single move worth
25 % of the gain, 6 knobs that lose alone, both workloads gaining. A random draw, so a floor.

**Arms on this cell**: `council` and `random` (uniform feasible draws, `BO_BATCH` a round, no
model). Default `ARMS` is `random, council`. All runs: 2 seeds, 1M/2M, on the second account's VM,
pulled to `results/vm2/` (tables, reports, transcripts) every 10 min by `results/vm2/pull_g1.sh`.

**Three council mechanisms were added for this cell**, each an env flag recorded in the report:

- `OPENINGS=N` - round 1 also measures N random feasible designs beside the analyst's opening; the
  council climbs from the best of them.
- `SOFT_CAPS=1` - a capped-out design may be the incumbent when it gains speed; dropped.
- `HOLD_JUMP=1` - a climb round in which every specialist holds becomes the jump round at once
  instead of spending the round and jumping the next. Trades designs for rounds: a held round costs
  0 to 3 designs, a jump round up to 9.

and one prompt change: each workload's IPC against its own floor, in the workload view
(`workload_view`), so the per-workload rule is visible before a proposal rather than after a
refusal.

**What was run, and what came out.** Stock (the shrunk baseline) 0.7539. Best feasible design
measured by any arm **0.9718 (+28.9 %)**, found by the council in `g5a` seed 1 at round 12; `g2a`
and `g4a` found 0.9714 independently. The standing caveat applies: no independent ceiling exists,
so every share below is measured against a design the council itself found.

| run | arm and flags | rounds | designs/seed | final, and share of the gap |
|---|---|---|---|---|
| `g1` | random, 5 a round | 20 | 100 | 0.9268 / 0.9547 (79 % / 92 %) |
| `g1` | council, hard caps, no openings | 20 | 126, 119 | 0.9392 / 0.9538 (85 % / 92 %) |
| `g2a` | council, `OPENINGS=5` | 19, 14 (cut) | 115, 72 | 0.9631 / 0.9714 (96 % / 100 %) |
| `g2b` | council, `SOFT_CAPS=1` | 20 (cut) | 84, 95 | 0.9627 / 0.9590 (96 % / 94 %); artifacts deleted |
| `g3r` | random, 5 a round | 75 | 375 | 0.9450 / 0.9547 (88 % / 92 %) |
| `g4a` | council, `OPENINGS=5 HOLD_JUMP=1` + the floor line | 20 | 135, 119 | 0.9578 / 0.9714 (94 % / 100 %) |
| `g5a` | council, `OPENINGS=10 HOLD_JUMP=1` + the floor line | 20, 18 | 139, 93 | 0.9547 / **0.9718** (92 % / 100 %) |

Share of the gap after N rounds, and rounds and designs to level, per seed:

| run, seed | R1 | R3 | R5 | R8 | R10 | R15 | R20 | 90 % | 95 % | 99 % |
|---|---|---|---|---|---|---|---|---|---|---|
| `g1` council s0 | 0 % | 24 % | 53 % | 84 % | 84 % | 85 % | 85 % | - | - | - |
| `g1` council s1 | 0 % | 24 % | 68 % | 78 % | 84 % | 90 % | 92 % | R14, 81 d | - | - |
| `g2a` s0 | 0 % | 24 % | 52 % | 83 % | 90 % | 95 % | 96 % | R12, 64 d | R12, 71 d | - |
| `g2a` s1 | 42 % | 80 % | 95 % | 97 % | 99 % | 100 % | - | R5, 25 d | R5, 25 d | R11, 54 d |
| `g4a` s0 | 26 % | 29 % | 53 % | 89 % | 94 % | 94 % | 94 % | R10, 78 d | - | - |
| `g4a` s1 | 42 % | 96 % | 97 % | 98 % | 98 % | 100 % | 100 % | R3, 16 d | R3, 16 d | R12, 67 d |
| `g5a` s0 | 54 % | 87 % | 88 % | 90 % | 91 % | 91 % | 92 % | R6, 47 d | - | - |
| `g5a` s1 | 42 % | 88 % | 95 % | 96 % | 97 % | 100 % | - | R4, 24 d | R4, 24 d | R12, 58 d |
| `g3r` random s0 | 0 % | 77 % | 77 % | 77 % | 77 % | 79 % | 79 % | never in 75 | never | never |
| `g3r` random s1 | 42 % | 42 % | 61 % | 87 % | 88 % | 92 % | 92 % | R13, 63 d | never | never |

**The findings.**

1. **Hard caps with a one-step climb are slow, not stuck** (`g1` council). Both seeds committed a
   feasible improvement from round 3 and 4 and ended at 85 % and 92 %, below every configuration
   with openings.
2. **The opening draw buys rounds, not the endpoint.** Rounds to 90 % across the four
   configurations, weak seed first: never/R14 with no openings, R12/R5 with five, R10/R3 with five
   plus the levers, R6/R4 with ten. The weak seed's final went 85 %, 96 %, 94 %, 92 % over the same
   four, which is the opposite ordering; with one seed per configuration that is noise, not a trend.
   Of 10 feasible draws per seed only 1 and 2 survived the caps, matching random's 22 % feasible
   rate over 750 draws.
3. **`SOFT_CAPS` works and is slower** (`g2b`): 96 % and 94 %, neither reaching 100 % in 20 rounds.
4. **`HOLD_JUMP` and the floor line buy rounds with designs.** Both `g4a` seeds reached 90 % sooner
   than `g2a` (R3 and R10 against R5 and R12); designs per round rose from 5.8 to 6.8. The endpoint
   did not improve, which 2 seeds cannot separate from noise.
5. **The gains come from the sweeps, not the climbs.** On `g2a` s0 every gain came from a jump
   round (12 ways, spp_dev at the LLC, va_ampm_lite at the L1D, ip_stride at the L2); 8 of 18 rounds
   committed nothing. The analyst's cross-concern compositions were refused on the datacenter
   workload four times while a single part alone gained; the rule that takes the best sketch instead
   carried the round each time.
6. **Random plateaus below the council.** Its 8 best feasible draws of 750 sit between 85 and 92 %
   of the gap; nothing reached 95 %. Matched at random's own best (0.9547, 92 %): random needs 13
   rounds and 63 designs on its lucky seed and never gets there on the other in 375 designs; the
   council reaches it in 3 to 5 rounds and 16 to 25 designs on its good seeds and 10 to 12 rounds on
   its weak ones, a **2.6 to 4.3x speedup in rounds**. Extrapolated from 0 hits in 750 draws, a
   design at 95 % of the gap is about 1 in 1500 feasible draws, so ~210 rounds of 5 for a coin-flip
   chance. Three of eight council seeds reach 100 %; no random seed passes 92 %.
7. **What the caps refuse.** 34 to 50 % of the council's designs and 77 to 78 % of random's draws
   break a cap. Of the council's designs that are faster than the baseline, 24 to 37 % are refused,
   the per-workload floor about twice as often as the power cap (the datacenter workload is what
   binds). Refused designs are measured, shown and never stood on.

**The best design measured** (`g5a` s1, round 12; 0.9718, 2.93 of 4.0 mm², 0.97 of 1.0 W; llama2
1.537, whiskey 0.615, both above the baseline):

```
L1D 128x12 va_ampm_lite | L2 256x16 va_ampm_lite drrip m64 | LLC 2048x16 spp_dev srrip m32
```

A prefetcher at every level, which is the coupled move across concerns the mobile cell could never
reach. `g4a` s1 reached 0.9714 with a different L2 shape (1024x4 va_ampm_lite ship), so the L2
capacity is nearly free at this point and the prefetcher placement is what pays.

**Reports on disk**: `results/vm2/runs/next_gen_{g1,g3r,g4a,g5a}.json` with per-design `mm2`,
`watts` and `violations`; transcripts for every council seed in `results/vm2/transcripts/`. `g2a`
and `g2b` were cut before a report was written, so their numbers come from transcripts and this file.

**Parked from the plan**: the one-page report (`loop.report`), the YAML brief,
the L1D latency ceiling as a fourth cap, a predicted-power line before proposing, the greedy-climb
baseline (~3 waves, no model), and random search run long enough to bound 95 % of the gap
(~210 rounds a seed, ~4 h, no model; it doubles as the independent ceiling).

## The memory, and the brief that moves (built; first cell cut short)

The paper is called *few-shot* and the loop was zero-shot, starting from nothing every run.
`loop/memory.py` gives it what past searches measured. **The scenario is the brief changing, not the
chip**: a cap moves after a chip is under design, every measurement stays true, and the design that
was legal may not be. Same chip, same workloads, only the budget line moves - so the answer memory
holds dies while the knowledge in it lives.

**A record** is one experiment a past council round actually ran: the **situation** before it (wait
share per level, channel fill - fractions only, so it reads on any chip and names none), the
**experiment** (which knobs moved from what to what), the **result** (speed, mm2, W, and any cap it
broke) and the **reasoning** the specialist wrote for that move. One line, ~50 tokens.

- **Cost is stored absolute, never as a share.** A share is wrong the moment the cap moves: the same
  design is 77 % of 4.0 mm2 and 123 % of 2.5. The share is computed when a record is read, against
  whatever cap is in force then (`memory.spent`). That is what makes one corpus serve a cut and a
  rise alike.
- **Corpus** `results/memory/<chip>.json`, built offline by `memory.build` from the run JSONs
  already on disk - no simulation. **539 records** from the four `next` reports: 110 carry a
  rationale, 478 moved exactly one knob, **162 were refused for breaking a cap** - the cliff data no
  fresh search has, and what a loosened cap makes valuable. Reasoning is dropped, never reassigned,
  where a round recorded one text for two concerns or a sweep's placeholder label.
- **Matching**: bucket, then rank inside. Bucket = how much of the wait is off chip (3 bands) x room
  under the caps in force (2). Measured: filing by **channel fill tracks which chip a record came
  from**, so it ranks rather than files. Ranking by closeness alone returns only moves that did
  nothing, so the five split 3 that gained / 2 that lost. An empty bucket is silence, and counted.
- **Where it speaks**: `MEMORY=1`, the analyst's round-1 prompt only - the one round with nothing
  measured to stand on. `MEMORY_JUMP=1` adds the stalled rounds; built, unrun. ~188 tokens, one
  round in twenty.

**The cells**: same chip, same `gen` cell, four cap settings (`AREA_CAP` / `POWER_CAP`, recorded in
every report). The baseline is re-shrunk per cell by `suite.shrunk_to_fit` - silicon and leakage
decide it before simulation, and a **measured** power reading decides it where the tables hold one,
because shrinking on leakage alone under-shrinks.

| cell | caps | baseline | speed |
|---|---|---|---|
| silicon cut | 2.5 / 1.0 | LLC 1024x16 | 0.7337 |
| power cut | 4.0 / 0.7 | LLC 1024x8 | 0.7197 |
| power raised | 4.0 / 1.4 | LLC 2048x16 | 0.7539 |
| swap which binds | 2.5 / 1.4 | LLC 1024x16 | 0.7337 |

Loosening **silicon is a no-op**: the stock sits at 98 % of the power cap and 77 % of the silicon
one, so 4.0 -> 6.0 mm2 leaves the legal set identical. Power binds, not area. A cap never changes
the chip's revision (`F_next-af8a` under all four), so every cached design and corpus key still
reads.

**`c1` (silicon cut), cut at round 5-6 of 20.** Arms `random`, council `MEMORY=0`, council
`MEMORY=1`, 2 seeds. Mechanism verified live: 1 memory block per memory-on transcript, 0 per
memory-off, offering `l1d_ways 8->12 + llc_sets 2048->1024` at 2.09 mm2 under the new cap. Random
reached **0.9287 over 100 designs in zero wall clock** - every draw was already in the tables, so
this cell is read in rounds and designs, never in minutes. Round 1: memory-on better on seed 1
(0.7750 vs 0.7633), worse on seed 0 (0.7175 vs 0.7479). Artifacts in `results/vm2/` as `c1off` /
`c1on`.

**Open: one dose may be too little to measure.** Seed 0's round 2 came out **byte-identical** with
memory on and off - memory speaks once, both arms then read near-identical prompts, and that seed is
a null by construction rather than by evidence. Before the four cells run for real: either
`MEMORY_JUMP=1` becomes a pre-registered fourth arm, or the null is reported as a null about *this
dose* and said so. Or trying to fix whats wrong, and find improvements to make memory valuable. 

**Open: cross-SoC memory, an ablation of its own.** A situation is shares only and names no chip, so
it travels; the cost does not. On four cores a private level is built once per core, so a move
measured on one core is mispriced there. Reading another chip's records needs the move **re-priced
on the chip being tuned** (from the knobs, which `space` already does) and its budget share read
against **its own** chip's caps - which is why each corpus file records `measured_under`. ~10 lines,
plus one run on the target chip. Also open: the "room under the cap" threshold.

## The shared machine (the cell the loop is on now)

A many-core cloud CPU, four of its cores modelled (`SOC=cloud`, `G_cloud-accd`): private L1 and L2
each, one shared last level, so a private byte costs four times a shared one. 3.44 GHz, 22 nm
standing in for a finer node, **2.48 B per core-cycle** against the single-core chips' 20.2 - a
tenth of the bandwidth per core, which is why a prefetcher here is a decision and not a free win.

**Why this chip.** Five shipping cloud CPUs of one generation disagree by **24x** on how to split
cache between the private level and the shared one - 1:4, 1:2, 1:1.6, 5.3:1, 6:1 across Genoa,
Bergamo, Grace, Graviton4 and AmpereOne - and none published which is right for a workload. Ruled
out: BOOM (Chipyard has no prefetcher generator, and every winning design spends its gain on
prefetchers); big+small cores; Lunar Lake (four levels, our space is three).

**Caps are a floorplan allowance, not a margin round the stock**: 2 mm2 of cache per core tile
(8 mm2) and 1 W per core (4 W, from the part's 500 W over 72 cores). The shipped hierarchy costs
11.78 mm2 and leaks 3.00 W - 1.5x over, so it cannot be ported.

**Cell `tenants`**: llama2 beside three datacenter services, one per core, one mix. Baseline (the
shrunk stock) **0.3832**, per core 0.79 / 0.38 / 0.34 / 0.27; 78 % of memory time in DRAM, 51 % of
the channels already used, the shared level hitting 0.18.

**Two rules the first runs forced.** Both generic, both in `space.py`, neither with a number in it.

- **The floor is per core, tolerance 5 %.** Four tenants share the last level, so any reallocation
  takes from someone. At zero tolerance a design 2.9 % faster over the machine was refused for
  costing one core 1.7 %, and after 13 designs nothing had ever moved.
- **The hierarchy must be ordered** (`space.hierarchy_is_ordered`): each array holds at least as
  much as the one in front of it and is no quicker to answer. All six SoCs' own hierarchies pass
  it. Without it the search bought a **1 MB first level at 6 cycles in front of a 512 KB level at
  4**, which is not a hierarchy. Arrays are compared as built, not per core, so a chip with more
  private capacity than shared is still ordered (Graviton4). Equal sizes are allowed - these caches
  are non-inclusive, so a last level the size of the one in front of it holds different lines
  (Skylake-SP: 1 MB L2 behind a 1.375 MB L3 slice). Strict growth instead costs 2.5 points.

**`t2`, measured under the 5 % floor and before the ordering rule** (council 20 rounds, random 50,
2 seeds, plus an independent forest of 300 designs at 5 a round):

| arm | designs | best | vs stock |
|---|---|---|---|
| random | 250 a seed | 0.4270 / 0.4201 | +11.4 % / +9.6 % |
| forest, independent | 300 | 0.4411 | +15.1 % |
| council | 121 / 143 | 0.4580 / 0.4199 | +19.5 % / +9.6 % |

Council at 73 % of the gap by round 8, random at 54 % after 50 and never past 58 %. **The +19.5 %
design is the one the ordering rule now forbids, so that number is retired.** Re-scored under the
rule (`results/vm2/t2_ordered.md`): forest 0.4411, council 0.4248, random 0.4069, and the council
leads random at every round, 72 % against 41 %. Not a fair race - no arm was searching under the
rule - which is what `t4` is for.

**`t4`, running**: council 35 rounds, random 60, 2 seeds, under both rules. A council round costs
~13 min (one four-core simulation is ~10 min and a wave runs in parallel), so ~7.5 h. The bar:
random already sits at 0.4249 (+10.9 %) and 0.4411 is known reachable.

    screen t4 on champsim-2 | results/cloud_t4.log | results/runs/cloud_tenants_t4.json
    sudo -u mateobuscarons bash /home/mateobuscarons/hackathon/status.sh   # one line, every run

**Open: no independent ceiling exists for the ordered space.** The forest that measured 0.4411
searched the unordered one and its log was deleted with `t1`/`t3`. Re-run it before quoting any
share; at 5 a round it costs 5 cores and runs beside the arms.

## How to run

```bash
SOC=next LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m loop.search preview gen   # every prompt,
                                                           # no model call; set the fidelity or it
                                                           # simulates the stock into another table
SOC=next SEEDS=1 python -m loop.search smoke <tag> council  # the gate before any launch
SOC=next SEEDS=2 ROUNDS=20 BO_ROUNDS=20 BUDGET=300 BO_BATCH=5 \
  LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=7 PARALLEL_RUNS=4 \
  python -m loop.search gen <tag> random,council            # both arms, sized in rounds
SOC=next LOOP_WARMUP=1000000 LOOP_SIM=2000000 python -m loop.search score results/runs/<file>
python -m loop.search ceiling <designs> <batch> <trace>...  # the independent ceiling

SOC=cloud SEEDS=2 ROUNDS=35 BO_ROUNDS=60 BUDGET=400 BO_BATCH=5 \
  LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=14 PARALLEL_RUNS=4 \
  OPENINGS=10 HOLD_JUMP=1 MEMORY=0 CHAMPSIM_BUILD_SHARE=8 \
  python -m loop.search tenants <tag> random,council        # the shared machine
```

On a VM this goes in a script launched detached - `screen -dmS <tag> bash -c '<script> > <log> 2>&1'`
so nothing on the laptop can take it down. Killing a process group from an ssh session kills that
session too, and on this VM it took sshd with it.

Env: `SOC`, `SEEDS`, `FIRST_SEED`, `BUDGET`, `ROUNDS` / `BO_ROUNDS`, `BO_BATCH`, `PARALLEL_RUNS`,
`LOOP_WARMUP` / `LOOP_SIM` (the run's fidelity, own tables), `PROBE_WARMUP` / `PROBE_SIM`, `PAIRS`,
`GEOM_LADDER`, `SWEEP`, `CHIP_VIEW`, `SIM_THREADS` (per workload per run; total = that x workloads
x `PARALLEL_RUNS`, keep near the core count), `CHAMPSIM_BUILD_SHARE` (8 on a VM), `TABLE_DIR`,
`ANALYST_MODEL`, `ANALYST_LOCATION` (`global`), `GCP_PROJECT`, `CACTI` / `CACTI_TEMPLATE`.

**What a design costs.** A build is ~6 s; simulation time follows prefetcher traffic. A `gen`
design at 1M/2M is two simulations; the Mac does 4.8 designs a minute at 10-way parallelism, a
32-core VM about twice that. A `tenants` design is ONE simulation of four cores and takes ~10 min,
so a round of either arm is 10-13 min on the 56-core VM whatever the wave size. Gemini 3.1 Pro is ~10 s a call, ~0.017 USD, ~7 calls a round.

## Machines and accounts

**Two GCP accounts, two config dirs, never mixed** (`CLOUDSDK_CONFIG`; ADC is one file per dir,
a login under the wrong one destroys the other's credentials):

| | first | second (short-term, hackathon grant) |
|---|---|---|
| config | `$HOME/.config/gcloud-chia` | `$HOME/.config/gcloud-chia2` |
| project | `project-c23a6080-f5d0-4871-9cb` | `a3-chia-hack26ath-7716` |
| VM | `champsim-1`, c2d-standard-32, europe-west4-a | `champsim-2`, **c2d-standard-56**, europe-west4-a, made from machine image `champsim-image` of champsim-1 |
| Vertex | yes; org policy forbids service-account keys | yes: `aiplatform` enabled, the VM's compute SA is `aiplatform.user`; run with `GCP_PROJECT=a3-chia-hack26ath-7716` |
| cost | ~1.5 USD/h + Gemini, on a granted credit (id in the session memory, not here) | **none, no quota**; expires without notice at the end of the hackathon window, deleted accounts are unrecoverable |

On both VMs ssh lands as another user; the tree is `/home/mateobuscarons/hackathon`, use
`sudo -u mateobuscarons`, code goes by `gcloud compute scp` (no git checkout). Launch as
`sudo -u mateobuscarons setsid nohup bash <chain>.sh` so the run is its own process group. The
second VM is disposable: anything on it is copied off continuously or already lost; the tables it
measures live in `results/vm2/tables/`, apart from `tables/` because a table belongs to the machine
that measured it. **Check `gcloud compute instances list` under both configs before anything
else; stop both when idle.** In zsh, gcloud flags must be written literally, not in an unquoted
variable (zsh does not word-split `$VAR`), and `timeout` does not exist on the Mac. On the Mac,
`caffeinate -i` stops idle sleep only.

**Talking to a VM: do not use gcloud for it.** `gcloud compute ssh/scp` re-authenticates, re-queries
the API and re-checks SSH keys on EVERY invocation - measured ~40-60 s of setup per command whatever
it carries, which is why a handful of small calls costs more than the work. Direct ssh over a held
connection is **0.5 s a command**, and the login user is the **Mac's local username**
(`mateo.buscarons`, dots and all), not the Google account: gcloud names the Linux user after the
local OS user, which is why `devstar7716*` is refused.

    ssh -i ~/.ssh/google_compute_engine -o ControlMaster=auto -o ControlPath=~/.ssh/cm2 \
        -o ControlPersist=8h mateo.buscarons@<external ip> 'sudo -u mateobuscarons ...'

Use gcloud only to list, describe, start and stop. Send a script by base64 rather than a heredoc
through the ssh command string, which does not survive the quoting.

Setup of a fresh machine (ChampSim with the spp_dev patch, CACTI 7 with the clang fixes, the
venv, the traces) is in git history under this heading; a machine image makes it unnecessary.

## Next, in this order

1. **Read `t4`** - council against random on the shared machine, under both rules, 2 seeds each.
2. **The independent ceiling on the ordered space** - without it no share on this cell is quotable.
3. **The CHIA blocks** (`CHIA_BLOCKS.md`, `CROSS_SOC_LOOP.md` §7): CACTI in cache mode upstream,
   since CHIA's own runner is memory-mode only (`cache type "ram"`, associativity 1, no write
   energy), then Chipyard's `.top.mems.conf` through its `characterize_top_mems_conf_with_cacti` as
   a check that the cost model ranks designs the way a real elaboration does - Chipyard per design
   is out, no prefetcher generator. Then `chia/dse/` with the brief, `cost` and `constraints` as
   pure functions, and the configuration-space ChampSim node.
4. **Fix memory, or discard.**
5. **Apply JEV AI to this (idea for profiling and evaluating: open item from the CHIA paper)**
6. **The paper and the README**.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only.** Every launch needs the
user's explicit OK with cost and time. Kill by process group or exact pid, never a broad `pkill`.
Code stays lean, one mechanism, no version archaeology; domain decisions are surfaced with options
and left to the user. Deleted material is in git history, never in the tree.
