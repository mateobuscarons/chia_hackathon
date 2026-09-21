# Few-shot cache tuning by a council of LLM specialists

Cache-hierarchy design-space exploration on ChampSim. The question: **given a stock chip and a
workload suite the loop has never seen, how many rounds does it take to get within x % of the
best design known?** Rounds-to-level, not level-at-N: a round is one wave of parallel
simulations, so it is the unit of time; a design is the unit of cost.

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
committee on "author-identified highlights, paper submission, artifact, and their judgment";
AI reviewers with personas comment but do not rank. Tracks this project answers, quoted:

- "Cross-SoC optimization of cache hierarchies" (the named track)
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
chip, against a tuned optimizer at the same budget — and **the chip enters the prompts only as
derived numbers, never as a name or a branch** (below).

## The rule the cross-SoC work rests on

> A prompt may name a quantity. It may never name a chip, a chip class, or branch on one.

Three sources of chip fact, each derived once, nothing hardcoded per SoC:

| source | kind | what it yields |
|---|---|---|
| the SoC profile (`loop/socs.py`) | declarative | the chip card (`loop/chip.py`) |
| the simulator's counters | measured | the constraint view in the prompts |
| the design space ∩ the area budget | derived | the shapes this chip can actually build |

`loop/socs.py` is the only file in `loop/` that names a chip. Adding an SoC is one entry there.
Verify with: `grep -rn "server\|mobile\|quad" loop/*.py | grep -v socs.py` — it should return
nothing but the word "server" in an HTTP comment.

**What the council reads about the chip**, all of it computed (`loop/chip.py`, `loop/council.py`):

```
Core: 4-wide issue (6-wide fetch), ROB 512, LQ 192 / SQ 114, 4.0 GHz, 1 core, 32 nm.
Memory: 2 channels x 3200 MT/s x 8 B = 12.8 B per core-cycle; a DRAM service costs ~180
        cycles on a row miss, ~60 on a row hit.
Levels: L1D, L2C, LLC, one of each.          (4 cores: "L1D and L2C private, one per core;
Area:   4608 KB of L2 + LLC data, a private   LLC shared by all 4 cores")
        level counted once per core.
```

and per workload, from the counters plus the card:

```
memory time L1D hits 11% L2C hits 8% LLC hits 0% DRAM 82%
off-chip 1.10 of the 12.8 B/cycle the channels deliver (9%)
a miss past the last level costs 187 cy, against the 180 a DRAM access takes with the
  channels idle (60 if its row is already open)
DRAM row-buffer hit 0.00        per-core ipc 0.10 / 0.43 / 0.10 / 0.43   (cores > 1 only)
```

**The bottleneck is ranked by share of memory time, not by misses per kilo-instruction.** The
mpki rule pinned the blame on the L1D in ~60 % of all sheets on every chip, because the L1D
always misses most; memory time gives DRAM 82 % on the server, 90 % on mobile, 91 % on the quad
at stock, and `bandwidth-bound`/`DRAM` — never once chosen under the old rule — is now the round-1
verdict on all three. The decomposition is AMAT-style (each level's demand hits times its hit
latency; the last level's demand misses times the measured DRAM service), so it ranks the tiers;
it does not account for cycles, and the hit tiers are time a wide core partly overlaps. It reads
each level's own hits, not the difference between two levels' misses: a miss that merges into a
miss already outstanding is counted at its level and never reaches the next, and on llama2 three
L1D misses in four merge (quad stock: 1.13 M L1D misses, 0.29 M requests reaching the L2C), so
the difference credited the L2C with a phantom 7 % tier at stock on quad and mobile where it
served about a thousand hits. The "of the level above's misses X miss again" line is net of
merges for the same reason; it used to say 0.25 where the truth was 1.00.

**Cache latency and area come from CACTI**, in cache mode, at the SoC's node and clock
(`loop/chip.py`): `cycles = ceil(access_time_ns x GHz)`, L1D in parallel ("normal") mode, L2 and
LLC sequential, 64-byte block, tag array, a non-power-of-two associativity interpolated on log
ways between the two powers of two that bracket it. Cached to `results/cacti_<node>nm.json`, so a
machine without a CACTI build only needs that file. This is not a refinement of ChampSim's size
formula — it inverts it:

| | 32 nm @ 4 GHz | 45 nm @ 2 GHz | ChampSim's formula (both) |
|---|---|---|---|
| L1D 16 KB×8 / 96 KB×12 | **4 / 6 cy** | **3 / 4 cy** | 3 / 5 cy |
| L2 64 KB / 1024 KB | 3 / 8 cy | **2 / 5 cy** | 4 / 12 cy |
| LLC 512 KB / 4096 KB | 6 / 14 cy | **4 / 9 cy** | 9 / 19 cy |
| LLC 4096 KB area | 9.5 mm² | 21.5 mm² | not modelled |

Associativity costs cycles and capacity barely does; on a 2 GHz part capacity is nearly free in
cycles and twice the silicon. The geometry principle that used to say "hit latency is a function
of capacity alone" was false against the ladder printed beneath it and has been corrected.

**Other things the chip now decides, with no branch anywhere:** the knob lists and the capacity
ladder show only shapes this chip can build within its budget (on mobile that removes two thirds
of the geometry); a proposal that does not fit is refused with the overshoot named and one retry,
never silently resized; a private level costs its capacity once per core in the area budget
(which is what finally makes the quad's 8192 KB bind at all); a level's report line says whether
it is private or shared and gives its per-core capacity and its mm²; and on a multi-core SoC the
workloads are per-core mixes built from the cell's programs by core count, so one cell runs on any
SoC.

## The loop as it runs

Two arms (`loop/search.py`), both from the stock chip, same budget, seeds and fidelity:

- `bo` — random forest + expected improvement, `BO_BATCH` designs per round.
- `council` (`loop/council.py`) — the analyst reads every design measured and writes the sheet:
  a verdict per level with its number, the levels ranked by share of memory time so the tier that
  holds the most is never "fine", the bottleneck, each finding with its delta, design ids
  and the counter behind it, what several knobs moving together make unattributable, which knobs
  fail by their own counters, a prediction checked against the next measurement. All four
  specialists (prefetch, geometry, replacement, concurrency; own knobs, allowed values, hold on
  no evidence) read it and propose at once, one move each on their own knobs, on one level (a
  proposal across levels is asked once more for one level, then taken as proposed). Every proposal
  is sketched at the probe rung (`PROBE_WARMUP` / `PROBE_SIM`, else the run's fidelity): the
  incumbent, each proposal alone on it, every pair, all together, one wave of parallel
  simulations; the analyst composes the round's design from the proposals with the sketches in
  front of it, choosing among them and altering no value; a composed design the wave did not
  sketch is measured then; a design the sketches predict to lose is not measured; a part the
  simulator cannot measure is out of the round and on the refused list, never a zero in the
  ledger. Every design evaluated counts against the budget; the council reads and stands on the
  composed designs, and the incumbent is the best of them; when the sketches run at the run's
  fidelity they are measurements, and the best sketch of the wave is the round's design when it
  beats what the analyst composed. Round 1 is the same with the analyst's opening design from
  the stock report, split by concern. When every specialist holds, the analyst takes the turn
  with one design of its own, its parts held to the same refused list as everyone's. A move is a
  hold before it is sketched when the sketches refused it against this design, or when every
  knob it touches has already lost twice against this design (`blocked`). One round that does
  not move the incumbent makes the next round a jump round (below).
- **The jump round** (`STALL_ROUNDS`, `FLAT_SHARE`, `SWEEP`, `SWEEP_CAP` in `council.py`): the
  council only climbs, so it stalls in a local optimum with budget left. After one round that does
  not move the incumbent the round measures instead of guessing: the wave sketches every value
  of every knob at one level never measured as a one-knob change from the incumbent, each alone,
  the level swept longest ago first and a never-swept level before any (on mobile at stock the
  neighbourhoods are L1D 6, LLC 9, L2C 14 moves), at most `SWEEP_CAP` = 8 designs a wave. The
  rotation is the run's, not the incumbent's: ordered by neighbourhood size it restarted at the
  L1D after every hair of a gain and never reached the L2C (m3, rounds 8 to 15, both seeds); beside them one coupled design,
  the analyst's **restructure**, written from its counterfactual bottleneck: the knobs across
  concerns it would move together, which is the one cross-concern move the loop makes. Nobody
  proposes in a jump round; the analyst composes from the parts that gained, never carrying a
  part the sketches refused (`FLAT_SHARE` of the incumbent is the tolerance), and a composition
  nobody sketched is sketched before it is measured. A jump the sketches do not predict to gain
  is not measured; the next round sweeps the next level; once every one-knob move from the
  incumbent is measured only the restructure is sketched, and when the analyst offers none it
  takes the turn with a design of its own. A move refused against the incumbent, by the sweep or
  by a specialist, is a hold when re-proposed and costs no sketch. `SWEEP=0` runs the earlier
  jump round, in which each specialist proposes the one untried move it expects most and the
  next empty jump sweeps the level blamed and sketched least (`sweep_level`); it is the arm the
  sweep is measured against. The stall counter is TuRBO's; the trigger was two rounds, the
  criterion of AutoTuring's critic (arXiv 2609.19387), until the sweep made a jump round cheaper
  than a second round of guessing.

  Why the sweep exists: on the mobile cell the whole 8-way L1D family caps at 1.2785 in the
  table (237 designs) and 12 ways reaches 1.5103 (456 designs). Seed 0 of the guess-based jump
  finished at exactly 1.2785 after 20 rounds and 115 designs; `l1d_ways: 12` was on its untried
  list in every jump round from round 5 and the geometry specialist never proposed it, holding
  ten rounds on "the L1D is at the lowest latency for its associativity" - the CACTI ladder says
  3 to 4 cycles and the sheet named the L1D hit tier - while W1 sat at 49 mpki on that level.
  Seed 1 proposed it once, in round 17, for +0.185 in one design. On the server the same knob
  goes the other way (12 to 8 ways gained +0.007 and +0.02 in rounds 1 and 2; 8 to 12 lost
  0.0125 in a jump round). The principle is chip-dependent; the measured neighbourhood is not.

**What a round costs.** The wave goes out as one parallel batch, so a round is one simulation
step plus about three serial model calls (~40 s), as long as the wave fits in `SIM_THREADS`:

| concerns proposing | new designs in the round |
|---|---|
| 1 | 1 |
| 2 | 3 |
| 3 | **7** (the usual) |
| 4 | 11 |
| jump round (sweep) | up to 9: at most 8 one-knob moves and the restructure |

(the incumbent is already in the table; the composed design is usually one the wave measured)

**Sizing.** `ROUNDS` caps the council and `BO_ROUNDS` the forest; `BUDGET` caps designs. Whichever
binds first stops the run, and all three are recorded in the report's `flags`. Size in rounds when
comparing arms — they spend designs per round at different rates.

**What a specialist sees** (`council.specialist_prompt`): its principles (textbook cache
architecture, what a knob does, never which value); the chip card and the objective; its knobs and
the values this chip can build, for prefetch a placement table with one factual line per
mechanism, for geometry the chip's capacity ladder in cycles and mm²; the sheet; the incumbent's
per-level report for the levels it owns and the constraint view derived from the counters; what
each workload contributes and how far it is below the best measured for it alone; its concern's
moves with their deltas and sketches; what each value of its knobs has done; the recent designs;
the task. In a jump round it also reads the caveat that knobs moved together are not separated,
and the values of its knobs never measured as a one-knob change from the current design. Nothing
names a workload, a trace or a chip.

**The model** (`loop/analyst.py`): Gemini through `google-genai`, one model per process
(`ANALYST_MODEL`, default 3.1 Pro): JSON mode, a thinking budget, 429 backoff. Cost per call is
logged to `results/llm_usage.json` from the price table in the module. gpt-oss-120b, DeepSeek V3.2
and Kimi K2 Thinking were run through Vertex's OpenAI-compatible endpoint on the same prompts:
gpt-oss opened poorly and committed little, Kimi was throttled and slow, DeepSeek climbed slower
than Gemini; that backend is in git history.

**Search space** (`loop/space.py`, 13 knobs, 6.6 M raw designs, area-coupled): L1D
sets/ways/prefetcher; L2 sets/ways/prefetcher/replacement/MSHR; LLC
sets/ways/prefetcher/replacement/MSHR. Area = L2 + LLC data capacity, a private level counted
once per core, against the SoC's budget; L1D and MSHRs cost nothing (known simplification).

**Scoring** (`search score`): per arm, the share of the stock-to-best-known gap after N rounds and
after N designs, mean over seeds with min..max below, and rounds-to-level. Best known = the best
design measured on every workload of the suite in the cached tables at the run's fidelity, so it
moves as searches land. **The ceiling must stay an independent mechanism**: `search ceiling` is
the same forest run long from the stock chip. Scoring against a design an arm found bounds the
metric at that arm — which is the standing caveat on every number below.

## The SoCs

| `SOC` | chip | clock / node | memory | area budget |
|---|---|---|---|---|
| `server` | `C_server-bf61` | 4 GHz, 32 nm | 2 ch x 3200 MT/s = 12.8 B/core-cycle | 4608 KB |
| `mobile` | `A_mobile-a868` | 2 GHz, 45 nm | 1 ch x 1600 MT/s = 6.4 B/core-cycle | 2048 KB |
| `quad` | `D_quad-063f` | 4 GHz, 32 nm, 4 cores | 2 ch x 3200 MT/s = 12.8 shared, 3.2 each | 8192 KB |

The revision token is a hash of everything about the SoC that is not a knob. Change a profile and
every design name changes, so rows and binaries from the old chip are never served for the new
one — and the old dataset stays intact instead of being silently mixed in.

## Cells and workloads

| cell | programs (heaviest first) | role |
|---|---|---|
| `aiml` | llama2_7b, stable-diffusion, clip (ML inference, DPC4 `ai-ml`, 200 MB prefixes) | the suite |
| `llama2` | llama2_7b alone | development, a third of the cost |
| `smoke` | mcf, lbm | the gate before any launch |

A cell names programs; the chip decides how they run. One core: one workload per program.
N cores: two mixes of N traces — the programs cycled onto the cores, and the first program on
every core (the worst case for a shared level). `mcf` on a 4-core SoC takes ~6 min per
simulation, so use `lbm` if a multi-core gate is ever needed.

## State

**The `llama2` cell on `C_server-bf61`** (`results/runs/server_llama2_x1.json`), both arms,
2 seeds, 1M/2M, council to 28 rounds and forest to 29, 0 failed, 98.6 min on the VM.
Stock 0.4819, best reached 1.4884 (+208.9 %).

| arm | 90 % | 95 % | 96 % | 99 % | 99.9 % |
|---|---|---|---|---|---|
| council | 4 (6,2) | 6 (7,5) | 10 (16,10) | 16 (16,10) | 19 (19,10) |
| tuned forest, `BO_BATCH=5` | 15 (15,13) | 25 (21,x) | 25 (23,x) | - (25,x) | - (26,x) |

Read mean-curve first, each seed in brackets, `x` for a seed that never reached it. The
denominator-free version, which is the one to quote: **both arms find the same best design;** the
council first measures it at round 10 (seed 1) and 19 (seed 0), the forest at round 26 (seed 0),
and the forest's seed 1 never does in 29 rounds.

```
L1D 128x8 va_ampm_lite | L2 256x4 va_ampm_lite drrip m64 | LLC 1024x8 spp_dev ship m64
```

The pre-CACTI chip's answer was `L1D 128x12`. **The council moved to 8-way** — 64 KB at 4 cycles
instead of 96 KB at 6 — which is the CACTI ladder being read and which the old latency model made
invisible. Against the pre-CACTI record, council to council: 90 % unchanged at round 4, 95 %
improved from 9 to 6, 96 % unchanged at 10, the tail 2 to 4 rounds slower. The tail is confounded:
that run had a target drawn from an ~800-design table, this one from its own 600.

**The `CHIP_VIEW` ablation** (`results/runs/server_llama2_cv1.json` and `cv0.json`), llama2 on
`C_server-bf61`, the Mac, 2 seeds x 8 rounds, council only. `CHIP_VIEW=0` reverts what the
council *reads* to the pre-derivation form and leaves the machine identical - same CACTI
latencies, same per-core area, same mixes.

| arm | R1 | R2 | R4 | R6 | R8 |
|---|---|---|---|---|---|
| `CHIP_VIEW=1` | **70 %** | **93 %** | **98 %** | **100 %** | **100 %** |
| `CHIP_VIEW=0` | 51 % | 90 % | 94 % | 97 % | 98 % |

Ahead at every round on both seeds, never behind, and the margin is largest at round 1. That is
the mechanism, not luck: the bottleneck the sheet names differs between the two readings only in
rounds 1-3 (below), and the opening is worth half the gap. By round 8 the advantage is ~2 points,
so on this cell **the chip view buys speed in the opening, not a better endpoint**. It costs
19 % more prompt (8309 against 6957 tokens a round). Kept.

**Where the reading actually changes the sheet.** Bottleneck named, by round, llama2:

| | rounds 1-3 | rounds 4-8 | rounds 9+ |
|---|---|---|---|
| pre-CACTI council | DRAM 0 %, L1D **100 %** | L1D 100 % | L1D 91 % |
| with the chip view | DRAM **33 %**, L1D 44 % | L1D 100 % | L1D 100 % |

Once the opening turns the prefetchers on, DRAM's share of memory time collapses (82 % -> 29 %)
and the L1D hit tier genuinely does hold the most time, so the rule is behaving correctly. It
also means the L1D tier - which is hit time a wide core partly overlaps - dominates the ranking
after round 3. Ranking by *removable* time (misses only) is an untested alternative, and
`preview` tests it for free.

**The `aiml` cell on `A_mobile-a868`** (`results/runs/mobile_aiml_m2.json`, reconstructed from
`results/transcripts/mobile-stdout-m2.log` because the run was stopped before `run_cell` wrote
its report; transcripts `mobile-council-m2-s{0,1}.log`, tables in `results/vm_tables/`).
Both arms, 2 seeds, council to 20 rounds and forest to 50, on the VM. Stock 1.0103, best reached
1.5103 (+49.5 %).

| arm | R1 | R5 | R10 | R20 | R50 | 90 % | best reached |
|---|---|---|---|---|---|---|---|
| tuned forest | 65 % | 87 % | 89 % | 91 % | 100 % | round 18 | 1.5103 / 1.5099 |
| council | 45 % | 53 % | 53 % | 72 % | - | **never** | 1.2785 / 1.4572 |

**The council loses this cell decisively, and the chip view did not fix it.** Seed 0 reached
1.2761 at round 2 and gained 0.2 % over the next eighteen rounds through five jump rounds. What
wins here is a structure neither seed reaches:

```
best (forest)  L1D  32x12 next_line    | L2 512x16 va_ampm_lite | LLC 1024x8 spp_dev
council s0     L1D 128x8  va_ampm_lite | L2 256x8  ip_stride    | LLC 2048x8 spp_dev
council s1     L1D 128x12 va_ampm_lite | L2 512x16 spp_dev      | LLC 1024x8 ip_stride
```

A 24 KB L1D with the simplest prefetcher, passing traffic down to a large 16-way L2 that covers
it. Reaching it takes four moves across three concerns at once - shrink the L1D, simplify its
prefetcher, grow the L2, change the L2's prefetcher. The council moves one knob on one level per
concern per round *by design*, so that every sketch is attributable, and can only land that
structure if three specialists propose the right thing in the same round. The opening committed
to the opposite structure and neither the climb nor the jump rounds could restructure out of it.

**The `aiml` cell on `A_mobile-a868` with the sweep, `m3`** (transcripts `mobile-council-m3-s{0,1}.log`
and `mobile-stdout-m3.log`; stopped at rounds 14 and 13 by hand, no report written). Council
only, 2 seeds, the sweep jump round plus the three climb rules (refused list for every proposer,
a knob that lost twice is a hold, one empty round triggers the jump), on the VM against the m2
tables. The first stall came at round 5 and 6, the L1D sweep found 8 to 12 ways at +0.1838 and
+0.1951 in one wave on both seeds, and the restructure lost on both. At round 8 and 9 both seeds
stood at 1.4567 / 1.4566 (89 % of the gap), where m2's better seed arrived at round 17 and its
other seed never did; the forest's seeds reached it at rounds 7 and 20. From there to round 14
neither seed moved past 1.4571: the sweep, ordered by neighbourhood size, restarted at the L1D
after each hair of a gain (L1D, LLC, L1D, LLC on both seeds) and never swept the L2C, whose
prefetcher is the missing move; and the climb round after each jump re-proposed the move that
had just lost 0.19 because the refused list was per incumbent. Both are fixed for `m4` (the
rotation by last swept; the refused list over every design within the flat tolerance).

**The `aiml` cell on `A_mobile-a868` with the sweep rotation, `m4`**
(`results/runs/mobile_aiml_m4.json`, transcripts `mobile-council-m4-*`): council only, 2 seeds,
20 rounds, the VM, 1.2 h. Everything m3 had plus the rotation by last-swept level and the
refused list over every design within the flat tolerance. Best known 1.5103 over 1218 designs.

| arm | R1 | R5 | R10 | R20 | 90 % | final |
|---|---|---|---|---|---|---|
| council m2 (guess-based jump) | 45 % | 53 % | 53 % | 72 % | never | 1.2785 / 1.4572 |
| council m4 (sweep) | 45 % | 88 % | 91 % | **95 %** | round 10 (10, 17) | 1.5061 / 1.4641 |
| tuned forest (m2) | 65 % | 87 % | 89 % | 91 % | round 18 | 1.5103 at R43 / 1.5099 at R47 |
| random search, 5 a round | 80 % | 88 % | 92 % | 93 % | round 8 | 1.4741 / 1.4802 at R30 |
| untuned forest, 1 a round | 69 % | 90 % | 90 % | 91 % | round 14 (5, x) | 1.5054 / 1.4576 at R30 |

The two easier baselines (a scratch script outside the tree, 30 rounds, 2 seeds, the VM, the same
tables): random search draws 5 feasible designs a round; the untuned forest is sklearn's
`RandomForestRegressor()` with index encoding, a fresh random pool of 2000 each round, 3 random
designs to warm up and one design a round. Random is at 92 % by round 8 on both seeds and never
passes 94 %; the untuned forest's seed 0 reached 99 % at round 26 on 29 designs and its seed 1
sat at 89 % for 28 rounds. **On this cell everything is at about 90 % by round 10, random
included; the last 10 % is a coupled pair (the L1D and L2 prefetchers together) that random
finds by luck and the council does not find by reasoning.**

Seed 0 reached 99 % at round 18 (1.5061): 12 ways at round 6, the L1D prefetcher to next_line
at round 10, 32 sets at round 15, each found by an L1D sweep. Seed 1 stopped at 1.4641: from
its incumbent (L2 on spp_dev) next_line at the L1D loses 0.02 to 0.03 alone in every L1D sweep,
and the L2 prefetcher to va_ampm_lite was never sketched (the L2C sweep's cap of 8 fell on the
first three prefetcher values twice). The missing move is a pair inside one concern, the L1D
and L2 prefetchers together, which no one-knob sweep can find and which the analyst's
restructure never wrote: its counterfactual named the LLC in every jump round of both seeds.
Climb rounds after a jump were mostly "every specialist held", and the analyst's turn parts
were dropped as refused, so they cost no design.
This is a search-structure problem, not a chip-reading one - the ablation above shows the reading
is worth ~19 points at round 1 and ~2 by round 8, and the mobile failure is a round-2 lock-in.
It is also where the forest's random warm-up earns its keep: it samples structures, the council
refines one. Three candidate fixes, none tried:

1. Let a jump round propose a coupled restructure **across** concerns once the single-knob
   neighbourhood is exhausted. Today only one concern may go coupled, and only within its levels.
2. Open from several structures - measure two or three structurally different openings in round 1
   and climb from the best, which is what the forest's warm-up does implicitly.
3. Rank by removable time rather than total memory time, so the sheet stops pointing at the L1D
   hit tier from round 4 onward.

**The caveat on every percentage here**: the best known is a design one of the arms found, nine
rounds in. `search ceiling` on this chip is what removes it, and it costs no model calls.

**The pre-CACTI record** — the jump round on llama2, the `PAIRS` / `GEOM_LADDER` ablation on aiml,
the mobile cell where the council's seeds diverged and the forest held the best known, the quad
cell where every good design sat at 87-92 % of peak bandwidth — is in
`results/archive-pre-cacti/`, with its own README. Nothing there is readable by the current loop.

**What the mobile failure taught, and why the chip view exists.** On the pre-CACTI mobile chip one
seed imported the server's answer (a big 12-way L1D with a prefetcher, a shrunken LLC) and lost
10 %: the geometry principle "a level with a low hit ratio is charging latency for nothing" fired
correctly and was wrong there, because the LLC hit nothing only while everything above it missed,
and a miss cost 180-318 cycles. The counters that would have said so — DRAM at 90 % of memory
time, traffic at half of what one channel can deliver — existed and were not shown. The one
bandwidth number that *was* shown was wrong by 16x on exactly the designs the council was choosing
between: it counted prefetch misses that merge into an outstanding MSHR and never leave the chip.
It now comes from the memory controller's own request counts.

## What is in `results/`

See `results/README.md`. In short: `runs/<soc>_<cell>_<tag>.json` is the report `search score`
reads, `transcripts/` holds every prompt and answer (gitignored), `tables/` is the simulation
cache and the dataset, `cacti_<node>nm.json` is the characterisation cache, and
`archive-pre-cacti/` is history.

## How to run

```bash
python -m loop.search preview <cell>                       # every prompt of an opening round,
                                                           # no model call, no search
SEEDS=1 python -m loop.search smoke <tag> council           # the gate before any launch
SOC=mobile SEEDS=2 ROUNDS=20 BO_ROUNDS=50 BUDGET=300 BO_BATCH=5 \
  LOOP_WARMUP=1000000 LOOP_SIM=2000000 SIM_THREADS=3 \
  python -m loop.search aiml <tag>                          # both arms, sized in rounds
python -m loop.search score results/runs/<soc>_<cell>_<tag>.json
python -m loop.search ceiling <designs> <batch> <trace>...  # the independent ceiling
python -m loop.workloads fetch <url> <out> [prefix_mb]      # a trace, or its prefix
```

**`preview` is the fast feedback loop.** It renders the analyst's prompt and all four
specialists' prompts for the stock design on the selected `SOC` and cell, with no model call and
no search — one simulation of the stock design pays for it, and none once that row is in the
tables. Change what the council reads, run it on each SoC in turn, and the difference between the
chips is the difference between the printouts. A search costs hours before it says the same thing.

Env: `SOC` (default `server`), `SEEDS`, `FIRST_SEED`, `BUDGET` (designs), `ROUNDS` /
`BO_ROUNDS` (per-arm round caps, 0 = size by designs), `BO_BATCH`, `PARALLEL_RUNS`,
`LOOP_WARMUP` / `LOOP_SIM` (the run's fidelity, own tables), `PROBE_WARMUP` / `PROBE_SIM` (the
council's sketch rung), `PAIRS`, `GEOM_LADDER` and `SWEEP` (the council's mechanisms, on unless
`=0`, all recorded in the report's `flags`), `SIM_THREADS` (per workload per run; total concurrency is
that times the workloads times `PARALLEL_RUNS` — keep it near the core count),
`CHAMPSIM_BUILD_SHARE` (8 on the VM), `TABLE_DIR`, `ANALYST_MODEL` (default
`gemini-3.1-pro-preview`, served from `ANALYST_LOCATION=global`), `CACTI` / `CACTI_TEMPLATE`.

**What a design costs.** A ChampSim build is ~6 s. Simulation time is set by the design's
prefetcher traffic, not by the trace: at 1M/2M on the VM a llama2 council round is ~1.5 min and a
3-workload aiml round is several. The VM does roughly twice the Mac. A quad design is two
four-core simulations; the pre-CACTI quad cell measured ~35 designs an hour on the VM. Gemini 3.1
Pro is ~10 s a call, ~0.017 USD, about 7 calls a round.

**Check `gcloud compute instances list` before anything else: the VM must be stopped whenever
nothing runs on it.**

## Next, in this order

1. **The opening lock-in** (the open problem above). It is what costs the council the mobile
   cell, and it is the difference between a loop that wins on one chip and one that is
   cross-SoC. Try the three candidates in order of cost; `preview` tests the third for free.
2. **The quad cell on `D_quad-063f`**: `aiml`, the two mixes, both arms. ~10 h on the VM at these
   round counts; the pre-CACTI run is the rate reference.
3. **`search ceiling` per SoC**, so no cell is scored against a design one of its own arms found.
   Forest-only, no model calls.
4. **The CHIA wiring** (`CHIA_BLOCKS.md`): the ChampSim configuration-space node, `chia/dse/`
   (space, store, report tool, harness), the Vertex generation config, CACTI cache mode, the loop
   in `chia/examples/*` layout with one switch between local calls and `chia_remote`.
5. **Profile the loop with CHIA** (`chia.trace.profiler`) and run one cell with CHIA and without:
   wall clock per round, CPU utilisation, lines of plumbing, a round replayed with `bypass`.
6. **The paper and the README**: per SoC the table by rounds, the final designs, the mechanism,
   the with-and-without-CHIA table, the blocks, the limits stated as measured.

**Postponed**: the case memory and the cross-SoC transfer it would measure (retrieval by k
nearest neighbours on a fixed feature vector, never by chip name); the remaining ablation arms
(without the analyst, without sketches, single agent, all knobs at once); a GP as a second tuned
baseline; the `gs://` and spp_dev pull requests.

**Open: a process metric for the loop, and Jev as its judge.** Jev (TypeSafe's System One
model, `POST https://api.typesafe.ai/v1/systemone`, key `jev_api` in `.env`, 0.042 USD per
million input tokens, output free, under a second a call) answers typed questions over a text
state - a choice among named options, a yes/no probability, a score on a rubric - and returns
probabilities, never generated text. It has graded two earlier runs (llama2 j8, aiml m1) in a
separate session: every number is checked in code, every judgement over free text is one Jev
call, about 0.012 USD and 30 s a run. Findings there: 265 findings across both runs and none
quoting a number the run never produced (12 pinned to the wrong comparison); 72 to 83 % of
specialist proposals cite no measured counter; predictions confirmed about 70 % of the time in
jump rounds and about 32 % in climb rounds, on both chips. The grader scores process, not
outcome: it did not separate the seed that stalled from the one that did not. Four uses, in
order of value:

1. **The metric as a CHIA block.** The CHIA paper names it as open ("metrics should be built
   into agentic platforms"; reward hacking of simulators). Per round, per answer: what the
   argument rests on (a measured counter of this design, a static chip fact such as a latency
   from the ladder, a principle, a deferral) and whether the report shown contradicts it. No
   model call on the search path. The probe showed the trap the criteria must handle: "at
   3 cycles and cannot go lower" was called a measured counter because it holds a number.
2. **A grounding gate in the climb.** One call before a proposal is sketched: does it rest on a
   measured counter of this design? A chip-fact or principle-only answer gets one retry asking
   for the counter. This is the mobile m2 failure - the geometry specialist held nine rounds on
   "the L1D is at 3 cycles for its associativity". A climb change: needs an OK and a measured
   cell.
3. **A contradiction check on the sheet.** A level called fine while it holds the worst mpki on
   the lowest-IPC workload (the m2 stock sheet: L1D fine at 147 mpki) sends the sheet back once.
4. **A prediction ledger.** Each hypothesis classified confirmed, refuted or unsettled against
   the next measurement; each specialist shown its own hit rate.

A grading script for the mobile m2 and server x1 transcripts (two questions per specialist
answer, 260 calls, 547 k tokens, 0.023 USD) ran from a scratch directory outside the tree.
The basis question discriminates: the mobile m2 seed-0 geometry specialist, the one that lost
the cell, argued from a chip fact in 53 % of its answers, against 21 % on seed 1 and 26 % on the
server; prefetch and replacement never do; half of all holds are deferrals. The server's
geometry specialist also leaned on chip facts and that run won, so a chip-fact argument is not
wrong in itself - it is wrong when nobody measures it, which is what the sweep addresses; use 2
is therefore not worth a cell now. The contradiction question as written answered yes on 96 to
100 % of answers on every run and carries no information; it has to become a code check (which
owned level holds the worst mpki on the lowest-IPC workload, and whether the reasoning names
that counter) before the metric is a block. Nothing of this is in `loop/`.

**Open: which baselines the paper sets.** What the loop does today is the hardest version: both
arms from the stock chip at the same budget, seeds, fidelity and parallelism, against a forest
whose hyperparameters are tuned — the comparison ArchGym's result demands. The literature's ladder
is wider and looser: AgentDSE gives random search, a GA and BO 1000 evaluations each and reports
convergence-matched speedup. Read what each prior work gives its baselines, then fix this paper's
ladder and report the ratio against every rung. Random search and a GA are free here — no model
calls. The tuned forest stays the primary baseline: it is the one this audience checks.

## Setup (not in repo)

```bash
git clone --depth 1 https://github.com/ChampSim/ChampSim.git champsim
cd champsim && git submodule update --init && ./vcpkg/bootstrap-vcpkg.sh && ./vcpkg/vcpkg install
git apply ../upstream/0002-champsim-spp-dev-ghr-victim.patch   # or spp_dev designs crash
./config.sh champsim_config.json && make -j8 && cd ..
uv venv --python 3.10 .venv && uv pip install -p .venv/bin/python -r requirements.txt
# CACTI 7. On gcc a plain clone builds; clang needs portable flags and one fix, the Nuca
# constructor's default argument moving from the definition to the declaration.
git clone --depth 1 https://github.com/ucb-bar/cacti.git cacti && cd cacti
sed -i.orig '79s|DeviceType \*dt);|DeviceType *dt = \&(g_tp.peri_global));|' nuca.h
sed -i.orig '46s| = &(g_tp.peri_global)||' nuca.cc && rm nuca.h.orig nuca.cc.orig
make opt -j8 OPT="-O2 -DNTHREADS=8 -Wno-reserved-user-defined-literal" && cd ..
```

A machine with no CACTI build only needs `results/cacti_<node>nm.json` for the nodes it will run;
`loop/chip.py` reads the cache and never shells out when every shape is in it.

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
quota of 32 vCPUs. **Budget: a granted credit** — 175 USD granted, 155 USD left when it was picked
up, paying for both Vertex and VM hours. The credit's id is in the session memory, not here,
because this file is published with the repo. The GCP console is the source of truth.

## Rules of engagement

No dates or deadlines anywhere. **Code and docs state plain facts only** — they describe the
mechanism and the measurements, do not speculate about how the work will be received, and name no
one outside the technical references. Every launch needs the user's explicit OK with cost and time.
The Mac runs under `caffeinate` (which stops idle sleep, **not** lid-close sleep). The VM is
stopped when idle, and a VM chain script's trailing `shutdown` fires the moment its run exits —
keep one chain per session with a single shutdown at the end, or `shutdown -c` after any early
stop. Kill by process group or exact pid, never a broad `pkill`. Code stays lean, one mechanism,
no version archaeology; domain decisions are surfaced with options and left to the user. Deleted
material is in git history, never in the tree.
